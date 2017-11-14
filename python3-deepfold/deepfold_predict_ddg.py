import re

import Bio
import Bio.PDB

from deepfold_markov_model import MarkovModel
# from deepfold_model_ddg_snapshot import *
from deepfold_model import *


class MissingResidueError(Exception):
    pass

def read_ddg_csv(filename):
    results = []
    for line in open(filename):

        line = line.strip()
        if len(line) == 0 or line[0] == "#":
            continue

        split_line = line.split(',')
        
        pdb_id = split_line[1]
        chain_id = split_line[2][0]
        ddg = float(split_line[3])

        mutation_entries = split_line[2].split('_')
        mutations = []
        for mutation_entry in mutation_entries:
            mutation_chain_id = mutation_entry[0]

            # Check that mutation chain id matches chain id (first one does by definition)
            assert(mutation_chain_id == chain_id)
            
            mutation_entry = mutation_entry[2:]
            wt, res_id, mutant = mutation_entry.split(' ')
            # if res_id.isdigit():
            #     res_num = int(res_id)
            # else:
            #     res_num = int(re.match("\d+", res_id).group(0))
                
            mutations.append([wt, res_id, mutant])
        
        results.append([(pdb_id+chain_id).upper(), mutations, ddg])
    return results

def calc_ddg(prob_wt, prob_mutant, baseline_prob, wt_aa_index, mutant_aa_index):
    prob_wt_base = baseline_prob[wt_aa_index]
    prob_mutant_base = baseline_prob[mutant_aa_index]

    dg_wt = np.log(prob_wt) - np.log(prob_wt_base)
    dg_mutant = np.log(prob_mutant) - np.log(prob_mutant_base)

    predicted_ddg = dg_wt - dg_mutant

    return prob_wt_base, prob_mutant_base, dg_wt, dg_mutant, predicted_ddg
    

def predict_ddg(model, markov_model_baseline, high_res_features_input_dir, low_res_features_input_dir, pdb_dir,
                pdb_id, mutations, ddg,
                include_high_res_model=True,
                include_low_res_model=False,
                max_batch_size=25):

    chain_id = None
    if len(pdb_id) == 5:
        chain_id = pdb_id[4]
        pdb_id = pdb_id[:4]
    
    exclude_at_center = ["aa_one_hot_w_unobserved"]
    
    batch_factory = BatchFactory()
    
    if high_res_features_input_dir is not None:
        high_res_protein_feature_filename = os.path.join(high_res_features_input_dir, pdb_id+"_protein_features.npz")
        high_res_residue_feature_filename = os.path.join(high_res_features_input_dir, pdb_id+"_residue_features.npz")
        if not (os.path.exists(high_res_protein_feature_filename) and
                os.path.exists(high_res_residue_feature_filename)):
            raise IOError(pdb_id + ": Input files not found")

        batch_factory.add_data_set("high_res",
                                   [high_res_protein_feature_filename],
                                   [high_res_residue_feature_filename])
        
    if low_res_features_input_dir is not None:
        low_res_protein_feature_filename = os.path.join(low_res_features_input_dir, pdb_id+"_protein_features.npz")
        low_res_residue_feature_filename = os.path.join(low_res_features_input_dir, pdb_id+"_residue_features.npz")
        if not (os.path.exists(low_res_protein_feature_filename) and
                os.path.exists(low_res_residue_feature_filename)):
            raise IOError("Input files not found")
    
        batch_factory.add_data_set("low_res",
                                   [low_res_protein_feature_filename],
                                   [low_res_residue_feature_filename],
                                   exclude_at_center = exclude_at_center)

        
    batch_factory.add_data_set("model_output",
                               [high_res_protein_feature_filename],
                               key_filter=["aa_one_hot"])

    batch_factory.add_data_set("chain_ids",
                               [high_res_protein_feature_filename],
                               key_filter=["chain_ids"])

    batch, sub_batch_sizes = batch_factory.next(batch_factory.data_size(),
                                                subbatch_max_size=max_batch_size,
                                                enforce_protein_boundaries=True)

    chain_ids = batch['chain_ids']
    
    # Extract index of first residue from PDB - and attempt to use this as offset into model
    pdb_parser = Bio.PDB.PDBParser()
    structure = pdb_parser.get_structure(pdb_id, os.path.join(pdb_dir, pdb_id+".pdb"))


    prob_wt_list = []
    prob_mutant_list = []

    prob_wt_base_m1_list = []
    prob_mutant_base_m1_list = []
    ddg_wt_m1_list = []
    ddg_mutant_m1_list = []
    pred_ddg_m1_list = []

    prob_wt_base_m0_list = []
    prob_mutant_base_m0_list = []
    ddg_wt_m0_list = []
    ddg_mutant_m0_list = []
    pred_ddg_m0_list = []
    
    for mutation in mutations:
    
        wt, res_id, mutant = mutation

        icode = ' '
        if res_id.isdigit():
            res_index = int(res_id)
        else:
            res_index = re.match("\d+", res_id).group(0)
            icode = res_id.replace(res_index, "")
            res_index = int(res_index)


        # ppb = Bio.PDB.PPBuilder()
        # pps = list(ppb.build_peptides(structure))

        # print "mutation: ", mutation

        try:
            # Extract residue in PDB
            pdb_res = structure[0][chain_id][(' ', res_index, icode)]
        except KeyError:
            raise MissingResidueError("Missing residue: " + str((' ', res_index, icode)) + ". Perhaps a removed HETATM?")

        # print structure[0][chain_id][(' ', res_index, 'B')]
        # for res in structure[0][chain_id]:
        #     print res
        #     print res.id
        # print pdb_res.id

        # Check that PDB and mutation record agree on wt
        assert(Bio.PDB.Polypeptide.three_to_one(pdb_res.get_resname()) == wt)

        chain_res_index = structure[0][chain_id].get_list().index(pdb_res)

        # pdb_chain_index_offset = structure[0][chain_id].get_list()[0].get_id()[1]
        # print "pdb_chain_index_offset: ", pdb_chain_index_offset

        # print "res_index: ", res_index

        # chain_res_index = res_index - pdb_chain_index_offset
        # print "chain_res_index: ", chain_res_index

        # pdb_chain_ids = sorted([chain.id for chain in structure[0]])
        # pdb_initial_res_index = structure[0][pdb_chain_ids[0]].get_id()[1]
        # pdb_initial_res_index = pps[0][0].get_id()[1]

        model_chain_index_offset = np.nonzero(chain_ids==chain_id)[0][0]
        # print "model_chain_index_offset: ", model_chain_index_offset

        model_res_index = model_chain_index_offset + chain_res_index
        # print "model_res_index: ", model_res_index

        model_sequence = ""
        for index in np.argmax(batch["model_output"], axis=1):
            if index < 20:
                model_sequence += Bio.PDB.Polypeptide.index_to_one(index)
            else:
                model_sequence += 'X'            
        # # model_sequence = "".join([Bio.PDB.Polypeptide.index_to_one(index) for index in np.argmax(batch["model_output"], axis=1) if index < 20 else])
        # model_res_index = res_index-pdb_initial_res_index
        # print model_sequence
        # print model_sequence[model_res_index], wt
        assert(model_sequence[model_res_index] == wt)

        wt_aa_index = Bio.PDB.Polypeptide.one_to_index(wt)
        mutant_aa_index = Bio.PDB.Polypeptide.one_to_index(mutant)

        #aa_dist = model.infer(batch, sub_batch_sizes)
        #aa_dist_at_res = aa_dist[model_res_index]

        res_batch = dict(list(zip(list(batch.keys()), get_batch(model_res_index, model_res_index+1, *list(batch.values())))))
        res_sub_batch_sizes = [1]
        aa_dist_at_res = model.infer(res_batch, res_sub_batch_sizes)[0]

        prob_wt = aa_dist_at_res[wt_aa_index]
        prob_mutant = aa_dist_at_res[mutant_aa_index]

        baseline_prob_markov1 = markov_model_baseline.marginal_prob_at_index(model_sequence, model_res_index)
        # baseline_prob_markov0 = markov_model_baseline.get_stationary_distribution()
        baseline_prob_markov0 = markov_model_baseline.get_frequencies()

        prob_wt_base_m1, prob_mutant_base_m1, ddg_wt_m1, ddg_mutant_m1, pred_ddg_m1 = calc_ddg(prob_wt, prob_mutant, baseline_prob_markov1, wt_aa_index, mutant_aa_index) 
        prob_wt_base_m0, prob_mutant_base_m0, ddg_wt_m0, ddg_mutant_m0, pred_ddg_m0 = calc_ddg(prob_wt, prob_mutant, baseline_prob_markov0, wt_aa_index, mutant_aa_index) 

        prob_wt_list.append(prob_wt)
        prob_mutant_list.append(prob_mutant)
        
        prob_wt_base_m1_list.append(prob_wt_base_m1)
        prob_mutant_base_m1_list.append(prob_mutant_base_m1)
        ddg_wt_m1_list.append(ddg_wt_m1)
        ddg_mutant_m1_list.append(ddg_mutant_m1)
        pred_ddg_m1_list.append(pred_ddg_m1)
        
        prob_wt_base_m0_list.append(prob_wt_base_m0)
        prob_mutant_base_m0_list.append(prob_mutant_base_m0)
        ddg_wt_m0_list.append(ddg_wt_m0)
        ddg_mutant_m0_list.append(ddg_mutant_m0)
        pred_ddg_m0_list.append(pred_ddg_m0)

    prob_wt_list = np.array(prob_wt_list)
    prob_mutant_list = np.array(prob_mutant_list)

    prob_wt_base_m1_list = np.array(prob_wt_base_m1_list)
    prob_mutant_base_m1_list = np.array(prob_mutant_base_m1_list)
    ddg_wt_m1_list = np.array(ddg_wt_m1_list)
    ddg_mutant_m1_list = np.array(ddg_mutant_m1_list)
    pred_ddg_m1_list = np.array(pred_ddg_m1_list)

    prob_wt_base_m0_list = np.array(prob_wt_base_m0_list)
    prob_mutant_base_m0_list = np.array(prob_mutant_base_m0_list)
    ddg_wt_m0_list = np.array(ddg_wt_m0_list)
    ddg_mutant_m0_list = np.array(ddg_mutant_m0_list)
    pred_ddg_m0_list = np.array(pred_ddg_m0_list)
        
    # prob_wt_base_markov1 = baseline_prob_markov1[wt_aa_index]
    # prob_mutant_base_markov1 = baseline_prob_markov1[mutant_aa_index]
    
    # prob_wt_base_markov0 = baseline_prob_markov0[wt_aa_index]
    # prob_mutant_base_markov0 = baseline_prob_markov0[mutant_aa_index]
    
    # dg_wt = np.log(prob_wt) - np.log(prob_wt_base_markov1)
    # dg_mutant = np.log(prob_mutant) - np.log(prob_mutant_base_markov1)
    
    # dg_wt = np.log(prob_wt) - np.log(prob_wt_base_markov0)
    # dg_mutant = np.log(prob_mutant) - np.log(prob_mutant_base_markov0)
    
    # predicted_ddg = dg_wt - dg_mutant

    # print pdb_id, "".join(map(str,mutation)), prob_wt, prob_mutant, np.log(prob_wt)-np.log(prob_mutant), prob_wt_base_m0, prob_mutant_base_m0, ddg_wt_m0, ddg_mutant_m0, pred_ddg_m0, prob_wt_base_m1, prob_mutant_base_m1, ddg_wt_m1, ddg_mutant_m1, pred_ddg_m1, ddg
    print(pdb_id, "".join(map(str,mutation)), np.prod(prob_wt_list), np.prod(prob_mutant_list), np.log(np.prod(prob_wt_list))-np.log(np.prod(prob_mutant_list)), np.prod(prob_wt_base_m0_list), np.prod(prob_mutant_base_m0_list), np.sum(ddg_wt_m0_list), np.sum(ddg_mutant_m0_list), np.sum(pred_ddg_m0_list), np.prod(prob_wt_base_m1_list), np.prod(prob_mutant_base_m1_list), np.sum(ddg_wt_m1_list), np.sum(ddg_mutant_m1_list), np.sum(pred_ddg_m1_list), ddg)
    

if __name__ == '__main__':
    import deepfold_model

    from Deepfold.Models import models
    from utils import str2bool
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--high-res-input-dir", dest="high_res_features_input_dir",
                        help="Location of input files containing high-res features")
    parser.add_argument("--low-res-input-dir", dest="low_res_features_input_dir",
                        help="Location of input files containing low-res features")
    parser.add_argument("--include-high-res-model", dest="include_high_res_model",
                        type=str2bool, nargs='?', const=True, default="True",
                        help="Whether to include atomic-resolution part of model")
    parser.add_argument("--include-low-res-model", dest="include_low_res_model",
                        type=str2bool, nargs='?', const=True, default="False",
                        help="Whether to include low resolution part of model")
    parser.add_argument("--pdb-dir", dest="pdb_dir",
                        help="Location of PDB files (only used to check residue-index offset)")
    parser.add_argument("--max-batch-size", dest="max_batch_size",
                        help="Maximum batch size used during model evaluation", type=int, default=25)
    parser.add_argument("--model-checkpoint-path", dest="model_checkpoint_path",
                        help="Where to dump/read model checkpoints", default="models")
    parser.add_argument("--ddg-csv-filename", dest="ddg_csv_filename",
                        help="CSV file containing ddG data")
    parser.add_argument("--markov-model-filename", dest="markov_model_filename",
                        help="Parameter file for Markov model")
    parser.add_argument("--model", choices=['default']+list(models.keys()), default="default",
                        help="Which model definition to use (default: %(default)s)")
    parser.add_argument("--step", type=int, default=None,
                        help="Which checkpoint file to use (default: %(default)s)")
    options = parser.parse_args()

    print("# Options")
    for key, value in sorted(vars(options).items()):
        print(key, "=", value)
    
    if options.model == "default":
        model = deepfold_model.Model(r_size_high_res         = 24,
                                     theta_size_high_res     = 76,
                                     phi_size_high_res       = 151,
                                     channels_high_res       = 2,
                                     r_size_low_res          = 6,
                                     theta_size_low_res      = 19,
                                     phi_size_low_res        = 38,
                                     channels_low_res        = 43,
                                     output_size             = 21,
                                     max_batch_size          = options.max_batch_size,
                                     max_gradient_batch_size = options.max_batch_size,
                                     include_high_res_model  = options.include_high_res_model,
                                     include_low_res_model   = options.include_low_res_model,
                                     model_checkpoint_path   = options.model_checkpoint_path)
    if options.model.startswith("Spherical"):
        model = models[options.model](r_size_high_res         = 24,
                                      theta_size_high_res     = 76,
                                      phi_size_high_res       = 151,
                                      channels_high_res       = 2,
                                      output_size             = 21,
                                      optimize_using_lbfgs    = False,
                                      reg_fact                = 0.001,
                                      learning_rate           = 0.0001,
                                      model_checkpoint_path   = "/dev/null",
                                      max_to_keep             = 0)
    elif options.model.startswith("CubedSphere"):
        model = models[options.model](patches_size_high_res   = 6,
                                      r_size_high_res         = 24,
                                      xi_size_high_res        = 38,
                                      eta_size_high_res       = 38,
                                      channels_high_res       = 2,
                                      output_size             = 21,
                                      optimize_using_lbfgs    = False,
                                      reg_fact                = 0.001,
                                      learning_rate           = 0.0001,
                                      model_checkpoint_path   = "/dev/null",
                                      max_to_keep             = 0)
    elif options.model.startswith("CartesianHighres"):
        model = models[options.model](x_size_high_res         = 60,
                                      y_size_high_res         = 60,
                                      z_size_high_res         = 60,
                                      channels_high_res       = 2,
                                      output_size             = 21,
                                      optimize_using_lbfgs    = False,
                                      reg_fact                = 0.001,
                                      learning_rate           = 0.0001,
                                      model_checkpoint_path   = "/dev/null",
                                      max_to_keep             = 0)
    elif options.model.startswith("Cartesian"):
        model = models[options.model](x_size_high_res         = 48,
                                      y_size_high_res         = 48,
                                      z_size_high_res         = 48,
                                      channels_high_res       = 2,
                                      output_size             = 21,
                                      optimize_using_lbfgs    = False,
                                      reg_fact                = 0.001,
                                      learning_rate           = 0.0001,
                                      model_checkpoint_path   = "/dev/null",
                                      max_to_keep             = 0)
    else:
        raise argparse.ArgumentTypeError("Deep model not supported: %s" % parser.deep_mode)
        

    model.restore(options.model_checkpoint_path, step=options.step)

    markov_model = MarkovModel()
    markov_model.restore(options.markov_model_filename)
    
    mutations = read_ddg_csv(options.ddg_csv_filename)

    for mutation in mutations:
        # if len(mutation[1]) > 1:
        #     print >> sys.stderr, "Skipping: ", mutation, ". 2-site mutations excluded for now..."
        #     continue

        try:
            predict_ddg(model=model,
                        markov_model_baseline = markov_model,
                        high_res_features_input_dir = options.high_res_features_input_dir,
                        low_res_features_input_dir = options.low_res_features_input_dir,
                        pdb_dir = options.pdb_dir,
                        include_high_res_model = options.include_high_res_model,
                        include_low_res_model = options.include_low_res_model,
                        pdb_id = mutation[0],
                        mutations = mutation[1],
                        ddg = mutation[2])
        except MissingResidueError as e:
            print("SKIPPING DUE TO MissingResidueError: ", e, file=sys.stderr)
        # except IOError as e:
        #     print >> sys.stderr, "IOError - skipping: ", mutation, e

        # except AssertionError as e:
        #     print >> sys.stderr, "AssertionError - skipping: ", mutation, e

        # except IndexError as e:
        #     print >> sys.stderr, "IndexError - skipping: ", mutation, e

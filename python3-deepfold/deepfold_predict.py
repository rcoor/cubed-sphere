import Bio
import Bio.PDB
from deepfold_model import *

def predict(model, high_res_features_input_dir, pdb_dir, pdb_id, chain_id, res_index):

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

    batch_factory.add_data_set("chain_ids",
                               [high_res_protein_feature_filename],
                               key_filter=["chain_ids"])
        
    batch_factory.add_data_set("model_output",
                               [high_res_protein_feature_filename],
                               key_filter=["aa_one_hot"])

    batch, sub_batch_sizes = batch_factory.next(batch_factory.data_size(),
                                                subbatch_max_size=10,
                                                enforce_protein_boundaries=True)

    chain_ids = batch['chain_ids']
    
    pdb_parser = Bio.PDB.PDBParser()
    structure = pdb_parser.get_structure(pdb_id, os.path.join(pdb_dir, pdb_id+".pdb"))

    pdb_res = structure[0][chain_id].get_list()[res_index]

    pdb_res_value = Bio.PDB.Polypeptide.three_to_one(pdb_res.get_resname())
    print("The original amino acid at this position is: ", pdb_res_value)
    
    model_sequence = ""
    for index in np.argmax(batch["model_output"], axis=1):
        if index < 20:
            model_sequence += Bio.PDB.Polypeptide.index_to_one(index)
        else:
            model_sequence += 'X'

    chain_res_index = structure[0][chain_id].get_list().index(pdb_res)
    model_chain_index_offset = np.nonzero(chain_ids==chain_id)[0][0]
    model_res_index = model_chain_index_offset + chain_res_index

            
    print(pdb_res_value)
    print(model_sequence)
    print(res_index)
    assert(pdb_res_value == model_sequence[model_res_index])

    res_batch = dict(list(zip(list(batch.keys()), get_batch(model_res_index, model_res_index+1, *list(batch.values())))))
    res_sub_batch_sizes = [1]
    aa_dist_at_res = model.infer(res_batch, res_sub_batch_sizes)[0]

    for item in sorted(zip(aa_dist_at_res, Bio.PDB.Polypeptide.aa1), reverse=True):
        print(item[1], item[0])
    

if __name__ == '__main__':
    import deepfold_model

    from deepfold_models import models
    from utils import str2bool
    import argparse
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--high-res-input-dir", dest="high_res_features_input_dir",
                        help="Location of input files containing high-res features")
    parser.add_argument("--pdb-dir", dest="pdb_dir",
                        help="Location of PDB files (only used to check residue-index offset)")
    parser.add_argument("--pdb-id", dest="pdb_id",
                        help="PDB ID")
    parser.add_argument("--chain-id", dest="chain_id",
                        help="PDB chain ID")
    parser.add_argument("--res-index", dest="res_index", type=int,
                        help="PDB residue index")
    parser.add_argument("--max-batch-size", dest="max_batch_size",
                        help="Maximum batch size used during model evaluation", type=int, default=25)
    parser.add_argument("--model-checkpoint-path", dest="model_checkpoint_path",
                        help="Where to dump/read model checkpoints", default="models")
    parser.add_argument("--model", choices=['default']+list(models.keys()), default="default",
                        help="Which model definition to use (default: %(default)s)")
    parser.add_argument("--step", type=int, default=None,
                        help="Which checkpoint file to use (default: %(default)s)")
    options = parser.parse_args()

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

    predict(model=model,
            high_res_features_input_dir = options.high_res_features_input_dir,
            pdb_dir = options.pdb_dir,
            pdb_id = options.pdb_id,
            chain_id = options.chain_id,
            res_index = options.res_index)


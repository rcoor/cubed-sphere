from delta_g_prepper import DeltaPrepper
from delta_predict import GetIndividualProteins
import tensorflow as tf
import pandas as pd
import numpy as np
from CNN import CNNModel
from Bio.PDB import Polypeptide
from Bio.PDB import PDBList
import Bio
import os
import re
from batch_factory.deepfold_batch_factory import BatchFactory

# Set flags
flags = tf.app.flags

flags.DEFINE_string(
    "input_delta", "/Users/thorn/Documents/projects/cubed-sphere/data/ddgs/kellogg.csv", "Input path")
flags.DEFINE_string("input_features", "/Users/thorn/Documents/projects/cubed-sphere/data/atomistic_features_cubed_sphere_ddg",
                    "Fraction of data set aside for testing")
flags.DEFINE_string("pdb_dir", "/Users/thorn/Documents/projects/cubed-sphere/data/PDB/", "Path to PDB files")

FLAGS = flags.FLAGS


class MissingResidueError(Exception):
    pass

def prepare_batch(input_dir_features, pdb_id):
    input_dir_residue = input_dir_features

    batch_factory = BatchFactory()

    # make sure the protein isn't shuffled
    #batch_factory.shuffle = False

    protein_feature_filename = os.path.join(
        input_dir_features, pdb_id + "_protein_features.npz")
    residue_feature_filename = os.path.join(
        input_dir_residue, pdb_id + "_residue_features.npz")
    if not (os.path.exists(protein_feature_filename) and
            os.path.exists(residue_feature_filename)):
        raise IOError("Input files not found")

    batch_factory.add_data_set("data",
                               [protein_feature_filename],
                               [residue_feature_filename])

    batch_factory.add_data_set("model_output",
                               [protein_feature_filename],
                               key_filter=["aa_one_hot"])

    ''' batch_factory.add_data_set("chain_ids",
                               [protein_feature_filename],
                               key_filter=["chain_ids"]) '''

    return batch_factory


def get_aa_probs(pdb_id, wildtype, mutation, position):
    getIndividualProteins = GetIndividualProteins()
    file_path = FLAGS.input_features

    # Get protein from protein features and add to batch_factory
    batchFactory = create_batch(file_path, pdb_id)

    # Print PDB_i, mutation in the form: wildtype:position:mutation
    print("PDB: {}, mutation: {}{}{}".format(
        pdb_id, wildtype, position, mutation))
    print("size of data: {}".format(batchFactory.data_size()))

    # Get next batch from batch_factory
    batch, _ = batchFactory.next(int(batchFactory.data_size()))
    # Locate the amino acid in the chain based on the mutational position
    aa_data = batch['data'][int(position) - 1]

    # Initialize graph and start session
    with tf.Graph().as_default():
        # Load the network model (convolutional neural network)
        model = CNNModel()
        session = tf.Session()
        model.batch_size = 1

        # The model initializes based on trained model parameters.
        # The model is then used to infer on the data– a softmax is added to
        # give appropriate probabilities.
        logits = model.predict(session, [aa_data])

        # Clean the objects to release memory
        del batchFactory, getIndividualProteins
        # Return inferred logits
        return logits

def predict_ddg(input_dir_features, pdb_id, mutations):
    mutation_dataframe = []

    chain_id = None
    if len(pdb_id) == 5:
        chain_id = pdb_id[4]
        pdb_id = pdb_id[:4]

    batch_factory = prepare_batch(
        input_dir_features=input_dir_features, pdb_id=pdb_id)

    print(batch_factory.data_size())
    batch, _ = batch_factory.next(batch_factory.data_size())

    # Extract index of first residue from PDB - and attempt to use this as
    # offset into model
    mmcif_parser = Bio.PDB.MMCIFParser()

    cif_path = os.path.join(FLAGS.pdb_dir, pdb_id.lower() + ".cif")

    if not os.path.exists(cif_path):
        pdbl = PDBList()
        pdbl.retrieve_pdb_file(pdb_id, pdir="./data/PDB/")

    structure = mmcif_parser.get_structure(pdb_id, cif_path)

    # Loop through all rows
    for _, mutation in mutations.iterrows():
        #mutation[['PDBFileID','chain','wildtype', 'mutation', 'position']]

        wt, res_id, mutant, chain = mutation[['wildtype', 'position', 'mutation', 'chain']]
        print(wt, res_id, mutant)
        icode = ' '
        if res_id.isdigit():
            res_index = int(res_id)
        else:
            res_index = re.match("\d+", res_id).group(0)
            icode = res_id.replace(res_index, "")
            res_index = int(res_index)

        try:
            # Extract residue in PDB
            pdb_res = structure[0][chain][(' ', res_index, icode)]
        except KeyError:
            raise MissingResidueError("Missing residue: " + str((' ', res_index, icode)) + ". Perhaps a removed HETATM?")

        # Check that PDB and mutation record agree on wt
        assert(Bio.PDB.Polypeptide.three_to_one(pdb_res.get_resname()) == wt)

        chain_res_index = structure[0][chain].get_list().index(pdb_res)

        try:
            mutant_index = Bio.PDB.Polypeptide.one_to_index(mutant)
            wt_index = Bio.PDB.Polypeptide.one_to_index(wt)

            with tf.Graph().as_default():
                model = CNNModel()
                logits = model.predict(tf.Session(), [batch['data'][res_index - 1]])[0][0]
                # wildtype and mutant probability:
                print("Wildtype prob: {} and mutation prob: {}.".format(logits[wt_index], logits[mutant_index]))
                mutation['w_prob'] = logits[wt_index]
                mutation['m_prob'] = logits[mutant_index]
                #print(pd.DataFrame(mutation).transpose())

            mutation_dataframe.append(pd.DataFrame(mutation).transpose())

        except Exception:
            continue


        ''' Her er det svært uden chain_ids '''
        #model_chain_index_offset = np.nonzero(chain_ids==chain_id)[0][0]

        #model_res_index = model_chain_index_offset + chain_res_index
    if len(mutation_dataframe) > 0:
         return pd.concat(mutation_dataframe)
    return []
    # TODO: forklar!

    ''' model_sequence = ""
    for index in np.argmax(batch["model_output"], axis=1):
        if index < 20:
            model_sequence += Bio.PDB.Polypeptide.index_to_one(index)
        else:
            model_sequence += 'X'
    #assert(model_sequence[model_res_index] == wt)

    wt_aa_index = Bio.PDB.Polypeptide.one_to_index(wt)
    mutant_aa_index = Bio.PDB.Polypeptide.one_to_index(mutant) '''




    ''' wt_aa_index = Bio.PDB.Polypeptide.one_to_index(wt)
    mutant_aa_index = Bio.PDB.Polypeptide.one_to_index(mutant)

    res_batch = dict(zip(batch.keys(), get_batch(model_res_index, model_res_index+1, *batch.values())))
    res_sub_batch_sizes = [1]
    aa_dist_at_res = model.infer(res_batch, res_sub_batch_sizes)[0]

    prob_wt = aa_dist_at_res[wt_aa_index]
    prob_mutant = aa_dist_at_res[mutant_aa_index] '''



# Prepare the DataPrepper
dp = DeltaPrepper()
# load data into a data frame and extract mutation position, chain, wildtype and mutant residue
dp.load_data(FLAGS.input_delta)

if not dp.dataframe.empty:
    # Create empty columns for computed probabilities
    sLength = len(dp.dataframe['mutation'])
    dp.dataframe['w_prob'] = pd.Series(np.random.randn(sLength), index=dp.dataframe.index)
    dp.dataframe['m_prob'] = pd.Series(np.random.randn(sLength), index=dp.dataframe.index)



    # dp.dataframe = dp.dataframe.tail(10)

complete_mutations_dataframe = []
for pdb in dp.dataframe['PDBFileID'].unique():
    mutations = dp.dataframe[dp.dataframe['PDBFileID'].str.contains(pdb) == True]
    print(pdb)
    try:
        mutation_dataframe = predict_ddg(FLAGS.input_features, pdb_id=pdb, mutations=mutations)
        print(mutation_dataframe)
        complete_mutations_dataframe.append(pd.DataFrame(mutation_dataframe))
    except MissingResidueError as e:
        print("SKIPPING DUE TO MissingResidueError: ", e)

df = pd.concat(complete_mutations_dataframe)
df.to_pickle('./{}.pickle'.format(os.path.basename(FLAGS.input_delta).split('.')[0]))
df.to_csv('./{}.csv'.format(os.path.basename(FLAGS.input_delta).split('.')[0]))

''' # Prepare the DataPrepper
dp = DeltaPrepper()
# load data into a data frame and extract mutation position, chain, wildtype and mutant residue
dp.load_data(FLAGS.input_delta)

if not dp.dataframe.empty:
    sLength = len(dp.dataframe['mutation'])
    dp.dataframe['w_prob'] = pd.Series(np.random.randn(sLength), index=dp.dataframe.index)
    dp.dataframe['m_prob'] = pd.Series(np.random.randn(sLength), index=dp.dataframe.index)

    print(dp.dataframe)
    dp.dataframe = dp.dataframe.iloc[6:]
    for index, row in dp.dataframe.iterrows():


        # get the computed list of probabilities
        prob_array = get_aa_probs(*row[['PDBFileID','wildtype', 'mutation', 'position']])[0][0]

        # get indexes of residues
        m_i = Polypeptide.one_to_index(row['mutation'])
        w_i = Polypeptide.one_to_index(row['wildtype'])

        # add specific probabilites of mutation and wildtype in the dataframe
        dp.dataframe['w_prob'][index] = prob_array[w_i]
        dp.dataframe['m_prob'][index] = prob_array[m_i]

    dp.dataframe.to_pickle('./{}.pickle'.format(os.path.basename(FLAGS.input_delta)))
    dp.dataframe.to_csv('./{}.csv'.format(os.path.basename(FLAGS.input_delta)))

 '''

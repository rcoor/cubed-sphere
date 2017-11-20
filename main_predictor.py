from delta_g_prepper import DeltaPrepper
from delta_predict import GetIndividualProteins
import tensorflow as tf
import pandas as pd
import numpy as np
from CNN import CNNModel
from Bio.PDB import Polypeptide
import os

from batch_factory.deepfold_batch_factory import BatchFactory

# Set flags
flags = tf.app.flags

flags.DEFINE_string(
    "input_delta", "/Users/thorn/Documents/projects/cubed-sphere/data/ddgs/kellogg.csv", "Input path")
flags.DEFINE_string("input_features", "/Users/thorn/Documents/projects/cubed-sphere/data/atomistic_features_cubed_sphere_ddg",
                    "Fraction of data set aside for testing")

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

    batch_factory.add_data_set("chain_ids",
                               [protein_feature_filename],
                               key_filter=["chain_ids"])

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

    chain_id = None
    if len(pdb_id) == 5:
        chain_id = pdb_id[4]
        pdb_id = pdb_id[:4]

    batch_factory = prepare_batch(
        input_dir_features=input_dir_features, pdb_id=pdb_id)

    batch, sub_batch_sizes = batch_factory.next(batch_factory.data_size(), subbatch_max_size=25, enforce_protein_boundaries=True)

    # Extract index of first residue from PDB - and attempt to use this as
    # offset into model
    pdb_parser = Bio.PDB.PDBParser()
    structure = pdb_parser.get_structure(
        pdb_id, os.path.join(pdb_dir, pdb_id + ".pdb"))

    # Loop through all rows
    for _, mutation in mutations.iterrows():
        *mutation[['PDBFileID','chain','wildtype', 'mutation', 'position']]

        wt, res_id, mutant = *mutation[['wildtype', 'position', 'mutation']]

        icode = ' '
        if res_id.isdigit():
            res_index = int(res_id)
        else:
            res_index = re.match("\d+", res_id).group(0)
            icode = res_id.replace(res_index, "")
            res_index = int(res_index)


    try:
        # Extract residue in PDB
        pdb_res = structure[0][chain_id][(' ', res_index, icode)]
    except KeyError:
        raise MissingResidueError("Missing residue: " + str((' ', res_index, icode)) + ". Perhaps a removed HETATM?")

    # Check that PDB and mutation record agree on wt
    assert(Bio.PDB.Polypeptide.three_to_one(pdb_res.get_resname()) == wt)




# Prepare the DataPrepper
dp = DeltaPrepper()
# load data into a data frame and extract mutation position, chain, wildtype and mutant residue
dp.load_data(FLAGS.input_delta)

if not dp.dataframe.empty:
    # Create empty columns for computed probabilities
    sLength = len(dp.dataframe['mutation'])
    dp.dataframe['w_prob'] = pd.Series(np.random.randn(sLength), index=dp.dataframe.index)
    dp.dataframe['m_prob'] = pd.Series(np.random.randn(sLength), index=dp.dataframe.index)

    # Only take the head
    mutations = dp.dataframe.iloc[6:]

    print(*mutation[['PDBFileID','chain','wildtype', 'mutation', 'position']])


try:
    predict_ddg(FLAGS.input_features, pdb_id="1ANK", mutations=mutations)

except MissingResidueError as e:
    print("SKIPPING DUE TO MissingResidueError: ", e)


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

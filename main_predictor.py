from delta_g_prepper import DeltaPrepper
from delta_predict import GetIndividualProteins
import tensorflow as tf
import pandas as pd
import numpy as np
from CNN import CNNModel
from Bio.PDB import Polypeptide
import os

# Set flags
flags = tf.app.flags

flags.DEFINE_string("input_delta", "/Users/thorn/Documents/projects/cubed-sphere/data/ddgs/kellogg.csv", "Input path")
flags.DEFINE_string("input_features", "/Users/thorn/Documents/projects/cubed-sphere/data/atomistic_features_cubed_sphere_ddg","Fraction of data set aside for testing")

FLAGS = flags.FLAGS

getIndividualProteins = GetIndividualProteins()
def get_aa_probs(pdb_id, wildtype, mutation, position):
    print(pdb_id, wildtype, mutation, position)
    file_path = FLAGS.input_features
    batchFactory = getIndividualProteins.create_batch(file_path, pdb_id, 1)
    batch, _ = batchFactory.next(batchFactory.data_size())
    aa_data = batch['data'][int(position)-1]

    with tf.Graph().as_default():
        model = CNNModel()
        session = tf.Session()
        model.batch_size = 1

        return model.predict(session, [aa_data])


# Prepare the DataPrepper
dp = DeltaPrepper()
# load data into a data frame and extract mutation position, chain, wildtype and mutant residue
dp.load_data(FLAGS.input_delta)

if not dp.dataframe.empty:
    sLength = len(dp.dataframe['mutation'])
    dp.dataframe['w_prob'] = pd.Series(np.random.randn(sLength), index=dp.dataframe.index)
    dp.dataframe['m_prob'] = pd.Series(np.random.randn(sLength), index=dp.dataframe.index)

    print(dp.dataframe)
    for index, row in dp.dataframe.iterrows():

        # get the computed list of probabilities
        prob_array = get_aa_probs(*row[['PDBFileID','wildtype', 'mutation', 'position']])[0][0]

        # get indexes of residues
        m_i = Polypeptide.one_to_index(row['mutation'])
        w_i = Polypeptide.one_to_index(row['wildtype'])

        # add specific probabilites of mutation and wildtype in the dataframe
        dp.dataframe['w_prob'][index] = prob_array[w_i]
        dp.dataframe['m_prob'][index] = prob_array[m_i]
        dp.dataframe.to_pickle('./{}_p.pickle'.format(os.path.basename(FLAGS.input_delta).split('.')[0]))
        dp.dataframe.to_csv('./{}_p.csv'.format(os.path.basename(FLAGS.input_delta).split('.')[0]))
    dp.dataframe.to_pickle('./{}.pickle'.format(os.path.basename(FLAGS.input_delta)))
    dp.dataframe.to_csv('./{}.csv'.format(os.path.basename(FLAGS.input_delta)))
    #print(dp.dataframe)


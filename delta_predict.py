import re

import Bio
import Bio.PDB
import os
from batch_factory.deepfold_batch_factory import BatchFactory

class GetIndividualProteins:
    def __init__(self):
        self

    def create_batch(self, input_dir, pdb_id, max_batch_size=25):

        if len(pdb_id) == 5:
            chain_id = pdb_id[4]
            pdb_id = pdb_id[:4]

        batch_factory = BatchFactory()

        # make sure the protein isn't shuffled
        #batch_factory.shuffle = False

        protein_feature_filename = os.path.join(input_dir, pdb_id+"_protein_features.npz")
        residue_feature_filename = os.path.join(input_dir, pdb_id+"_residue_features.npz")
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

class PrepareDeltaDataframe:
    ''' Generate delta g predictions from datasheets '''

    def __init__(self):
        self.input = None

    def load_data(self):
        if self.input == None:
            raise ValueError("Input must be defined.")

        # Load curated dataset
        df = pd.read_csv(self.input, skiprows=21)

        # Print a sample
        print(df[['PDBFileID', 'Mutations','DDG']])

        # Convert an array of length 4 to a pandas dataframe
        split_mut = lambda x: pd.Series({'chain':x[0], 'wildtype': x[1], 'position': x[2], 'mutation': x[3] })

        # Split elements in the mutations column into seperate columns
        df_mut = df['Mutations'].apply(lambda m: split_mut(m.split(' ')))
        df = pd.concat([df, df_mut], axis=1)
        return df

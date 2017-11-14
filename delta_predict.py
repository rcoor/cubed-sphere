import re

import Bio
import Bio.PDB
import os
from batch_factory.deepfold_batch_factory import BatchFactory


def get_protein(input_dir, pdb_id, max_batch_size=25):

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

#chain_ids = batch['chain_ids']

''' # Extract index of first residue from PDB - and attempt to use this as offset into model
pdb_parser = Bio.PDB.PDBParser()
structure = pdb_parser.get_structure(pdb_id, os.path.join(pdb_dir, pdb_id+".pdb")) '''


# get_protein("./atomistic_features_cubed_sphere/", "1CML")

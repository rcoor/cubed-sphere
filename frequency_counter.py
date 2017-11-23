from Bio.PDB import Polypeptide
from Bio.PDB import PDBList
from Bio.PDB.Polypeptide import PPBuilder
import Bio
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle



class ResidueCounter():
    def __init__(self, pdb_dir, pdb_list, save_dir, save_name):
        self.freq = {}
        self.aa_counts = {}
        self.pdb_list = pdb_list
        self.pdb_dir = pdb_dir
        self.save_dir = save_dir
        self.save_name = save_name

    def count_to_frequency(self, d: dict):
        total = 0
        for key in d.keys():
            total += d[key]
        for key in d.keys():
            d[key] = np.divide(float(d[key]), float(total))

        return d

    def create_count_dict(self):
        # Create dictionary of counts
        aa_list = [Polypeptide.index_to_one(aa) for aa in range(20)]
        return dict([aa, 0] for aa in aa_list)

    def count_residues(self, pdb_id):
        cif_path = os.path.join(self.pdb_dir, pdb_id.lower() + ".cif")

        if not os.path.exists(cif_path):
                pdbl = PDBList()
                pdbl.retrieve_pdb_file(pdb_id, pdir="./data/PDB/")

        mmcif_parser = Bio.PDB.MMCIFParser()
        structure = mmcif_parser.get_structure(pdb_id, cif_path)

        # Create dictionary of counts
        aa_counts = self.create_count_dict()

        ppb=PPBuilder()
        for pp in ppb.build_peptides(structure):
            sequence = pp.get_sequence()
            for aa in sequence:
                if aa in aa_counts:
                    aa_counts[aa] += 1
                else:
                    continue
        self.aa_counts = aa_counts
        return aa_counts

    def join_counts(self, d1:dict, d2:dict):
        for key in d1.keys():
            d2[key] += d1[key]
        return d2

    def get_dataset_residue_count(self):
        joined_aa_counts = self.create_count_dict()
        for pdb_id in self.pdb_list:
            aa_counts = self.count_residues(pdb_id)
            joined_aa_counts = self.join_counts(aa_counts, joined_aa_counts)

        self.aa_counts = joined_aa_counts.copy()
        self.freq = self.count_to_frequency(joined_aa_counts)

    def save_frequency_to_pickle(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        freq_count_dict = {"counts": self.aa_counts, "frequencies": self.freq}
        pickle.dump(freq_count_dict, open( os.path.join(self.save_dir, self.save_name+"_freq.p" ), "wb"))



flags = tf.app.flags
flags.DEFINE_string("pdb_dir", "/Users/thorn/Documents/projects/cubed-sphere/data/PDB/", "Path to PDB files")
flags.DEFINE_string("save_dir", "/Users/thorn/Documents/projects/cubed-sphere/data/frequencies/", "Path to save directory")
flags.DEFINE_string("data_set", "/Users/thorn/Documents/projects/cubed-sphere/data/ddgs/kellogg.csv", "Input ddg datafile")

FLAGS = flags.FLAGS

def main():
    df = pd.read_csv(FLAGS.data_set, skiprows=21)
    rc = ResidueCounter(
        pdb_dir=FLAGS.pdb_dir,
        pdb_list= df['PDBFileID'].unique(),
        save_dir=FLAGS.save_dir,
        save_name = os.path.basename(FLAGS.data_set).split('.')[0]
        )
    rc.get_dataset_residue_count()
    print(rc.freq['A'])
    rc.save_frequency_to_pickle()

    print(rc.aa_counts)

if __name__ == '__main__':
    main()

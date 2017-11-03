from Bio.PDB import PDBParser, MMCIFParser, Polypeptide

from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB import PDBList

import numpy as np
pdbl = PDBList()
pdbl.retrieve_pdb_file("1A6M",pdir="PDB/")
structure = MMCIFParser().get_structure("1A6M", "PDB/1a6m.cif")

ppb=PPBuilder()
for pp in ppb.build_peptides(structure):
    seq = pp.get_sequence()


pp_index = [Polypeptide.one_to_index(AA) for AA in seq]


logits = np.load("protein_logits.npz")

new_list = []
for i in logits:
    print(logits['arr_0'])
    for k in logits['arr_0']:
        AA = list(k[0][0])
        AA.pop()
        new_list.append(AA)


protein_probs = []
for i, pp_id in enumerate(pp_index):
    print(i)
    protein_probs.append(new_list[i][pp_id])

print(sum(protein_probs))

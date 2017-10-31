import numpy as np
import pandas as pd
import tensorflow as tf
from Bio.PDB import Polypeptide
from batch_factory.deepfold_batch_factory import ProteinData
file_path = './atomistic_features_cubed_sphere/1A0E_residue_features.npz'

proteinData = ProteinData(file_path)

with np.load(file_path) as data:
    for key in data.keys():
        #aa = data['aa_one_hot']
        #print(len(aa[0]))
        #print(data[key])
        print(key)
    print(data['indices'])
    print(data['selector'])



PP = [Polypeptide.index_to_three(x) for x in range(20)]
i = [i for i in range(20)]
print(i)
print(PP)

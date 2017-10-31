import numpy as np

file_path = './atomistic_features_cubed_sphere/1A0E_protein_features.npz'

with np.load(file_path) as data:
    for key in data.keys():
        print(key)
        print(data['masses'])
        #print(data[key])
'''
    for aa in data['aa_one_hot']:
        print(aa)
    print(len(data['aa_one_hot']))

 '''

def aa_markov_chain_prob:
    pass

def aa_prob:
    pass

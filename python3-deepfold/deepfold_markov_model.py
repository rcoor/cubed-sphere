import glob
import numpy as np
import numpy.linalg
import os
from deepfold_model import BatchFactory, get_batch
import Bio

class MarkovModel:

    def __init__(self):

        self.transition_matrix = np.zeros([21,21])
        self.frequencies = np.zeros(21)

    def train(self,
              train_batch_factory):

        batch, batch_sizes = batch_factory.next(train_batch_factory.data_size(),
                                                subbatch_max_size=train_batch_factory.data_size(),
                                                enforce_protein_boundaries=True)
        batch = list(batch.values())[0]
        for index, length in zip(np.cumsum(batch_sizes)-batch_sizes, batch_sizes):

            protein_batch = get_batch(index, index+length,
                                      batch)

            protein_batch = protein_batch[0]
            indices = np.argmax(protein_batch, axis=1)

            for pair in zip(indices[:-1], indices[1:]):
                self.transition_matrix[pair] += 1

            for index in indices:
                self.frequencies[index] += 1

        np.set_printoptions(threshold=np.nan)        
        self.transition_matrix /= np.sum(self.transition_matrix, axis=1)[:,np.newaxis]
        self.frequencies /= np.sum(self.frequencies)

    def marginal_prob_at_index(self, sequence, index):
        if index==0 or index == len(sequence)-1:
            # return self.get_stationary_distribution()
            return self.get_frequencies()
        aa_before = Bio.PDB.Polypeptide.one_to_index(sequence[index-1])
        aa_after = Bio.PDB.Polypeptide.one_to_index(sequence[index+1])
        prob = np.zeros([20])
        for i in range(20):
            prob[i] = self.transition_matrix[aa_before, i]*self.transition_matrix[i, aa_after]
        prob /= np.sum(prob)
        return prob

    def get_stationary_distribution(self):
        print(self.transition_matrix)
        eigen_values, eigen_vectors = np.linalg.eig(self.transition_matrix.T)
        # print "???", max(eigen_values)
        # print "!!!", eigen_values
        assert(np.argmax(eigen_values) == 0)
        return np.real_if_close(eigen_vectors[:,0])

    def get_frequencies(self):
        return self.frequencies
    
    def save(self, model_filename):
        np.savez_compressed(model_filename,
                            transition_matrix=self.transition_matrix,
                            frequencies=self.frequencies)

    def restore(self, model_filename):
        loader = np.load(model_filename+".npz")
        self.transition_matrix = loader["transition_matrix"]
        self.frequencies = loader["frequencies"]

        
if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--input-dir", dest="input_dir",
                        help="Location of input files containing high-res features")
    parser.add_argument("--test-set-fraction", dest="test_set_fraction",
                        help="Fraction of data set aside for testing", type=float, default=0.25)
    parser.add_argument("--validation-set-size", dest="validation_set_size",
                        help="Size of validation set (taken out of training set)", type=int, default=10)

    options = parser.parse_args()

    protein_feature_filenames = sorted(glob.glob(os.path.join(options.input_dir, "*protein_features.npz")))    

    validation_end = test_start = int(len(protein_feature_filenames)*(1.-options.test_set_fraction))
    train_end = validation_start = int(validation_end-options.validation_set_size)    
        
    batch_factory = BatchFactory()
    batch_factory.add_data_set("data",
                               protein_feature_filenames[:train_end],
                               key_filter=["aa_one_hot"])

    print(train_end)
    # print protein_feature_filenames
    print(batch_factory.data_size())
    
    model = MarkovModel()
    model.train(batch_factory)

    print("1:", model.transition_matrix)

    print(model.get_frequencies())

    model.save("markov_model")

    # new_model = MarkovModel()
    # new_model.restore("markov_model")

    # print "2: ", new_model.transition_matrix

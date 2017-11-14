import os
import sys
import glob
import numpy as np
import random
import tensorflow as tf
import tf_pad_wrap

import deepfold_grid as grid

from tensorflow.contrib import opt
from tensorflow.contrib.opt.python.training.external_optimizer import _get_shape_tuple
from functools import reduce

class ScipyOptimizerInterfaceAccumulatedGradients(opt.ScipyOptimizerInterface):
    def __init__(self, loss, var_list=None, equalities=None, inequalities=None,
                 batch_size=25,
                 **optimizer_kwargs):
        opt.ScipyOptimizerInterface.__init__(self, loss=loss, var_list=var_list, equalities=equalities, inequalities=inequalities, **optimizer_kwargs)
        self.batch_size = batch_size

    def _make_eval_func(self, tensors, session, feed_dict, fetches,
                      callback=None):
        """Construct a function that evaluates a `Tensor` or list of `Tensor`s."""
        if not isinstance(tensors, list):
            tensors = [tensors]
        num_tensors = len(tensors)

        def eval_func(x):
            """Function to evaluate a `Tensor`."""

            gradient_batch_sizes = feed_dict["gradient_batch_sizes"]

            n_batches = 0
            accumulated_augmented_fetch_vals = None
            for index, length in zip(np.cumsum(gradient_batch_sizes)-gradient_batch_sizes, gradient_batch_sizes):

                batch_feed_dict = feed_dict.copy()
                
                for key in list(batch_feed_dict.keys()):
                    # Remove non-tf keys (i.e. gradient_batch_sizes)
                    if isinstance(key, str):
                        del batch_feed_dict[key]
                    elif not np.isscalar(batch_feed_dict[key]) and batch_feed_dict[key] is not None:
                        batch_feed_dict[key] = batch_feed_dict[key][index:index+length] 
                
                augmented_feed_dict = {
                    var: x[packing_slice].reshape(_get_shape_tuple(var))
                    for var, packing_slice in zip(self._vars, self._packing_slices)
                }
                augmented_feed_dict.update(batch_feed_dict)
                augmented_fetches = tensors + fetches

                augmented_fetch_vals = session.run(
                    augmented_fetches, feed_dict=augmented_feed_dict)

                if accumulated_augmented_fetch_vals is None:
                    accumulated_augmented_fetch_vals = augmented_fetch_vals
                else:
                    for i, fetch_val in enumerate(accumulated_augmented_fetch_vals):
                        accumulated_augmented_fetch_vals[i] += augmented_fetch_vals[i]

                n_batches += 1
                        
            for i, fetch_val in enumerate(accumulated_augmented_fetch_vals):
                accumulated_augmented_fetch_vals[i] /= n_batches
                    
            if callable(callback):
                callback(*accumulated_augmented_fetch_vals)

            return accumulated_augmented_fetch_vals[:num_tensors]
                        
            # total_size = feed_dict.values()[0].shape[0]
            # n_batches = total_size//self.batch_size

            # accumulated_augmented_fetch_vals = None
            # for i in range(n_batches):
            #     start_index = i*self.batch_size
            #     end_index = start_index+self.batch_size

            #     batch_feed_dict = feed_dict.copy()
            #     for key in batch_feed_dict:
            #         if not np.isscalar(batch_feed_dict[key]) and batch_feed_dict[key] is not None:
            #             batch_feed_dict[key] = batch_feed_dict[key][start_index:end_index] 

            #     augmented_feed_dict = {
            #         var: x[packing_slice].reshape(_get_shape_tuple(var))
            #         for var, packing_slice in zip(self._vars, self._packing_slices)
            #     }
            #     augmented_feed_dict.update(batch_feed_dict)
            #     augmented_fetches = tensors + fetches

            #     augmented_fetch_vals = session.run(
            #         augmented_fetches, feed_dict=augmented_feed_dict)

            #     if accumulated_augmented_fetch_vals is None:
            #         accumulated_augmented_fetch_vals = augmented_fetch_vals
            #     else:
            #         for i, fetch_val in enumerate(accumulated_augmented_fetch_vals):
            #             accumulated_augmented_fetch_vals[i] += augmented_fetch_vals[i]

            # for i, fetch_val in enumerate(accumulated_augmented_fetch_vals):
            #     accumulated_augmented_fetch_vals[i] /= n_batches
                    
            # if callable(callback):
            #     callback(*accumulated_augmented_fetch_vals)

            # return accumulated_augmented_fetch_vals[:num_tensors]

        return eval_func        

def get_batch(start_index, end_index, *values):
    values_batch = []
    for value in values:
        if value is None:
            values_batch.append(None)
        else:
            values_batch.append(value[start_index:end_index])
    return values_batch

class ProteinData:
    '''Training data for a single protein, summarizing info at protein and residue level'''

    def __init__(self, protein_feature_filename, key_filter=[]):

        self.features = {}
        protein_loader = np.load(protein_feature_filename)
        self.seq_length = 0
        selected_feature_keys = []
        self.dimensions = 0
        for key in list(protein_loader.keys()):
            value = protein_loader[key]
            if len(value.shape) == 0:
                value = np.asscalar(value)
            else:
                if key in key_filter:
                    self.seq_length = value.shape[0]
                    selected_feature_keys.append(key)
            self.features[key] = value

        if len(selected_feature_keys) > 0:
            self.selected_features = self.features[selected_feature_keys[0]]
        for i in range(1, len(selected_feature_keys)):
            self.selected_features = np.vstack([self.selected_features,
                                                self.features[selected_feature_keys[i]]])
        # self.features_per_index = []
        # for i in range(self.seq_length):
        #     self.features_per_index.append([])
        #     for key in seq_features:
        #         self.features_per_index[-1].append(self.features[key][i])
            
        # self.protein_feature_filename = protein_feature_filename

    def initialize_residue_features(self, size):
        # self.features_per_index = []
        # for i in range(self.seq_length):
        #     self.features_per_index.append([])
        #     for key in self.seq_features:
        #         self.features_per_index[-1].append(self.features[key][i])
        return np.zeros([size]+list(self.selected_features.shape[1:]))
        # return [None for i in range(size)]
        
    def __len__(self):
        return self.seq_length
        
    def fetch_residue_features(self):
        pass

    def forget_residue_features(self):
        pass

    def get_residue_features(self, residue_index):
        return self.selected_features[residue_index]
        # r_value = []
        # for key in self.seq_features:
        #     r_value.append(self.features[key][residue_index])
        # return np.array(r_value)
        # return self.features_per_index[residue_index]
        
    
class ProteinGridData(ProteinData):
    '''Training data for a single protein, specialized for data in a 3D grid layout'''
    
    def __init__(self, protein_feature_filename, grid_feature_filename, exclude_at_center=[]):
        ProteinData.__init__(self, protein_feature_filename)
        # self.features = {}
        # protein_loader = np.load(protein_feature_filename)
        # for key in protein_loader.keys():
        #     value = protein_loader[key]
        #     if len(value.shape) == 0:
        #         value = np.asscalar(value)
        #     self.features[key] = value 
        # self.n_residues = int(protein_loader["n_residues"])
        # self.max_radius = int(protein_loader["max_radius"])
        # self.n_features = int(protein_loader["n_features"])
        # self.bins_per_angstrom = int(protein_loader["bins_per_angstrom"])
        # self.masses_full = protein_loader["masses"]
        # self.charges_full = protein_loader["charges"]
        # self.ss = protein_loader["ss"]
        self.grid_feature_filename = grid_feature_filename

        self.grid_shape = grid.get_grid_shape(max_radius=self.features["max_radius"],
                                              n_features=self.features["n_features"],
                                              bins_per_angstrom=self.features["bins_per_angstrom"])
        self.selector_array = None
        self.indices_array = None
        self.exclude_at_center = exclude_at_center

    def initialize_residue_features(self, size):
        return np.zeros((size, self.grid_shape[0], self.grid_shape[1], self.grid_shape[2], self.grid_shape[3]))
        
    def fetch_residue_features(self):
        '''Read in residue information. This takes up quite some space, and is therefore
           not done by default during construction'''
        if self.selector_array is None:
            residue_loader = np.load(self.grid_feature_filename)
            self.selector_array = residue_loader["selector"]
            self.indices_array = residue_loader["indices"]

    def forget_residue_features(self):
        '''Forget about residue information to free up space in memort.'''
        del self.selector_array
        del self.indices_array
        self.selector_array = None
        self.indices_array = None
        
    def get_residue_features(self, residue_index):
        '''Construct grid matrix from residue features'''

        # If selector_array has not been set, we fetch it here, but remember
        # to remove it again at the end of the function.
        fetch_temporarily = False
        if self.selector_array is None:
            self.fetch_grid_features()
            fetch_temporarily = True

        # Extract information for the current residue index
        selector = self.selector_array[residue_index][self.selector_array[residue_index]>=0]

        # Extract data on which grid indices to set
        indices = self.indices_array[residue_index][self.indices_array[residue_index][:,0]>=0]

        # Create grid
        grid_matrix = grid.create_grid(max_radius=self.features["max_radius"],
                                       n_features=self.features["n_features"],
                                       bins_per_angstrom=self.features["bins_per_angstrom"])

        start_index = 0
        for feature_name in self.features["residue_features"]:

            if feature_name == "residue_index":
                full_feature = np.arange(self.features['n_residues']).reshape((self.features['n_residues'],1)) - residue_index
            else :
                full_feature = self.features[feature_name]
            feature = full_feature[selector]
            # print feature
            
            end_index = start_index + feature.shape[1]

            # print feature_name, start_index, end_index

            # Get default values if they are available
            if (feature_name+"_default") in self.features:
                # print feature_name + "_default", getattr(self, feature_name + "_default")
                grid_matrix[:, :, :, start_index:end_index] = self.features[feature_name + "_default"]
            
            grid_matrix[indices[:,0], indices[:,1], indices[:,2], start_index:end_index] = feature

            start_index += feature.shape[1]

        # # Limit data to indices specified by selector
        # masses = self.masses_full[selector]
        # charges = self.charges_full[selector]
        
        # # Populate grid
        # grid_matrix[indices[:,0], indices[:,1], indices[:,2], 0] = masses[:,0]
        # grid_matrix[indices[:,0], indices[:,1], indices[:,2], 1] = charges[:,0]

        if len(self.exclude_at_center) > 0:
            start_index = 0
            for feature_name in self.features["residue_features"]:
                if feature_name == "residue_index":
                    full_feature = np.arange(self.features['n_residues']).reshape((self.features['n_residues'],1)) - residue_index
                else :
                    full_feature = self.features[feature_name]
                feature = full_feature[selector]
                end_index = start_index + feature.shape[1]
                if feature_name in self.exclude_at_center and (feature_name+"_default") in self.features:
                    grid_matrix[0, :, :, start_index:end_index] = self.features[feature_name + "_default"]
                start_index += feature.shape[1]

        # Experiment: Set origin values in all theta,phi bins
        if (indices[:,0]==0).any():
            assert(np.count_nonzero(indices[:,0]==0) == 1)
            index = np.asscalar(np.where(indices[:,0]==0)[0])
            grid_matrix[0, :, :, :] = grid_matrix[0, indices[index,1], indices[index,2], :]
        # index_theta = indices[index,1]
        # index_phi = indices[index,2]
        # # print index, grid_matrix[0, :, :, :].shape, grid_matrix[0, index_theta, index_phi, :].shape
        # grid_matrix[0, :, :, :] = grid_matrix[0, index_theta, index_phi, :]
        
        # Forget residue features again, if necessary
        if fetch_temporarily:
            self.forget_grid_features()
        
        return grid_matrix
    
        
class BatchFactory:
    '''Create batches for training'''
    
    def __init__(self):

        # pdb_id -> ProteinData map 
        self.features = {}

        # List of (pdb_ids, res_index) pairs 
        self.features_expanded = []

        # Current index into features_expanded list
        self.feature_index = 0


    def add_data_set(self, key, protein_feature_filenames, grid_feature_filenames=None, key_filter=[], exclude_at_center=[]):
        
        if grid_feature_filenames is None:
            grid_feature_filenames = [None]*len(protein_feature_filenames)
        
        for protein_feature_filename, grid_feature_filename in zip(sorted(protein_feature_filenames),
                                                                   sorted(grid_feature_filenames)):

            pdb_id = os.path.basename(protein_feature_filename)[0:5]

            if grid_feature_filename is not None:
                # Test that protein and residue data files have the same pdb_id prefix
                assert(pdb_id == os.path.basename(grid_feature_filename)[0:5])

            # Create feature data
            if grid_feature_filename is not None:
                protein_data = ProteinGridData(protein_feature_filename, grid_feature_filename, exclude_at_center)
            else:
                protein_data = ProteinData(protein_feature_filename, key_filter)
            if pdb_id not in self.features:
                self.features[pdb_id] = {}
            self.features[pdb_id][key] = protein_data

            # Randomize order of proteins
            self.shuffle_features()

    def data_size(self):
        return len(self.features_expanded)
            
    def shuffle_features(self):
        '''Randomize order of pdb_ids'''

        # Randomize order
        feature_pdb_ids = list(self.features.keys())
        random.shuffle(feature_pdb_ids)

        # Repopulate self.features_expanded
        self.features_expanded = []
        for pdb_id in feature_pdb_ids:
            n_residues = len(list(self.features[pdb_id].values())[0])
            self.features_expanded += list(zip([pdb_id]*n_residues, list(range(n_residues))))

        # Reset index counter
        self.feature_index = 0

        
    def next(self, max_size=10, enforce_protein_boundaries=True, subbatch_max_size=None, increment_counter=True):
        '''Create next batch
        '''

        # index_max = self.feature_index
        # if count_in_proteins:
        #     pdb_indices = [index_max]
        #     while len(pdb_indices) <= size:
        #         index = index_max % len(self.features_expanded)

        #         if index == 0:
        #             pdb_indices[-1] = (pdb_indices[-1], None)
        #             pdb_indices.append(0)
        #         elif self.features_expanded[index][0] != self.features_expanded[pdb_indices[-1]][0]:
        #             pdb_indices[-1] = (pdb_indices[-1], index)
        #             pdb_indices.append(index)
        #         index_max += 1
        #     print pdb_indices
        #     print [self.features_expanded[index][0] for index,_ in pdb_indices]
        
        # print "next: ", max_size
        # print "pdb_ids: ", [self.features_expanded[self.feature_index+index][0] for index in xrange(max_size)]
        # print "indices: ", sorted(np.unique([self.features_expanded[self.feature_index+index][0] for index in xrange(max_size)], return_index=True)[1])

        subbatch_sizes = None
        size = max_size
        if subbatch_max_size is not None:
            subbatch_sizes = []
            if enforce_protein_boundaries:
                pdb_ids = []
                for i in range(max_size):
                    index = (self.feature_index+i) % len(self.features_expanded)
                    pdb_ids.append(self.features_expanded[index][0])
                indices = sorted(np.unique(pdb_ids, return_index=True)[1])
                    
                # if the pdb_index changes right at the boundary, or if the entry spans the entire range, we can include the last entry
                last_entry_index = (self.feature_index+max_size-1) % len(self.features_expanded)
                last_entry_index_next = (self.feature_index+max_size) % len(self.features_expanded)
                if (len(indices) == 1 or
                    self.features_expanded[last_entry_index][0] !=
                    self.features_expanded[last_entry_index_next][0]):
                    # include last entry
                    indices.append(max_size)
                size = indices[-1]
                for i in range(len(indices)-1):
                    index = indices[i]+self.feature_index
                    length = indices[i+1]-indices[i]
                    n_subbatches = (length//subbatch_max_size)
                    if length%subbatch_max_size > 0:
                        n_subbatches += 1
                    subbatch_size_array = np.array([length/n_subbatches]*n_subbatches)
                    remainder = length%n_subbatches
                    # subbatch_size_array[np.random.choice(len(subbatch_size_array), remainder, replace=False)] += 1
                    subbatch_size_array[np.arange(remainder)] += 1
                    subbatch_sizes += list(subbatch_size_array)
                    # print index, self.features_expanded[index][0], length, n_subbatches, subbatch_size_array, sum(subbatch_size_array)
                subbatch_sizes = np.array(subbatch_sizes)
            else:
                n_subbatches = max_size//subbatch_max_size
                if max_size%subbatch_max_size > 0:
                    n_subbatches += 1
                subbatch_size_array = np.array([max_size/n_subbatches]*n_subbatches)
                remainder = max_size%n_subbatches
                # subbatch_size_array[np.random.choice(len(subbatch_size_array), remainder, replace=False)] += 1
                subbatch_size_array[np.arange(remainder)] += 1
                subbatch_sizes = subbatch_size_array

            assert(np.sum(subbatch_sizes) == size)

        residue_features = None
        pdb_ids = []
        for i in range(size):
            
            # Extract ProteinData object
            index = (self.feature_index+i) % len(self.features_expanded)
            pdb_id, residue_index = self.features_expanded[index]

            # Create container if necessary
            # This is done here because the shape is not known until we see the first instance
            if residue_features is None:
                residue_features = {}
                for key in self.features[pdb_id]:
                    residue_features[key] = self.features[pdb_id][key].initialize_residue_features(size)
                    # print dir(self.features[pdb_id][key]), type(self.features[pdb_id][key])
                    # grid_shape = self.features[pdb_id][key].grid_shape
                    # grid_features[key] = np.zeros((size, grid_shape[0], grid_shape[1], grid_shape[2], grid_shape[3]))
                # print self.features[pdb_id].keys()
                # grid_shape = self.features[pdb_id]["grid_shape"]
                # grid_features = np.zeros((size, grid_shape[0], grid_shape[1], grid_shape[2], grid_shape[3]))
                # labels = np.zeros(size, dtype=np.int8)
                
            for key in self.features[pdb_id]:
                
                # Pre-fetch residue features
                self.features[pdb_id][key].fetch_residue_features()

                # Get residue features
                residue_features[key][i] = self.features[pdb_id][key].get_residue_features(residue_index)

            # Keep track of which pdb_ids we have prefetched residue features for
            pdb_ids.append(pdb_id)


        # For all pre-fetched pdb_ids, make ProteinData object forget its residue data
        # (too large to keep it all in memory)
        for pdb_id in pdb_ids:
            for key in self.features[pdb_id]:
                self.features[pdb_id][key].forget_residue_features()

        # Increment counter
        if increment_counter:
            self.feature_index += size

        # If counter passes total number of features, reshuffle and reset
        if self.feature_index >= len(self.features_expanded):
            self.shuffle_features()

        # Make sure that feature_index is correctly set to zero if running on the complete se.
        if max_size == len(self.features_expanded):
            assert(self.feature_index == 0)
            
        return residue_features, subbatch_sizes

    
class Model:
    '''Model definition'''

    def __init__(self,
                 r_size_high_res, theta_size_high_res, phi_size_high_res, channels_high_res,
                 r_size_low_res, theta_size_low_res, phi_size_low_res, channels_low_res,
                 output_size,
                 max_batch_size=1000, max_gradient_batch_size=25,
                 include_high_res_model=True,
                 include_low_res_model=False,
                 optimize_using_lbfgs=False,
                 lbfgs_max_iterations=10,
                 model_checkpoint_path="models"):

        self.output_size = output_size
        self.include_low_res_model = include_low_res_model
        self.include_high_res_model = include_high_res_model
        self.optimize_using_lbfgs = optimize_using_lbfgs
        self.max_batch_size = max_batch_size
        self.max_gradient_batch_size = max_gradient_batch_size
        self.model_checkpoint_path = model_checkpoint_path
        
        self.x_high_res = tf.placeholder(tf.float32)
        if include_high_res_model:
            self.x_high_res = tf.placeholder(tf.float32, [None, r_size_high_res, theta_size_high_res, phi_size_high_res, channels_high_res])
            print("input (high_res): ", self.x_high_res.shape, file=sys.stderr)

        self.x_low_res = tf.placeholder(tf.float32)
        if include_low_res_model:
            self.x_low_res = tf.placeholder(tf.float32, [None, r_size_low_res, theta_size_low_res, phi_size_low_res, channels_low_res])
            print("input (low_res): ", self.x_low_res.shape, file=sys.stderr)

        self.y = tf.placeholder(tf.float32, [None, output_size])

        self.dropout_keep_prob = tf.placeholder(tf.float32)
        
        self.layers = []
        self.layers_low_res = []

        ### LAYER 1 ###
        if include_high_res_model:
            self.layers.append({})
            self.layers[-1].update(self.create_conv3D_separate_r_layer(len(self.layers)-1,
                                                                       self.x_high_res,
                                                                       window_size_r=3,
                                                                       window_size_theta=5,
                                                                       window_size_phi=5,
                                                                       # channels_out=48,
                                                                       # channels_out=12,
                                                                       channels_out=8,
                                                                       stride_r=2,
                                                                       stride_theta=1,
                                                                       stride_phi=1))
            self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['conv'])
            self.layers[-1].update(self.create_maxpool_layer(len(self.layers)-1,
                                                             self.layers[-1]['activation'],
                                                             ksize=[1,1,3,3,1],
                                                             strides=[1,1,2,2,1]))
            print("layer %s (high res) - activation: %s" % (len(self.layers), self.layers[-1]['activation'].shape), file=sys.stderr)
            print("layer %s (high res) - pooling: %s" % (len(self.layers), self.layers[-1]['pool'].shape), file=sys.stderr)

        if include_low_res_model:
            self.layers_low_res.append({})
            self.layers_low_res[-1].update(self.create_conv3D_separate_r_layer(len(self.layers_low_res)-1,
                                                                       self.x_low_res,
                                                                       window_size_r=1,
                                                                       window_size_theta=3,
                                                                       window_size_phi=3,
                                                                       # channels_out=48,
                                                                       # channels_out=12,
                                                                       channels_out=43,
                                                                       stride_r=1,
                                                                       stride_theta=1,
                                                                       stride_phi=1))
            self.layers_low_res[-1]['activation'] = tf.nn.relu(self.layers_low_res[-1]['conv'])
            self.layers_low_res[-1].update(self.create_maxpool_layer(len(self.layers_low_res)-1,
                                                             self.layers_low_res[-1]['activation'],
                                                             ksize=[1,1,3,3,1],
                                                             strides=[1,1,1,1,1]))
            print("layer %s (low res) - activation: %s" % (len(self.layers_low_res), self.layers_low_res[-1]['activation'].shape), file=sys.stderr)
            print("layer %s (low res) - pooling: %s" % (len(self.layers_low_res), self.layers_low_res[-1]['pool'].shape), file=sys.stderr)

        if include_high_res_model:
            ### LAYER 2 ###
            self.layers.append({})
            self.layers[-1].update(self.create_conv3D_separate_r_layer(len(self.layers)-1,
                                                                       self.layers[-2]['pool'],
                                                                       window_size_r=1,
                                                                       window_size_theta=3,
                                                                       window_size_phi=3,
                                                                       # channels_out=128,
                                                                       # channels_out=64,
                                                                       # channels_out=48,
                                                                       channels_out=24,
                                                                       stride_r=1,
                                                                       stride_theta=1,
                                                                       stride_phi=1))
            self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['conv'])
            self.layers[-1].update(self.create_maxpool_layer(len(self.layers)-1,
                                                             self.layers[-1]['activation'],
                                                             ksize=[1,1,3,3,1],
                                                             strides=[1,2,2,2,1]))

            print("layer %s (high res) - activation: %s" % (len(self.layers), self.layers[-1]['activation'].shape), file=sys.stderr)
            print("layer %s (high res) - pooling: %s" % (len(self.layers), self.layers[-1]['pool'].shape), file=sys.stderr)

        if include_high_res_model and include_low_res_model:
            # merged_layer = tf.concat((self.layers[-1]['pool'], self.x_low_res), axis=-1)
            merged_layer = tf.concat((self.layers[-1]['pool'], self.layers_low_res[-1]['pool']), axis=-1)
        elif include_high_res_model:
            merged_layer = self.layers[-1]['pool']
        elif include_low_res_model:
            # merged_layer = self.x_low_res
            merged_layer = self.layers_low_res[-1]['pool']
            
        ### LAYER 3 ###
        self.layers.append({})
        self.layers[-1].update(self.create_conv3D_separate_r_layer(len(self.layers)-1,
                                                                   # self.layers[-2]['pool'],
                                                                   merged_layer,
                                                                   window_size_r=1,
                                                                   window_size_theta=3,
                                                                   window_size_phi=3,
                                                                   # window_size_theta=2,
                                                                   # window_size_phi=2,
                                                                   # channels_out=192,
                                                                   # channels_out=128,
                                                                   # channels_out=96,
                                                                   channels_out=64,
                                                                   # channels_out=48,
                                                                   # channels_out=48,
                                                                   # channels_out=24,
                                                                   stride_r=1,
                                                                   stride_theta=1,
                                                                   stride_phi=1))
        self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['conv'])
        self.layers[-1].update(self.create_maxpool_layer(len(self.layers)-1,
                                                         self.layers[-1]['activation'],
                                                         ksize=[1,3,3,3,1],
                                                         strides=[1,2,2,2,1]))

        print("layer %s (high res) - activation: %s" % (len(self.layers), self.layers[-1]['activation'].shape), file=sys.stderr)
        print("layer %s (high res) - pooling: %s" % (len(self.layers), self.layers[-1]['pool'].shape), file=sys.stderr)

            
        ### LAYER 4 ###
        self.layers.append({})
        self.layers[-1].update(self.create_conv3D_separate_r_layer(len(self.layers)-1,
                                                                   self.layers[-2]['pool'],
                                                                   window_size_r=1,
                                                                   window_size_theta=3,
                                                                   window_size_phi=3,
                                                                   # window_size_theta=2,
                                                                   # window_size_phi=2,
                                                                   # channels_out=128,
                                                                   # channels_out=64,
                                                                   # channels_out=48,
                                                                   channels_out=24,
                                                                   # channels_out=12,
                                                                   stride_r=1,
                                                                   stride_theta=1,
                                                                   stride_phi=1))
        self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['conv'])
        self.layers[-1].update(self.create_maxpool_layer(len(self.layers)-1,
                                                         self.layers[-1]['activation'],
                                                         ksize=[1,1,3,3,1],
                                                         strides=[1,1,2,2,1]))
        print("layer %s (high res) - activation: %s" % (len(self.layers), self.layers[-1]['activation'].shape), file=sys.stderr)
        print("layer %s (high res) - pooling: %s" % (len(self.layers), self.layers[-1]['pool'].shape), file=sys.stderr)


        ### LAYER 5 ###
        self.layers.append({})
        self.layers[-1].update(self.create_dense_layer(len(self.layers)-1,
                                                       self.layers[-2]['pool'],
                                                       output_size=-1))
        self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['dense'])
        # self.layers[-1].update(self.create_dropout_layer(len(self.layers)-1,
        #                                                  self.layers[-1]['activation'],
        #                                                  self.dropout_keep_prob))
        self.layers[-1]['dropout'] = tf.nn.dropout(self.layers[-1]['activation'], self.dropout_keep_prob)
        print("layer %s (high res) - W: %s" % (len(self.layers), self.layers[-1]['W'].get_shape()), file=sys.stderr)
        print("layer %s (high res) - activation: %s" % (len(self.layers), self.layers[-1]['activation'].shape), file=sys.stderr)

            
            
        # ### LAYER 6 ###
        # self.layers.append({})
        # self.layers[-1].update(self.create_dense_layer(len(self.layers)-1,
        #                                                self.layers[-2]['dropout'],
        #                                                output_size=-1))
        # self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['dense'])
        # # self.layers[-1].update(self.create_dropout_layer(len(self.layers)-1,
        # #                                                  self.layers[-1]['activation'],
        # #                                                  self.dropout_keep_prob))
        # self.layers[-1]['dropout'] = tf.nn.dropout(self.layers[-1]['activation'], self.dropout_keep_prob)
        # print "layer %s (high res) - W: %s" % (len(self.layers), self.layers[-1]['W'].get_shape())
        # print "layer %s (high res) - activation: %s" % (len(self.layers), self.layers[-1]['activation'].shape)


        ### LAYER 7 ###
        self.layers.append({})
        self.layers[-1].update(self.create_dense_layer(len(self.layers)-1,
                                                       self.layers[-2]['dropout'],
                                                       output_size=output_size))
        self.layers[-1]['activation'] = tf.nn.softmax(self.layers[-1]['dense'])
        print("layer %s (high res) - W: %s" % (len(self.layers), self.layers[-1]['W'].get_shape()), file=sys.stderr)
        print("layer %s (high res) - activation: %s" % (len(self.layers), self.layers[-1]['activation'].shape), file=sys.stderr)


        # if include_high_res_model and include_low_res_model:
        #     raise NotImplementedError   # TODO: merge models into the same output layers
        # elif include_high_res_model:
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.layers[-1]['dense'], labels=self.y))
        # elif include_low_res_model:
        #     self.loss = tf.reduce_mean(
        #         tf.nn.softmax_cross_entropy_with_logits(logits=self.layers_low_res[-1]['activation'], labels=self.y))
        # else:
        #     raise ValueError("Either high or low res model should be included")

        # self.train_step = tf.train.AdamOptimizer(0.001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss)
        # optimizer = tf.train.AdamOptimizer(0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
        if optimize_using_lbfgs:
            self.optimizer = ScipyOptimizerInterfaceAccumulatedGradients(self.loss,
                                                                         batch_size=self.max_gradient_batch_size,
		                                                         method='L-BFGS-B', options={'maxiter': options.lbfgs_max_iterations})
        else:
            optimizer = tf.train.AdamOptimizer(0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)


        # Variables for accumulating gradients
        trainable_vars = tf.trainable_variables()
        accumulated_gradients = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in trainable_vars]

        # Operation for resetting gradient variables after each batch
        self.zero_op = [var.assign(tf.zeros_like(var)) for var in accumulated_gradients]

        # Compute gradients
        gradients = tf.gradients(self.loss, trainable_vars)

        # Accumulation operator
        self.gradient_accumulation_op = [accumulated_gradients[i].assign_add(gradient) for i, gradient in enumerate(gradients)]

        # Normalize gradients
        self.gradient_normalization_constant = tf.placeholder(tf.float32, shape=(), name="gradient_normalization_contant")
        gradients_normalized = np.asarray(accumulated_gradients) / np.asarray(self.gradient_normalization_constant)

        if not optimize_using_lbfgs:
            self.train_step = optimizer.apply_gradients(list(zip(gradients_normalized, trainable_vars)))


        print("Number of parameters: ", sum(reduce(lambda x, y: x*y, v.get_shape().as_list()) for v in tf.trainable_variables()), file=sys.stderr)
        
        # config = tf.ConfigProto()
        # config.gpu_options.allocator_type = 'BFC'        
        # self.session = tf.InteractiveSession(config=config)
        self.session = tf.InteractiveSession()
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.saver = tf.train.Saver(max_to_keep=2)
        
        # print self.layers[-1]['conv'].shape
        # print self.layers[-1]['pool'].shape
        
        # self.layers.append(self.create_conv2D_layer(len(self.layers),
        #                                             self.x,
        #                                             window_size_theta=5,
        #                                             window_size_phi=5,
        #                                             channels_out=48,
        #                                             stride_theta=2,
        #                                             stride_phi=2))
        # self.layers.append(self.create_conv3D_layer(len(self.layers),
        #                                             self.x,
        #                                             window_size_r=20,
        #                                             window_size_theta=5,
        #                                             window_size_phi=5,
        #                                             channels_out=48,
        #                                             stride_r=2,
        #                                             stride_theta=2,
        #                                             stride_phi=2))

        
        # i=0
        
        # filter_shape = [window_size_r, window_size_theta, window_size_phi, channels, channels] 
        # self.Ws.append(tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W%d"%i))
        # self.convs.append(tf.nn.conv3d(
        #     self.x,
        #     self.Ws[-1],
        #     strides=[1,1,1,1,1],
        #     padding="SAME",
        #     name="conv%d"%i))
        
        # var = tf.placeholder(tf.float32, [channels, phi_size])
        # tf.matmul(x, 

    @staticmethod
    def create_dropout_layer(index,
                             input,
                             keep_prob):
        random_tensor = tf.placeholder(tf.float32, input.shape, name="dropout_random_tensor%d"%index)
        random_tensor.set_shape(input.get_shape())

        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = tf.floor(random_tensor + keep_prob)        
        ret = tf.div(input, keep_prob) * binary_tensor
        ret.set_shape(input.get_shape())
        return {'dropout_random_tensor': random_tensor, 'binary_tensor':binary_tensor, 'dropout': ret}

    @staticmethod
    def generate_dropout_feed_dict(layers, batch_size):
        dropout_feed_dict = {}
        for layer in layers:
            if 'dropout_random_tensor' in layer:
                random_tensor_placeholder = layer['dropout_random_tensor']
                dropout_feed_dict[random_tensor_placeholder] = np.random.uniform(size=[batch_size]+random_tensor_placeholder.shape[1:].as_list())
        return dropout_feed_dict
    
    @staticmethod
    def create_maxpool_layer(index,
                             input,
                             ksize,
                             strides):
        # Pad input with periodic image
        padded_input = tf_pad_wrap.tf_pad_wrap(input, [(0,0), (0,0), (ksize[2]/2, ksize[2]/2), (ksize[3]/2, ksize[3]/2), (0,0)])

        return {'pool': tf.nn.max_pool3d(padded_input,
                                         ksize=ksize,
                                         strides=strides,
                                         padding='VALID')}
        
    @staticmethod
    def create_dense_layer(index,
                           input,
                           output_size):
        reshaped_input = tf.reshape(input, [-1, np.prod(input.get_shape().as_list()[1:])])

        if output_size == -1:
            output_size = reshaped_input.get_shape().as_list()[1]

        W = tf.Variable(tf.truncated_normal([reshaped_input.get_shape().as_list()[1], output_size], stddev=0.1), name="W%d"%index)
        b = tf.Variable(tf.truncated_normal([output_size], stddev=0.1), name="b%d"%index)
        dense = tf.nn.bias_add(tf.matmul(reshaped_input, W), b)

        return {'W':W, 'b':b, 'dense':dense}
        
        
        
    @staticmethod
    def create_conv3D_layer(index,
                            input,
                            window_size_r,
                            window_size_theta,
                            window_size_phi,
                            channels_out,
                            stride_r=1,
                            stride_theta=1,
                            stride_phi=1):

        # Pad input with periodic image
        padded_input = tf_pad_wrap.tf_pad_wrap(input, [(0,0), (0,0), (window_size_theta/2, window_size_theta/2), (window_size_phi/2, window_size_phi/2), (0,0)])
                                   
        filter_shape = [window_size_r, window_size_theta, window_size_phi, padded_input.get_shape().as_list()[-1], channels_out] 
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W%d"%index)
        b = tf.Variable(tf.truncated_normal(filter_shape[-1:], stddev=0.1), name="b%d"%i)
        conv = tf.tf.nn.bias_add(
            tf.nn.conv3d(padded_input,
                         W,
                         strides=[1, stride_r, stride_theta, stride_phi, 1],
                         padding="VALID",
                         name="conv%d"%index),
            b)
        # print input.shape
        # print padded_input.shape
        # print conv.shape
        # print conv2.shape
        return {'W':W, 'b':b, 'conv':conv}

    # @staticmethod
    # def create_conv2D_layer(index,
    #                         input,
    #                         window_size_theta,
    #                         window_size_phi,
    #                         channels_in,
    #                         channels_out,
    #                         stride_theta=1,
    #                         stride_phi=1):

    #     # Create convolutions for each r value
    #     convs = []
    #     for i in range(input.shape[1]):

    #         input_fixed_r = input[:,i,:,:,:]
        
    #         # Pad input with periodic image
    #         padded_input = tf_pad_wrap.tf_pad_wrap(input_fixed_r, [(0,0), (window_size_theta/2, window_size_theta/2), (window_size_phi/2, window_size_phi/2), (0,0)])
                                   
    #         filter_shape = [window_size_theta, window_size_phi, channels_in, channels_out] 
    #         W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_%d_%d"%(i,index))
    #         convs.append(tf.expand_dims(tf.nn.conv2d(padded_input,
    #                                                  W,
    #                                                  strides=[1, stride_theta, stride_phi, 1],
    #                                                  padding="VALID",
    #                                                  name="conv_%d_%d"%(i,index)), axis=1))
    #     conv = tf.concat(convs, axis=1)
    #     return {'W':W, 'conv':conv}
    
    @staticmethod
    def create_conv3D_separate_r_layer(index,
                                       input,
                                       window_size_r,
                                       window_size_theta,
                                       window_size_phi,
                                       channels_out,
                                       stride_r=1,
                                       stride_theta=1,
                                       stride_phi=1):

        # Create convolutions for each r value
        convs = []
        for i in range(window_size_r/2, input.shape[1]-window_size_r/2, stride_r):

            input_fixed_r = input[:,i-window_size_r/2:i+window_size_r/2+1,:,:,:]
        
            # Pad input with periodic image
            padded_input = tf_pad_wrap.tf_pad_wrap(input_fixed_r, [(0,0), (0,0), (window_size_theta/2, window_size_theta/2), (window_size_phi/2, window_size_phi/2), (0,0)])

            filter_shape = [window_size_r, window_size_theta, window_size_phi, padded_input.get_shape().as_list()[-1], channels_out]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_%d_%d"%(i,index))
            b = tf.Variable(tf.truncated_normal(filter_shape[-1:], stddev=0.1), name="b%d"%i)
            convs.append(
                tf.nn.bias_add(
                    tf.nn.conv3d(padded_input,
                                 W,
                                 strides=[1, 1, stride_theta, stride_phi, 1],
                                 padding="VALID",
                                 name="conv_%d_%d"%(i,index)),
                    b))
        conv = tf.concat(convs, axis=1)
        return {'W':W, 'b':b, 'conv':conv}
    
        
    def train(self,
              train_batch_factory, 
              num_passes=100,
              validation_batch_factory=None,
              output_interval=None,
              dropout_keep_prob=0.5):

        print("dropout keep probability: ", dropout_keep_prob, file=sys.stderr)
        
        if output_interval is None:
            if self.optimize_using_lbfgs:
                output_interval = 1
            else:
                output_interval = 10

        total_data_size = batch_factory.data_size()

        iteration = 0
        for i in range(num_passes):
            data_size = 0
            while data_size < total_data_size:

                batch, gradient_batch_sizes = batch_factory.next(self.max_batch_size,
                                                                 subbatch_max_size=self.max_gradient_batch_size,
                                                                 enforce_protein_boundaries=True)
                grid_matrix = None
                if "high_res" in batch:
                    grid_matrix = batch["high_res"]

                low_res_grid_matrix = None
                if "low_res" in batch:
                    low_res_grid_matrix = batch["low_res"]

                labels = batch["model_output"]
                    
                if self.optimize_using_lbfgs:

                    def loss_callback(loss, _):
                        print("%s Loss: %s" % (iteration+1, loss), file=sys.stderr)

                    self.optimizer.minimize(self.session,# fetches=[self.loss], loss_callback=optimizer_callback,
                                            loss_callback=loss_callback,
                                            feed_dict=dict(list({self.x_high_res: grid_matrix,
                                                            self.x_low_res: low_res_grid_matrix,
                                                            self.y: labels,
                                                            "gradient_batch_sizes": gradient_batch_sizes,
                                                            self.dropout_keep_prob:dropout_keep_prob}.items())
                                                           + list(self.generate_dropout_feed_dict(self.layers, labels.shape[0]).items())))
                else:

                    # Accumulate gradients
                    accumulated_loss_value = 0
                    for index, length in zip(np.cumsum(gradient_batch_sizes)-gradient_batch_sizes, gradient_batch_sizes):

                        grid_matrix_batch, low_res_grid_matrix_batch, labels_batch = get_batch(index, index+length,
                                                                                               grid_matrix, low_res_grid_matrix, labels)

                        _, loss_value = self.session.run([self.gradient_accumulation_op, self.loss],
                                                         feed_dict=dict(list({self.x_high_res: grid_matrix_batch,
                                                                         self.x_low_res: low_res_grid_matrix_batch,
                                                                         self.y: labels_batch,
                                                                         self.dropout_keep_prob:dropout_keep_prob}.items())
                                                                        + list(self.generate_dropout_feed_dict(self.layers, length).items())))

                        accumulated_loss_value += loss_value

                    print("%s Loss: %s" % (iteration+1,  accumulated_loss_value/float(len(gradient_batch_sizes))), file=sys.stderr)

                    self.session.run([self.train_step],
                                     feed_dict={self.gradient_normalization_constant: float(len(gradient_batch_sizes))})

                    # Clear the accumulated gradients
                    self.session.run(self.zero_op)

                    
                if (iteration+1) % output_interval == 0:
                    Q_training_batch = self.Q_accuracy(batch, gradient_batch_sizes, raw=True)
                    print("%s Q%s score (training batch): %s" % (iteration+1, self.output_size, Q_training_batch), file=sys.stderr)

                    validation_batch, validation_gradient_batch_sizes = validation_batch_factory.next(validation_batch_factory.data_size(),
                                                                                                      subbatch_max_size=self.max_gradient_batch_size,
                                                                                                      enforce_protein_boundaries=True)
                    Q_validation = self.Q_accuracy(validation_batch, validation_gradient_batch_sizes)
                    print("%s Q%s score (validation set): %s" % (iteration+1, self.output_size, Q_validation), file=sys.stderr)

                    self.save(self.model_checkpoint_path, iteration)

                    
                iteration += 1    

                
    def save(self, checkpoint_path, step):

        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        self.saver.save(self.session, os.path.join(checkpoint_path, 'model.ckpt'), global_step=step)
        print("Model saved", file=sys.stderr)

    def restore(self, checkpoint_path):

        ckpt = tf.train.get_checkpoint_state(checkpoint_path)

        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.session, tf.train.latest_checkpoint(checkpoint_path))
        else:
            print("Could not load file", file=sys.stderr)    

    def infer(self, batch, gradient_batch_sizes):

        grid_matrix = None
        if "high_res" in batch:
            grid_matrix = batch["high_res"]

        low_res_grid_matrix = None
        if "low_res" in batch:
            low_res_grid_matrix = batch["low_res"]

        results = []
        for index, length in zip(np.cumsum(gradient_batch_sizes)-gradient_batch_sizes, gradient_batch_sizes):
            grid_matrix_batch, low_res_grid_matrix_batch = \
                 get_batch(index, index+length,
                           grid_matrix, low_res_grid_matrix)

            results += list(self.session.run(self.layers[-1]['activation'],
                                             feed_dict=dict(list({self.x_high_res: grid_matrix_batch,
                                                             self.x_low_res: low_res_grid_matrix_batch,
                                                             self.dropout_keep_prob:1.0}.items())
                                                            + list(self.generate_dropout_feed_dict(self.layers, length).items()))))
        return np.array(results)

    def Q_accuracy(self, batch, gradient_batch_sizes, raw=False):

        y = batch["model_output"]
        
        # predictions = self.infer(X_high_res, X_low_res)
        predictions = self.infer(batch, gradient_batch_sizes)
        np.set_printoptions(threshold=np.nan)
        # print predictions[:20]
        # print y[:20]
        predictions = tf.argmax(predictions, 1).eval()
        y_argmax = tf.argmax(y, 1).eval()
        identical = (predictions==y_argmax)
        return np.mean(identical)
    

if __name__ == '__main__':

    from argparse import ArgumentParser

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = ArgumentParser()
    parser.add_argument("--high-res-input-dir", dest="high_res_features_input_dir",
                        help="Location of input files containing high-res features")
    parser.add_argument("--low-res-input-dir", dest="low_res_features_input_dir",
                        help="Location of input files containing low-res features")
    parser.add_argument("--test-set-fraction", dest="test_set_fraction",
                        help="Fraction of data set aside for testing", type=float, default=0.25)
    parser.add_argument("--validation-set-size", dest="validation_set_size",
                        help="Size of validation set (taken out of training set)", type=int, default=10)
    parser.add_argument("--include-high-res-model", dest="include_high_res_model",
                        type=str2bool, nargs='?', const=True, default="True",
                        help="Whether to include atomic-resolution part of model")
    parser.add_argument("--include-low-res-model", dest="include_low_res_model",
                        type=str2bool, nargs='?', const=True, default="False",
                        help="Whether to include low resolution part of model")
    parser.add_argument("--optimize-using-lbfgs", dest="optimize_using_lbfgs",
                        help="Whether to use the LBFGS optimizer", type=bool, default=False)
    parser.add_argument("--lbfgs-max-iterations", type=int, default=10, 
                        help="Number of iterations in each BFGS step")
    parser.add_argument("--max-batch-size", dest="max_batch_size",
                        help="Maximum batch size used during training", type=int, default=1000)
    parser.add_argument("--max-gradient-batch-size", dest="max_gradient_batch_size",
                        help="Maximum batch size used for gradient calculation", type=int, default=25)
    parser.add_argument("--model-checkpoint-path", dest="model_checkpoint_path",
                        help="Where to dump/read model checkpoints", default="models")
    parser.add_argument("--read-from-checkpoint", action="store_true", dest="read_from_checkpoint",
                        help="Whether to read model from checkpoint")
    parser.add_argument("--mode", choices=['train', 'test'], dest="mode", default="train", 
                        help="Mode of operation: train or test")
    parser.add_argument("--model-output-type", choices=['aa', 'ss'], dest="model_output_type", default="ss", 
                        help="Whether the model should output secondary structure or amino acid labels")
    parser.add_argument("--dropout-keep-prob", type=float, default=0.5, 
                        help="Probability for leaving out node in dropout")

    options = parser.parse_args()
    
    high_res_protein_feature_filenames = sorted(glob.glob(os.path.join(options.high_res_features_input_dir, "*protein_features.npz")))
    high_res_grid_feature_filenames = sorted(glob.glob(os.path.join(options.high_res_features_input_dir, "*residue_features.npz")))

    low_res_protein_feature_filenames = sorted(glob.glob(os.path.join(options.low_res_features_input_dir, "*protein_features.npz")))
    low_res_grid_feature_filenames = sorted(glob.glob(os.path.join(options.low_res_features_input_dir, "*residue_features.npz")))

    # train_end = test_start = int(len(protein_feature_filenames)-10)
    validation_end = test_start = int(len(high_res_protein_feature_filenames)*(1.-options.test_set_fraction))
    train_end = validation_start = int(validation_end-options.validation_set_size)

    exclude_at_center = []
    if options.model_output_type == "aa":
        exclude_at_center = ["aa_one_hot_w_unobserved"]

    batch_factory = BatchFactory()
    batch_factory.add_data_set("high_res",
                               high_res_protein_feature_filenames[:train_end],
                               high_res_grid_feature_filenames[:train_end])
    batch_factory.add_data_set("low_res",
                               low_res_protein_feature_filenames[:train_end],
                               low_res_grid_feature_filenames[:train_end],
                               exclude_at_center = exclude_at_center)
    batch_factory.add_data_set("model_output",
                               low_res_protein_feature_filenames[:train_end],
                               key_filter=[options.model_output_type+"_one_hot"])
    
    validation_batch_factory = BatchFactory()
    validation_batch_factory.add_data_set("high_res",
                                          high_res_protein_feature_filenames[validation_start:validation_end],
                                          high_res_grid_feature_filenames[validation_start:validation_end])
    validation_batch_factory.add_data_set("low_res",
                                          low_res_protein_feature_filenames[validation_start:validation_end],
                                          low_res_grid_feature_filenames[validation_start:validation_end],
                                          exclude_at_center = exclude_at_center)
    validation_batch_factory.add_data_set("model_output",
                                          low_res_protein_feature_filenames[validation_start:validation_end],
                                          key_filter=[options.model_output_type+"_one_hot"])

    # high_res_training_batch_factory = BatchFactory(high_res_protein_feature_filenames[:train_end],
    #                                                high_res_grid_feature_filenames[:train_end])
    # high_res_validation_batch_factory = BatchFactory(high_res_protein_feature_filenames[validation_start:validation_end],
    #                                                  high_res_grid_feature_filenames[validation_start:validation_end])

    # low_res_training_batch_factory = BatchFactory(low_res_protein_feature_filenames[:train_end],
    #                                               low_res_grid_feature_filenames[:train_end])
    # low_res_validation_batch_factory = BatchFactory(low_res_protein_feature_filenames[validation_start:validation_end],
    #                                                 low_res_grid_feature_filenames[validation_start:validation_end])

    # model_output_batch_factory = BatchFactory(protein_feature_filenames=low_res_protein_feature_filenames[validation_start:validation_end],
    #                                           grid_feature_filenames=None)
    # for i in range(10):
    #     print i
    #     grid_matrix = batch_factory.next(1000)

    # # print np.array(sorted(indices, key=lambda x:x[0]))
    # print grid_matrix.shape
    # print np.vstack(grid_matrix.nonzero()).T
    high_res_grid_size = batch_factory.next(1, increment_counter=False)[0]["high_res"].shape
    low_res_grid_size  = batch_factory.next(1, increment_counter=False)[0]["low_res"].shape
    output_size        = batch_factory.next(1, increment_counter=False)[0]["model_output"].shape[1]

    model = Model(r_size_high_res         = high_res_grid_size[1],
                  theta_size_high_res     = high_res_grid_size[2],
                  phi_size_high_res       = high_res_grid_size[3],
                  channels_high_res       = high_res_grid_size[4],
                  r_size_low_res          = low_res_grid_size[1],
                  theta_size_low_res      = low_res_grid_size[2],
                  phi_size_low_res        = low_res_grid_size[3],
                  channels_low_res        = low_res_grid_size[4],
                  output_size             = output_size,
                  max_batch_size          = options.max_batch_size,
                  max_gradient_batch_size = options.max_gradient_batch_size,
                  include_high_res_model  = options.include_high_res_model,
                  include_low_res_model   = options.include_low_res_model,
                  optimize_using_lbfgs    = options.optimize_using_lbfgs,
                  lbfgs_max_iterations    = options.lbfgs_max_iterations,
                  model_checkpoint_path   = options.model_checkpoint_path)

    if options.read_from_checkpoint:
        model.restore(options.model_checkpoint_path)
    
    
    # model.train(high_res_training_batch_factory   = high_res_training_batch_factory,
    #             low_res_training_batch_factory    = low_res_training_batch_factory,
    #             high_res_validation_batch_factory = high_res_validation_batch_factory,
    #             low_res_validation_batch_factory  = low_res_validation_batch_factory,
    #             model_output_batch_factory        = model_output_batch_factory)
    if options.mode == 'train':
        model.train(train_batch_factory      = batch_factory,
                    validation_batch_factory = validation_batch_factory,
                    dropout_keep_prob        = options.dropout_keep_prob)

    elif options.mode == 'test':
        validation_batch, validation_gradient_batch_sizes = validation_batch_factory.next(validation_batch_factory.data_size(),
                                                                                          subbatch_max_size=options.max_gradient_batch_size,
                                                                                          enforce_protein_boundaries=True)
        model.infer(validation_batch, validation_gradient_batch_sizes)
        Q_validation = model.Q_accuracy(validation_batch, validation_gradient_batch_sizes)
        print("Q%s score (validation set): %s" % (output_size, Q_validation), file=sys.stderr)

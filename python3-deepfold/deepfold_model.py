import glob
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib import opt
from tensorflow.contrib.opt.python.training.external_optimizer import _get_shape_tuple

from Deepfold.Utils import tf_pad_wrap
from Deepfold.batch_factory import get_batch, BatchFactory
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

            # for key,val in feed_dict.items():
            #     if not np.isscalar(val) and not isinstance(key, str):
            #         print key.name, val.shape
            
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
                
                # for key,val in batch_feed_dict.items():
                #     if not np.isscalar(val):
                #         print "\t\t\t", key.name, val.shape
                
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
                accumulated_augmented_fetch_vals[i] /= len(gradient_batch_sizes)
                    
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
    
class Model:
    '''Model definition'''

    def __init__(self,
                 r_size_high_res, theta_size_high_res, phi_size_high_res, channels_high_res,
                 r_size_low_res, theta_size_low_res, phi_size_low_res, channels_low_res,
                 output_size,
                 max_batch_size=1000, max_gradient_batch_size=25,
                 include_high_res_model=True,
                 include_low_res_model=False,
                 train_high_res_model=True,
                 train_low_res_model=True,
                 accumulate_gradients=False,
                 optimize_using_lbfgs=False,
                 lbfgs_max_iterations=10,
                 model_checkpoint_path="models",
                 regularization_factor = 0.001,
                 learning_rate = 0.0001):

        self.output_size = output_size
        self.include_low_res_model = include_low_res_model
        self.include_high_res_model = include_high_res_model
        self.optimize_using_lbfgs = optimize_using_lbfgs
        self.max_batch_size = max_batch_size
        self.max_gradient_batch_size = max_gradient_batch_size
        self.model_checkpoint_path = model_checkpoint_path
        self.accumulate_gradients = accumulate_gradients
        
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

        ### LAYER 1 - high res ###
        if include_high_res_model:
            self.layers.append({})
            # self.layers[-1].update(self.create_conv3D_separate_r_layer(len(self.layers)-1,
            self.layers[-1].update(self.create_spherical_conv(len(self.layers)-1,
                                                              self.x_high_res,
                                                              window_size_r=3,
                                                              window_size_theta=5,
                                                              window_size_phi=5,
                                                              # channels_out=48,
                                                              # channels_out=12,
                                                              channels_out=8,
                                                              # channels_out=16,
                                                              stride_r=1,
                                                              stride_theta=2,
                                                              stride_phi=2,
                                                              name_suffix="_high_res"))
            self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['conv'])
            self.layers[-1].update(self.create_avgpool_layer(len(self.layers)-1,
                                                             self.layers[-1]['activation'],
                                                             ksize=[1,1,1,3,1],
                                                             strides=[1,1,1,2,1]))
            if not train_high_res_model:
                self.layers[-1]['pool'] = tf.stop_gradient(self.layers[-1]['pool'])
            print("layer %s (high res) - activation: %s" % (len(self.layers), self.layers[-1]['activation'].shape), file=sys.stderr)
            print("layer %s (high res) - pooling: %s" % (len(self.layers), self.layers[-1]['pool'].shape), file=sys.stderr)

        ### LAYER 1 - low res ###
        if include_low_res_model:
            self.layers_low_res.append({})
            # self.layers_low_res[-1].update(self.create_conv3D_separate_r_layer(len(self.layers_low_res)-1,
            self.layers_low_res[-1].update(self.create_spherical_conv(len(self.layers_low_res)-1,
                                                                      self.x_low_res,
                                                                      window_size_r=1,
                                                                      window_size_theta=3,
                                                                      window_size_phi=3,
                                                                      # channels_out=48,
                                                                      # channels_out=12,
                                                                      # channels_out=43,
                                                                      channels_out=22,
                                                                      stride_r=1,
                                                                      stride_theta=1,
                                                                      stride_phi=1,
                                                                      name_suffix="_low_res"))
            self.layers_low_res[-1]['activation'] = tf.nn.relu(self.layers_low_res[-1]['conv'])
            self.layers_low_res[-1].update(self.create_avgpool_layer(len(self.layers_low_res)-1,
                                                             self.layers_low_res[-1]['activation'],
                                                             ksize=[1,1,3,3,1],
                                                             strides=[1,2,2,1,1]))
            if not train_low_res_model:
                self.layers[-1]['pool'] = tf.stop_gradient(self.layers[-1]['pool'])
            print("layer %s (low res) - activation: %s" % (len(self.layers_low_res), self.layers_low_res[-1]['activation'].shape), file=sys.stderr)
            print("layer %s (low res) - pooling: %s" % (len(self.layers_low_res), self.layers_low_res[-1]['pool'].shape), file=sys.stderr)


        ### LAYER 2 - high res ###
        if include_high_res_model:
            self.layers.append({})
            # self.layers[-1].update(self.create_conv3D_separate_r_layer(len(self.layers)-1,
            self.layers[-1].update(self.create_spherical_conv(len(self.layers)-1,
                                                              self.layers[-2]['pool'],
                                                              window_size_r=3,
                                                              window_size_theta=3,
                                                              window_size_phi=3,
                                                              # channels_out=128,
                                                              # channels_out=64,
                                                              # channels_out=48,
                                                              # channels_out=24,
                                                              # channels_out=32,
                                                              channels_out=16,
                                                              stride_r=1,
                                                              stride_theta=1,
                                                              stride_phi=1,
                                                              name_suffix="_high_res"))
            self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['conv'])
            self.layers[-1].update(self.create_avgpool_layer(len(self.layers)-1,
                                                             self.layers[-1]['activation'],
                                                             ksize=[1,3,3,3,1],
                                                             strides=[1,2,2,2,1]))
            if not train_high_res_model:
                self.layers[-1]['pool'] = tf.stop_gradient(self.layers[-1]['pool'])

            print("layer %s (high res) - activation: %s" % (len(self.layers), self.layers[-1]['activation'].shape), file=sys.stderr)
            print("layer %s (high res) - pooling: %s" % (len(self.layers), self.layers[-1]['pool'].shape), file=sys.stderr)

        ### LAYER 2 - low res ###
        if include_low_res_model:
            self.layers_low_res.append({})
            # self.layers_low_res[-1].update(self.create_conv3D_separate_r_layer(len(self.layers_low_res)-1,
            self.layers_low_res[-1].update(self.create_spherical_conv(len(self.layers_low_res)-1,
                                                                      self.layers_low_res[-2]['pool'],
                                                                      window_size_r=1,
                                                                      window_size_theta=3,
                                                                      window_size_phi=3,
                                                                      # channels_out=48,
                                                                      # channels_out=12,
                                                                      channels_out=24,
                                                                      stride_r=1,
                                                                      stride_theta=1,
                                                                      stride_phi=1,
                                                                      name_suffix="_low_res"))
            self.layers_low_res[-1]['activation'] = tf.nn.relu(self.layers_low_res[-1]['conv'])
            self.layers_low_res[-1].update(self.create_avgpool_layer(len(self.layers_low_res)-1,
                                                             self.layers_low_res[-1]['activation'],
                                                             ksize=[1,1,3,3,1],
                                                             strides=[1,1,1,1,1]))
            if not train_low_res_model:
                self.layers[-1]['pool'] = tf.stop_gradient(self.layers[-1]['pool'])
            print("layer %s (low res) - activation: %s" % (len(self.layers_low_res), self.layers_low_res[-1]['activation'].shape), file=sys.stderr)
            print("layer %s (low res) - pooling: %s" % (len(self.layers_low_res), self.layers_low_res[-1]['pool'].shape), file=sys.stderr)

            
        ### LAYER 3 - high res ###
        if include_high_res_model:
            self.layers.append({})
            # self.layers[-1].update(self.create_conv3D_separate_r_layer(len(self.layers)-1,
            self.layers[-1].update(self.create_spherical_conv(len(self.layers)-1,
                                                              # self.layers[-2]['pool'],
                                                              self.layers[-2]['pool'],
                                                              # merged_layer,
                                                              window_size_r=3,
                                                              window_size_theta=3,
                                                              window_size_phi=3,
                                                              # window_size_theta=2,
                                                              # window_size_phi=2,
                                                              # channels_out=192,
                                                              # channels_out=128,
                                                              # channels_out=96,
                                                              # channels_out=64,
                                                              # channels_out=24,
                                                              # channels_out=64,
                                                              # channels_out=48,
                                                              # channels_out=64,
                                                              channels_out=32,
                                                              stride_r=1,
                                                              stride_theta=1,
                                                              stride_phi=1,
                                                              name_suffix="_high_res"))
            self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['conv'])
            self.layers[-1].update(self.create_avgpool_layer(len(self.layers)-1,
                                                             self.layers[-1]['activation'],
                                                             ksize=[1,1,3,3,1],
                                                             strides=[1,1,2,2,1]))
            
            if not train_high_res_model:
                self.layers[-1]['pool'] = tf.stop_gradient(self.layers[-1]['pool'])
                
            print("layer %s (high res) - activation: %s" % (len(self.layers), self.layers[-1]['activation'].shape), file=sys.stderr)
            print("layer %s (high res) - pooling: %s" % (len(self.layers), self.layers[-1]['pool'].shape), file=sys.stderr)

        ### LAYER 3 - low res ###
        if include_low_res_model:
            self.layers_low_res.append({})
            # self.layers_low_res[-1].update(self.create_conv3D_separate_r_layer(len(self.layers_low_res)-1,
            self.layers_low_res[-1].update(self.create_spherical_conv(len(self.layers_low_res)-1,
                                                                      # self.layers[-2]['pool'],
                                                                      self.layers_low_res[-2]['pool'],
                                                                      # merged_layer,
                                                                      window_size_r=1,
                                                                      window_size_theta=3,
                                                                      window_size_phi=3,
                                                                      # window_size_theta=2,
                                                                      # window_size_phi=2,
                                                                      # channels_out=192,
                                                                      # channels_out=128,
                                                                      # channels_out=96,
                                                                      # channels_out=64,
                                                                      channels_out=24,
                                                                      # channels_out=48,
                                                                      # channels_out=24,
                                                                      stride_r=1,
                                                                      stride_theta=1,
                                                                      stride_phi=1,
                                                                      name_suffix="_low_res"))
            self.layers_low_res[-1]['activation'] = tf.nn.relu(self.layers_low_res[-1]['conv'])
            self.layers_low_res[-1].update(self.create_avgpool_layer(len(self.layers_low_res)-1,
                                                                     self.layers_low_res[-1]['activation'],
                                                                     ksize=[1,3,3,3,1],
                                                                     strides=[1,2,2,2,1]))
            
            if not train_low_res_model:
                self.layers[-1]['pool'] = tf.stop_gradient(self.layers[-1]['pool'])
                
            print("layer %s (low res) - activation: %s" % (len(self.layers_low_res), self.layers_low_res[-1]['activation'].shape), file=sys.stderr)
            print("layer %s (low res) - pooling: %s" % (len(self.layers_low_res), self.layers_low_res[-1]['pool'].shape), file=sys.stderr)


        if include_high_res_model and include_low_res_model:
            # merged_layer = tf.concat((self.layers[-1]['pool'], self.x_low_res), axis=-1)
            merged_layer = tf.concat((self.layers[-1]['pool'], self.layers_low_res[-1]['pool']), axis=-1)
            print("Merged layer : %s" % (merged_layer.shape), file=sys.stderr)
        elif include_high_res_model:
            merged_layer = self.layers[-1]['pool']
        elif include_low_res_model:
            # merged_layer = self.x_low_res
            merged_layer = self.layers_low_res[-1]['pool']
            
        
        ### LAYER 4 ###
        self.layers.append({})
        # self.layers[-1].update(self.create_conv3D_separate_r_layer(len(self.layers)-1,
        self.layers[-1].update(self.create_spherical_conv(len(self.layers)-1,
                                                          # self.layers[-2]['pool'],
                                                          merged_layer,
                                                          window_size_r=3,
                                                          window_size_theta=3,
                                                          window_size_phi=3,
                                                          # window_size_theta=2,
                                                          # window_size_phi=2,
                                                          # channels_out=128,
                                                          # channels_out=32,
                                                          # channels_out=48,
                                                          # channels_out=24,
                                                          # channels_out=128,
                                                          # channels_out=128,
                                                          channels_out=64,
                                                          stride_r=1,
                                                          stride_theta=1,
                                                          stride_phi=1))
        self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['conv'])
        self.layers[-1].update(self.create_avgpool_layer(len(self.layers)-1,
                                                         self.layers[-1]['activation'],
                                                         ksize=[1,1,1,3,1],
                                                         strides=[1,1,1,2,1]))
        print("layer %s (high res) - activation: %s" % (len(self.layers), self.layers[-1]['activation'].shape), file=sys.stderr)
        print("layer %s (high res) - pooling: %s" % (len(self.layers), self.layers[-1]['pool'].shape), file=sys.stderr)


        
        ### LAYER 5 ###
        self.layers.append({})
        self.layers[-1].update(self.create_dense_layer(len(self.layers)-1,
                                                       self.layers[-2]['pool'],
                                                       output_size=2048))
        self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['dense'])
        # self.layers[-1].update(self.create_dropout_layer(len(self.layers)-1,
        #                                                  self.layers[-1]['activation'],
        #                                                  self.dropout_keep_prob))
        self.layers[-1]['dropout'] = tf.nn.dropout(self.layers[-1]['activation'], self.dropout_keep_prob)
        print("layer %s (high res) - W: %s" % (len(self.layers), self.layers[-1]['W'].get_shape()), file=sys.stderr)
        print("layer %s (high res) - activation: %s" % (len(self.layers), self.layers[-1]['activation'].shape), file=sys.stderr)

            
            
        ### LAYER 6 ###
        self.layers.append({})
        self.layers[-1].update(self.create_dense_layer(len(self.layers)-1,
                                                       self.layers[-2]['dropout'],
                                                       output_size=-1))
        self.layers[-1]['activation'] = tf.nn.relu(self.layers[-1]['dense'])
        # self.layers[-1].update(self.create_dropout_layer(len(self.layers)-1,
        #                                                  self.layers[-1]['activation'],
        #                                                  self.dropout_keep_prob))
        self.layers[-1]['dropout'] = tf.nn.dropout(self.layers[-1]['activation'], self.dropout_keep_prob)
        print("layer %s (high res) - W: %s" % (len(self.layers), self.layers[-1]['W'].get_shape()), file=sys.stderr)
        print("layer %s (high res) - activation: %s" % (len(self.layers), self.layers[-1]['activation'].shape), file=sys.stderr)


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

        self.loss += tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name ]) * regularization_factor
        
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
            optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)


        # Variables for accumulating gradients
        trainable_vars = tf.trainable_variables()
        trainable_vars_high_res = [v for v in trainable_vars if "high_res" in v.name]
        trainable_vars_low_res = [v for v in trainable_vars if "low_res" in v.name]

        if accumulate_gradients:
            # trainable_vars_remainder = [v for v in trainable_vars if ("low_res" not in v.name and "high_res" not in v.name)]

            # if train_high_res_model and train_low_res_model:
            #     pass
            # elif train_high_res_model:
            #     trainable_vars = trainable_vars_remainder+trainable_vars_high_res
            # elif train_low_res_model:
            #     trainable_vars = trainable_vars_remainder+trainable_vars_low_res            

            accumulated_gradients = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in trainable_vars]

            # Operation for resetting gradient variables after each batch
            self.zero_op = [var.assign(tf.zeros_like(var)) for var in accumulated_gradients]

            # Compute gradients
            gradients = tf.gradients(self.loss, trainable_vars)

            # Accumulation operator
            # self.gradient_accumulation_op = []
            # for i,gradient in enumerate(gradients):
            #     print i,gradient
            #     self.gradient_accumulation_op.append(accumulated_gradients[i].assign_add(gradient))
            self.gradient_accumulation_op = [accumulated_gradients[i].assign_add(gradient) for i, gradient in enumerate(gradients) if gradient is not None]

            # Normalize gradients
            self.gradient_normalization_constant = tf.placeholder(tf.float32, shape=(), name="gradient_normalization_contant")
            gradients_normalized = np.asarray(accumulated_gradients) / np.asarray(self.gradient_normalization_constant)

            if not optimize_using_lbfgs:
                self.train_step = optimizer.apply_gradients(list(zip(gradients_normalized, trainable_vars)))
        else:
            self.train_step = optimizer.minimize(self.loss)


        # print >> sys.stderr, "Number of parameters: ", sum(reduce(lambda x, y: x*y, v.get_shape().as_list()) for v in tf.trainable_variables())
        print("Number of parameters: ", sum(reduce(lambda x, y: x*y, v.get_shape().as_list()) for v in trainable_vars), file=sys.stderr)
        
        # config = tf.ConfigProto()
        # config.gpu_options.allocator_type = 'BFC'        
        # self.session = tf.InteractiveSession(config=config)
        self.session = tf.InteractiveSession()
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        print("Variables initialized", file=sys.stderr)

        self.saver = tf.train.Saver(max_to_keep=2)
        self.saver_high_res = None
        if include_high_res_model:        
            self.saver_high_res = tf.train.Saver(trainable_vars_high_res, max_to_keep=2)
        self.saver_low_res = None
        if include_low_res_model:        
            self.saver_low_res = tf.train.Saver(trainable_vars_low_res, max_to_keep=2)
        
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
                             keep_prob,
                             name_suffix=""):
        random_tensor = tf.placeholder(tf.float32, input.shape, name="dropout_random_tensor%d%s"%(index,name_suffix))
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
                             strides,
                             name_suffix=""):
        # Pad input with periodic image - only in phi
        # padded_input = tf_pad_wrap.tf_pad_wrap(input, [(0,0), (0,0), (ksize[2]/2, ksize[2]/2), (ksize[3]/2, ksize[3]/2), (0,0)])
        padded_input = tf_pad_wrap.tf_pad_wrap(input, [(0, 0), (0, 0), (0, 0), (ksize[3] / 2, ksize[3] / 2), (0, 0)])

        return {'pool': tf.nn.max_pool3d(padded_input,
                                         ksize=ksize,
                                         strides=strides,
                                         padding='VALID')}
        
    @staticmethod
    def create_avgpool_layer(index,
                             input,
                             ksize,
                             strides,
                             name_suffix=""):
        # Pad input with periodic image - only in phi
        # padded_input = tf_pad_wrap.tf_pad_wrap(input, [(0,0), (0,0), (ksize[2]/2, ksize[2]/2), (ksize[3]/2, ksize[3]/2), (0,0)])
        padded_input = tf_pad_wrap.tf_pad_wrap(input, [(0, 0), (0, 0), (0, 0), (ksize[3] / 2, ksize[3] / 2), (0, 0)])

        return {'pool': tf.nn.avg_pool3d(padded_input,
                                         ksize=ksize,
                                         strides=strides,
                                         padding='VALID')}
        
    @staticmethod
    def create_dense_layer(index,
                           input,
                           output_size,
                           name_suffix=""):
        reshaped_input = tf.reshape(input, [-1, np.prod(input.get_shape().as_list()[1:])])

        if output_size == -1:
            output_size = reshaped_input.get_shape().as_list()[1]

        W = tf.Variable(tf.truncated_normal([reshaped_input.get_shape().as_list()[1], output_size], stddev=0.1), name="W%d%s"%(index,name_suffix))
        b = tf.Variable(tf.truncated_normal([output_size], stddev=0.1), name="bias%d%s"%(index,name_suffix))
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
                            stride_phi=1,
                            name_suffix=""):

        # Pad input with periodic image - only in phi
        # padded_input = tf_pad_wrap.tf_pad_wrap(input, [(0,0), (0,0), (window_size_theta/2, window_size_theta/2), (window_size_phi/2, window_size_phi/2), (0,0)])
        padded_input = tf_pad_wrap.tf_pad_wrap(input, [(0, 0), (0, 0), (0, 0), (window_size_phi / 2, window_size_phi / 2), (0, 0)])
                                   
        filter_shape = [window_size_r, window_size_theta, window_size_phi, padded_input.get_shape().as_list()[-1], channels_out] 
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W%d%s"%(index, name_suffix))
        b = tf.Variable(tf.truncated_normal(filter_shape[-1:], stddev=0.1), name="bias%d%s"%(index,name_suffix))
        conv = tf.tf.nn.bias_add(
            tf.nn.conv3d(padded_input,
                         W,
                         strides=[1, stride_r, stride_theta, stride_phi, 1],
                         padding="VALID",
                         name="conv%d%s"%(index,name_suffix)),
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
                                       stride_phi=1,
                                       name_suffix=""):

        # Create convolutions for each r value
        convs = []
        for i in range(window_size_r/2, input.shape[1]-window_size_r/2, stride_r):

            input_fixed_r = input[:,i-window_size_r/2:i+window_size_r/2+1,:,:,:]
        
            # Pad input with periodic image - only in phi
            # padded_input = tf_pad_wrap.tf_pad_wrap(input_fixed_r, [(0,0), (0,0), (window_size_theta/2, window_size_theta/2), (window_size_phi/2, window_size_phi/2), (0,0)])
            padded_input = tf_pad_wrap.tf_pad_wrap(input_fixed_r, [(0, 0), (0, 0), (0, 0), (window_size_phi / 2, window_size_phi / 2), (0, 0)])

            filter_shape = [window_size_r, window_size_theta, window_size_phi, padded_input.get_shape().as_list()[-1], channels_out]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_%d_%d%s"%(i,index,name_suffix))
            b = tf.Variable(tf.truncated_normal(filter_shape[-1:], stddev=0.1), name="bias%d%s"%(i,name_suffix))
            convs.append(
                tf.nn.bias_add(
                    tf.nn.conv3d(padded_input,
                                 W,
                                 strides=[1, 1, stride_theta, stride_phi, 1],
                                 padding="VALID",
                                 name="conv_%d_%d%s"%(i,index,name_suffix)),
                    b))
        conv = tf.concat(convs, axis=1)
        return {'W':W, 'b':b, 'conv':conv}
    
        
    @staticmethod
    def create_spherical_conv(index,
                              input,
                              window_size_r,
                              window_size_theta,
                              window_size_phi,
                              channels_out,
                              stride_r=1,
                              stride_theta=1,
                              stride_phi=1,
                              name_suffix=""):

        # Create convolutions for each r value
        convs = []
        for i in range(window_size_r/2, input.shape[1]-window_size_r/2, stride_r):

            convs.append([])
            for j in range(window_size_theta/2, input.shape[2]-window_size_theta/2, stride_theta):
            
                input_fixed_r_theta = input[:,i-window_size_r/2:i+window_size_r/2+1,j-window_size_theta/2:j+window_size_theta/2+1,:,:]

                # Pad input with periodic image - only in phi
                padded_input = tf_pad_wrap.tf_pad_wrap(input_fixed_r_theta, [(0, 0), (0, 0), (0, 0), (window_size_phi / 2, window_size_phi / 2), (0, 0)])

                filter_shape = [window_size_r, window_size_theta, window_size_phi, padded_input.get_shape().as_list()[-1], channels_out]

                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_%d_%d%s"%(i,index,name_suffix))
                b = tf.Variable(tf.truncated_normal(filter_shape[-1:], stddev=0.1), name="bias%d%s"%(i,name_suffix))
                convs[-1].append(
                    tf.nn.bias_add(
                        tf.nn.conv3d(padded_input,
                                     W,
                                     strides=[1, 1, 1, stride_phi, 1],
                                     padding="VALID",
                                     name="spherical_conv_%d_%d_%d%s"%(i,j,index,name_suffix)),
                        b))
            convs[-1] = tf.concat(convs[-1], axis=2)
        conv = tf.concat(convs, axis=1)
        return {'W':W, 'b':b, 'conv':conv}
    
        
    def train(self,
              train_batch_factory, 
              num_passes=100,
              validation_batch_factory=None,
              output_interval=None,
              dropout_keep_prob=0.5,
              enforce_protein_boundaries=False,
              dry_run=False):

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
                                                                 enforce_protein_boundaries=enforce_protein_boundaries)
                grid_matrix = None
                if "high_res" in batch:
                    grid_matrix = batch["high_res"]

                low_res_grid_matrix = None
                if "low_res" in batch:
                    low_res_grid_matrix = batch["low_res"]

                labels = batch["model_output"]

                data_size += labels.shape[0]

                if dry_run:
                    # print i+1, iteration+1, data_size, total_data_size, batch_factory.feature_index

                    if (iteration+1) % output_interval == 0:
                        validation_batch, validation_gradient_batch_sizes = validation_batch_factory.next(validation_batch_factory.data_size(),
                                                                                                          subbatch_max_size=self.max_gradient_batch_size,
                                                                                                          enforce_protein_boundaries=enforce_protein_boundaries)
                    iteration += 1
                    continue
                
                if self.optimize_using_lbfgs:

                    if not self.accumulated_gradients:
                        raise "Not yet implemented"
                    
                    def loss_callback(loss, _):
                        print("%s %s Loss: %s" % (i+1, iteration+1, loss), file=sys.stderr)

                    self.optimizer.minimize(self.session,# fetches=[self.loss], loss_callback=optimizer_callback,
                                            loss_callback=loss_callback,
                                            feed_dict=dict(list({self.x_high_res: grid_matrix,
                                                            self.x_low_res: low_res_grid_matrix,
                                                            self.y: labels,
                                                            "gradient_batch_sizes": gradient_batch_sizes,
                                                            self.dropout_keep_prob:dropout_keep_prob}.items())
                                                           + list(self.generate_dropout_feed_dict(self.layers, labels.shape[0]).items())))

                    # loss_tmp = 0
                    # for index, length in zip(np.cumsum(gradient_batch_sizes)-gradient_batch_sizes, gradient_batch_sizes):
                    #     grid_matrix_batch, low_res_grid_matrix_batch, labels_batch = \
                    #          get_batch(index, index+length,
                    #                    grid_matrix, low_res_grid_matrix, labels)
                    #     print grid_matrix_batch.shape
                    #     loss_tmp += self.session.run(self.loss,
                    #                                  feed_dict=dict({self.x_high_res: grid_matrix_batch,
                    #                                                  self.x_low_res: low_res_grid_matrix_batch,
                    #                                                  self.y: labels_batch,
                    #                                                  self.dropout_keep_prob:dropout_keep_prob}.items()
                    #                                                 + self.generate_dropout_feed_dict(self.layers, length).items()))
                    # print "loss_tmp: " , loss_tmp/float(len(gradient_batch_sizes))
                    
                else:

                    if not self.accumulate_gradients:
                        _, accumulated_loss_value = self.session.run([self.train_step, self.loss],
                                                                     feed_dict={self.x_high_res: grid_matrix,
                                                                                self.x_low_res: low_res_grid_matrix,
                                                                                self.y: labels,
                                                                                self.dropout_keep_prob:dropout_keep_prob})
                        
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

                        self.session.run([self.train_step],
                                         feed_dict={self.gradient_normalization_constant: float(len(gradient_batch_sizes))})

                        # Clear the accumulated gradients
                        self.session.run(self.zero_op)

                    print("%s %s Loss: %s" % (i+1, iteration+1,  accumulated_loss_value/float(len(gradient_batch_sizes))), file=sys.stderr)

                    
                if (iteration+1) % output_interval == 0:
                    Q_training_batch = self.Q_accuracy(batch, gradient_batch_sizes, raw=True)
                    print("%s Q%s score (training batch): %s" % (iteration+1, self.output_size, Q_training_batch), file=sys.stderr)

                    validation_batch, validation_gradient_batch_sizes = validation_batch_factory.next(validation_batch_factory.data_size(),
                                                                                                      subbatch_max_size=self.max_gradient_batch_size,
                                                                                                      enforce_protein_boundaries=enforce_protein_boundaries)
                    Q_validation = self.Q_accuracy(validation_batch, validation_gradient_batch_sizes)
                    print("%s Q%s score (validation set): %s" % (iteration+1, self.output_size, Q_validation), file=sys.stderr)

                    self.save(self.model_checkpoint_path, iteration)

                    
                iteration += 1    

                
    def save(self, checkpoint_path, step):

        checkpoint_path_high_res = os.path.join(checkpoint_path, "high_res")
        checkpoint_path_low_res = os.path.join(checkpoint_path, "low_res")
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        if self.saver_high_res is not None and not os.path.exists(checkpoint_path_high_res):
            os.mkdir(checkpoint_path_high_res)
        if self.saver_low_res is not None and not os.path.exists(checkpoint_path_low_res):
            os.mkdir(checkpoint_path_low_res)

        self.saver.save(self.session, os.path.join(checkpoint_path, 'model.ckpt'), global_step=step)
        if self.saver_high_res is not None:
            self.saver_high_res.save(self.session, os.path.join(checkpoint_path_high_res, 'model.ckpt'), global_step=step)
        if self.saver_low_res is not None:
            self.saver_low_res.save(self.session, os.path.join(checkpoint_path_low_res, 'model.ckpt'), global_step=step)
        print("Model saved", file=sys.stderr)

    def restore(self, checkpoint_path):

        ckpt = tf.train.get_checkpoint_state(checkpoint_path)

        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.session, tf.train.latest_checkpoint(checkpoint_path))
        else:
            print("Could not load file", checkpoint_path, file=sys.stderr)    

            
    def restore_high_res(self, checkpoint_path):

        checkpoint_path = os.path.join(checkpoint_path, "high_res")
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)

        if ckpt and ckpt.model_checkpoint_path:
            self.saver_high_res.restore(self.session, tf.train.latest_checkpoint(checkpoint_path))
        else:
            print("Could not load file", checkpoint_path, file=sys.stderr)

            
    def restore_low_res(self, checkpoint_path):

        checkpoint_path = os.path.join(checkpoint_path, "low_res")
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)

        if ckpt and ckpt.model_checkpoint_path:
            self.saver_low_res.restore(self.session, tf.train.latest_checkpoint(checkpoint_path))
        else:
            print("Could not load file", checkpoint_path, file=sys.stderr)    
        
            
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
        # np.set_printoptions(threshold=np.nan)
        # print predictions[:20]
        # print y[:20]
        predictions = tf.argmax(predictions, 1).eval()
        y_argmax = tf.argmax(y, 1).eval()
        identical = (predictions==y_argmax)
        return np.mean(identical)
    

if __name__ == '__main__':

    from argparse import ArgumentParser

    from utils import str2bool
    
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
    parser.add_argument("--train-high-res-model", dest="train_high_res_model",
                        type=str2bool, nargs='?', const=True, default="True",
                        help="Whether to train atomic-resolution part of model")
    parser.add_argument("--train-low-res-model", dest="train_low_res_model",
                        type=str2bool, nargs='?', const=True, default="True",
                        help="Whether to train low resolution part of model")
    parser.add_argument("--optimize-using-lbfgs", dest="optimize_using_lbfgs",
                        type=str2bool, nargs='?', const=True, default="False",
                        help="Whether to use the LBFGS optimizer")
    parser.add_argument("--lbfgs-max-iterations", type=int, default=10, 
                        help="Number of iterations in each BFGS step")
    parser.add_argument("--max-batch-size", dest="max_batch_size",
                        help="Maximum batch size used during training", type=int, default=1000)
    parser.add_argument("--output-interval", dest="output_interval",
                        help="How often to output during training", type=int, default=None)
    parser.add_argument("--max-gradient-batch-size", dest="max_gradient_batch_size",
                        help="Maximum batch size used for gradient calculation", type=int, default=25)
    parser.add_argument("--model-checkpoint-path", dest="model_checkpoint_path",
                        help="Where to dump/read model checkpoints", default="models")
    parser.add_argument("--read-from-checkpoint", action="store_true", dest="read_from_checkpoint",
                        help="Whether to read model from checkpoint")
    parser.add_argument("--read-from-high-res-checkpoint", action="store_true", dest="read_from_high_res_checkpoint",
                        help="Whether to read model from high resolution checkpoint")
    parser.add_argument("--read-from-low-res-checkpoint", action="store_true", dest="read_from_low_res_checkpoint",
                        help="Whether to read model from low resolution checkpoint")
    parser.add_argument("--mode", choices=['train', 'test'], dest="mode", default="train", 
                        help="Mode of operation: train or test")
    parser.add_argument("--model-output-type", choices=['aa', 'ss'], dest="model_output_type", default="ss", 
                        help="Whether the model should output secondary structure or amino acid labels")
    parser.add_argument("--dropout-keep-prob", dest="dropout_keep_prob", type=float, default=0.5, 
                        help="Probability for leaving out node in dropout")
    parser.add_argument("--regularization-factor", dest="regularization_factor", type=float, default=0.001, 
                        help="Regularization factor for L2 regularization")
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=0.0001, 
                        help="Optimization learning rate")
    parser.add_argument("--accumulate-gradients", dest="accumulate_gradients",
                        type=str2bool, nargs='?', const=True, default="False",
                        help="Whether to accumulate small gradient batches during training")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", default=False,
                        help="Run through data without doing actual training")
    parser.add_argument("--enforce-protein-boundaries", action="store_true", dest="enforce_protein_boundaries", default=False,
                        help="Whether to ensure that BatchFactory never returns a batch containing elements from multiple proteins")
    parser.add_argument("--duplicate-origin",
                        type=str2bool, nargs='?', const=True, default="True",
                        help="Whether to duplicate the atoms in all bins at the origin for the spherical model")

    options = parser.parse_args()
    
    high_res_protein_feature_filenames = []
    high_res_grid_feature_filenames = []
    if options.high_res_features_input_dir is not None:
        high_res_protein_feature_filenames = sorted(glob.glob(os.path.join(options.high_res_features_input_dir, "*protein_features.npz")))
        high_res_grid_feature_filenames = sorted(glob.glob(os.path.join(options.high_res_features_input_dir, "*residue_features.npz")))

    low_res_protein_feature_filenames = []
    low_res_grid_feature_filenames = []
    if options.low_res_features_input_dir is not None:
        low_res_protein_feature_filenames = sorted(glob.glob(os.path.join(options.low_res_features_input_dir, "*protein_features.npz")))
        low_res_grid_feature_filenames = sorted(glob.glob(os.path.join(options.low_res_features_input_dir, "*residue_features.npz")))

    # train_end = test_start = int(len(protein_feature_filenames)-10)
    validation_end = test_start = int(len(high_res_protein_feature_filenames)*(1.-options.test_set_fraction))
    train_end = validation_start = int(validation_end-options.validation_set_size)

    # exclude_at_center = []
    # if options.model_output_type == "aa":
    #     exclude_at_center = ["aa_one_hot_w_unobserved"]

    batch_factory = BatchFactory()
    if options.high_res_features_input_dir is not None:
        batch_factory.add_data_set("high_res",
                                   high_res_protein_feature_filenames[:train_end],
                                   high_res_grid_feature_filenames[:train_end],
                                   duplicate_origin=options.duplicate_origin)
    if options.low_res_features_input_dir is not None:
        batch_factory.add_data_set("low_res",
                                   low_res_protein_feature_filenames[:train_end],
                                   low_res_grid_feature_filenames[:train_end],
                                   # exclude_at_center = exclude_at_center,
                                   duplicate_origin=options.duplicate_origin)
    if options.high_res_features_input_dir is not None:
        batch_factory.add_data_set("model_output",
                                   high_res_protein_feature_filenames[:train_end],
                                   key_filter=[options.model_output_type+"_one_hot"],
                                   duplicate_origin=options.duplicate_origin)
    elif options.low_res_features_input_dir is not None:
        batch_factory.add_data_set("model_output",
                                   low_res_protein_feature_filenames[:train_end],
                                   key_filter=[options.model_output_type+"_one_hot"],
                                   duplicate_origin=options.duplicate_origin)
        
        
    
    validation_batch_factory = BatchFactory()
    if options.high_res_features_input_dir is not None:
        validation_batch_factory.add_data_set("high_res",
                                              high_res_protein_feature_filenames[validation_start:validation_end],
                                              high_res_grid_feature_filenames[validation_start:validation_end],
                                              duplicate_origin=options.duplicate_origin)
    if options.low_res_features_input_dir is not None:
        validation_batch_factory.add_data_set("low_res",
                                              low_res_protein_feature_filenames[validation_start:validation_end],
                                              low_res_grid_feature_filenames[validation_start:validation_end],
                                              # exclude_at_center = exclude_at_center,
                                              duplicate_origin=options.duplicate_origin)
    if options.high_res_features_input_dir is not None:
        validation_batch_factory.add_data_set("model_output",
                                              high_res_protein_feature_filenames[validation_start:validation_end],
                                              key_filter=[options.model_output_type+"_one_hot"],
                                              duplicate_origin=options.duplicate_origin)
    elif options.low_res_features_input_dir is not None:
        validation_batch_factory.add_data_set("model_output",
                                              low_res_protein_feature_filenames[validation_start:validation_end],
                                              key_filter=[options.model_output_type+"_one_hot"],
                                              duplicate_origin=options.duplicate_origin)
    
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
    high_res_grid_size = (0,0,0,0,0)
    if options.high_res_features_input_dir is not None:
        high_res_grid_size = batch_factory.next(1, increment_counter=False)[0]["high_res"].shape
    low_res_grid_size = (0,0,0,0,0)
    if options.low_res_features_input_dir is not None:
        low_res_grid_size = batch_factory.next(1, increment_counter=False)[0]["low_res"].shape

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
                  train_high_res_model    = options.train_high_res_model,
                  train_low_res_model     = options.train_low_res_model,
                  optimize_using_lbfgs    = options.optimize_using_lbfgs,
                  lbfgs_max_iterations    = options.lbfgs_max_iterations,
                  model_checkpoint_path   = options.model_checkpoint_path,
                  regularization_factor   = options.regularization_factor,
                  learning_rate           = options.learning_rate,
                  accumulate_gradients    = options.accumulate_gradients)

    if options.read_from_checkpoint:
        model.restore(options.model_checkpoint_path)
    if options.read_from_high_res_checkpoint:
        model.restore_high_res(options.model_checkpoint_path)
    if options.read_from_low_res_checkpoint:
        model.restore_low_res(options.model_checkpoint_path)
    
    # model.train(high_res_training_batch_factory   = high_res_training_batch_factory,
    #             low_res_training_batch_factory    = low_res_training_batch_factory,
    #             high_res_validation_batch_factory = high_res_validation_batch_factory,
    #             low_res_validation_batch_factory  = low_res_validation_batch_factory,
    #             model_output_batch_factory        = model_output_batch_factory)
    if options.mode == 'train':
        model.train(train_batch_factory      = batch_factory,
                    validation_batch_factory = validation_batch_factory,
                    output_interval          = options.output_interval,
                    dropout_keep_prob        = options.dropout_keep_prob,
                    dry_run                  = options.dry_run)

    elif options.mode == 'test':
        validation_batch, validation_gradient_batch_sizes = validation_batch_factory.next(validation_batch_factory.data_size(),
                                                                                          subbatch_max_size=options.max_gradient_batch_size,
                                                                                          enforce_protein_boundaries=options.enforce_protein_boundaries)
        model.infer(validation_batch, validation_gradient_batch_sizes)
        Q_validation = model.Q_accuracy(validation_batch, validation_gradient_batch_sizes)
        print("Q%s score (validation set): %s" % (output_size, Q_validation), file=sys.stderr)

import tensorflow as tf
import tensorflow_contrib.crf.python.ops.crf as crf
import os.path
import numpy as np

def next_batch(x, i, j):
    return x[i:j]

class Model:

    def __init__(self, input_size, output_size, max_length, layers=3, filter_size=11, filter_depth=10, crf_output_layer=False, regularization_factor=0.001, optimize_using_lbfgs=False, lbfgs_maxiter=100):

        self.optimize_using_lbfgs = optimize_using_lbfgs
        self.crf_output_layer = crf_output_layer
        self.session = tf.InteractiveSession()
        
        self.x = tf.placeholder(tf.float32, [None, max_length, input_size])
        self.y = tf.placeholder(tf.float32, [None, max_length, output_size])
        self.y_argmax = tf.placeholder(tf.int32, [None, max_length])
        self.sequence_lengths = tf.placeholder(tf.int64, [None])

        # Convolution Layers
        self.Ws = []
        self.bs = []
        self.convs = []
        self.activations = []

        for i in range(0, layers):

            filter_shape = [filter_size, filter_depth, filter_depth]
            if i==0:
                filter_shape[1] = input_size
            if i==layers-1:
                filter_shape[2] = output_size
                if self.crf_output_layer:
                    filter_shape[0] = 1
            
            value = None
            if i==0:
                value = self.x
            else:
                value = self.activations[i-1]
                
            self.Ws.append(tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W%d"%i))
            self.bs.append(tf.Variable(tf.truncated_normal(filter_shape[-1:], stddev=0.1), name="b%d"%i))
            self.convs.append(tf.nn.bias_add(tf.nn.conv1d(
                value,
                self.Ws[-1],
                stride=1,
                padding="SAME",
                name="conv%d"%i), self.bs[-1]))

            if i < (layers-1):
                self.activations.append(tf.nn.relu(self.convs[-1]))
                # self.activations.append(tf.nn.tanh(self.convs[-1]))
                # self.activations.append(tf.nn.sigmoid(self.convs[-1]))
            else:
                if crf_output_layer:
                    self.activations.append(tf.nn.tanh(self.convs[-1]))
                    # self.activations.append(tf.nn.relu(self.convs[-1]))
                    # self.activations.append(tf.nn.softmax(self.convs[-1]))
                else:
                    # self.activations.append(self.convs[-1])
                    self.activations.append(tf.nn.softmax(self.convs[-1]))

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.convs[-1], labels=self.y))

        # # In case loss is changed - we can still evaluate the loss for the nn part individually
        # self.loss_nn = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits(logits=self.convs[-1], labels=self.y))

        if crf_output_layer:
            # self.weights_crf = tf.Variable(tf.truncated_normal([output_size, output_size], stddev=0.1), name="Ws_crf")
            self.weights_crf = tf.Variable(tf.eye(output_size), name="W_crf")
            # self.weights_crf = tf.constant(np.zeros([output_size, output_size]).astype(np.float32))
            log_likelihood, self.transition_params, self.seq_scores = crf.crf_log_likelihood(
                self.activations[-1], self.y_argmax, self.sequence_lengths, self.weights_crf)
            self.loss = tf.reduce_mean(-log_likelihood)
            
        # Add regularization (should be estimated using cross validation)
        # Note, regularization should not be applied on biases (but we have none here, so it's ok)
        # self.loss_nn += tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()
        #                            if 'crf' not in v.name]) * regularization_factor
        self.loss += tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables() ]) * regularization_factor

        if self.optimize_using_lbfgs:

            from tensorflow.contrib import opt
            
            self.optimizer = opt.ScipyOptimizerInterface(self.loss, 
		                                         method='L-BFGS-B', options={'maxiter': lbfgs_maxiter})
        else:
            self.train_step = tf.train.AdamOptimizer(0.001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss)
        # self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.saver = tf.train.Saver(max_to_keep=1)
        

        
    def train(self, X, y, model_checkpoint_path="models", sequence_lengths=None, batch_size=10, num_passes=10, dump_frequency=100, test_X=None, test_y=None):

        if self.crf_output_layer:
            y_argmax = tf.cast(tf.argmax(y, 2), tf.int32).eval()

        if self.optimize_using_lbfgs:
            from tensorflow.contrib import opt
            # def optimizer_callback(loss):
            #     print loss
            
        for j in range(num_passes/(X.shape[0]/batch_size)):
            for i in range(X.shape[0]/batch_size):

                batch_xs, batch_ys = (next_batch(X, i*batch_size, (i+1)*batch_size),
                                      next_batch(y, i*batch_size, (i+1)*batch_size))
                if self.crf_output_layer:
                    batch_y_argmaxs = next_batch(y_argmax, i*batch_size, (i+1)*batch_size)
                    batch_sl = next_batch(sequence_lengths, i*batch_size, (i+1)*batch_size)

                    if self.optimize_using_lbfgs:
                        self.optimizer.minimize(self.session,# fetches=[self.loss], loss_callback=optimizer_callback,
                                                feed_dict={self.x: batch_xs,
                                                           # self.y: batch_ys,
                                                           self.y_argmax: batch_y_argmaxs,
                                                           self.sequence_lengths: batch_sl})
                    else:
                        self.session.run([self.train_step],
                                         feed_dict={self.x: batch_xs,
                                                    # self.y: batch_ys,
                                                    self.y_argmax: batch_y_argmaxs,
                                                    self.sequence_lengths: batch_sl})
                else:

                    if self.optimize_using_lbfgs:
                        self.optimizer.minimize(self.session, #fetches=[self.loss], loss_callback=optimizer_callback,
                                                feed_dict={self.x: batch_xs,
                                                           self.y: batch_ys})
                    else:
                        self.session.run([self.train_step],
                                         feed_dict={self.x: batch_xs,
                                                    self.y: batch_ys})

                if ((j*X.shape[0]/batch_size)+i)%dump_frequency ==0:

                    iteration = (j*X.shape[0]/batch_size)+i
                    if self.crf_output_layer:
                        unary_scores, transition_matrix, loss_value = self.session.run([self.activations[-1], self.transition_params, self.loss],
                                                                                                      feed_dict={self.x: batch_xs,
                                                                                                                 self.y: batch_ys,
                                                                                                                 self.y_argmax: batch_y_argmaxs,
                                                                                                                 self.sequence_lengths: batch_sl})
                        print("transition_matrix:\n", transition_matrix)
                        print("loss: %s" % (loss_value))

                    else:
                        loss_value = self.session.run([self.loss],
                                                      feed_dict={self.x: batch_xs,
                                                                 self.y: batch_ys})
                        print("loss: %s" % (loss_value))
                    print("Q3 score (training batch): ", self.Q3_accuracy(batch_xs, batch_ys))
                    if test_X is not None and test_y is not None:
                        print("Q3 score (test set): ", self.Q3_accuracy(test_X, test_y))
                    self.save(model_checkpoint_path, iteration)

            # for i in range(num_passes):
            #     _, loss_val = self.session.run([self.train_step, self.loss],
            #                                    feed_dict={self.x: X, self.y: y})
            #     if i%dump_frequency == 0:
            #         print i, loss_val

                


    def infer(self, X):

        if self.crf_output_layer:
            unary_scores, transition_matrix = self.session.run([self.activations[-1], self.transition_params],
                                                               feed_dict={self.x: X})
            viterbi_sequences = np.zeros([unary_scores.shape[0], unary_scores.shape[1]])
            for i in range(unary_scores.shape[0]):
                viterbi_sequences[i], _ = tf.contrib.crf.viterbi_decode(
                    unary_scores[i], transition_matrix)
            return viterbi_sequences
        else:
            return self.session.run(self.activations[-1],
                                    feed_dict={self.x: X})

    def save(self, checkpoint_path, step):

        if self.crf_output_layer:
            checkpoint_path = checkpoint_path + "_crf"

        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        self.saver.save(self.session, os.path.join(checkpoint_path, 'model.ckpt'), global_step=step)
        print("Model saved")

    def restore(self, checkpoint_path):

        if self.crf_output_layer:
            checkpoint_path = checkpoint_path + "_crf"
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)

        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.session, tf.train.latest_checkpoint(checkpoint_path))
        else:
            print("Could not load file")

    def Q3_accuracy(self, X, y):

        predictions = self.infer(X)
        if not self.crf_output_layer:
            predictions = tf.argmax(predictions, 2).eval()
        y_argmax = tf.argmax(y, 2).eval()
        X_sel = (X[:,:,-1] != 1)
        predictions_sel = predictions[X_sel]
        y_sel = y_argmax[X_sel]
        return np.count_nonzero(y_sel == predictions_sel) / float(predictions_sel.size)
        

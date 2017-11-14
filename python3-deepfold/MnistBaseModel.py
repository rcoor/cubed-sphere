import numpy as np
import tensorflow as tf
import mnist_data
from functools import reduce

class MnistBaseModel:
    '''Model definition'''

    def __init__(self,
                 reg_fact,
                 *args, **kwargs):

        ### Define the model ###
        self._init_model(*args, **kwargs)

        ### Loss function ###
        self.entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.layers[-1]['dense'], labels=self.y)
        self.regularization = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if not v.name.startswith("b")]) * reg_fact

        self.loss = tf.reduce_mean(self.entropy) + self.regularization

        print("Number of parameters: ", sum(reduce(lambda x, y: x*y, v.get_shape().as_list()) for v in tf.trainable_variables()))

        ### OPTIMIZER ###
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

        ### Session and saver ###
        #self.saver = tf.train.Saver(max_to_keep=max_to_keep)
        self.sess = tf.Session()

        # Setup prediction part
        self.prediction = tf.argmax(self.layers[-1]['dense'], 1)
        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.layers[-1]['dense'], 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        ### Initialize variables ###
        tf.global_variables_initializer().run(session=self.sess)
        print("Variables initialized")

    def _init_model(self):
        raise NotImplementedError

    def train(self, X, y, batch_size=128, epochs=50, dropout_keep_prob=0.5):

        with self.sess.as_default():
            tf.global_variables_initializer().run()

            for epoch in range(epochs):

                for b, (X_batch, y_batch) in enumerate(mnist_data.chunks([X, y], 128, shuffle=True)):
                    _, loss = self.sess.run([self.train_step, self.loss], feed_dict={self.X: X_batch,
                                                                                     self.y: y_batch,
                                                                                     self.dropout_keep_prob: dropout_keep_prob})
                    print("loss[%d,%d]=%f" %(epoch, b, loss))

                    if b % 50 == 0:
                        print("\tTrain accuracy: ", self.calc_accuracy(X_batch, y_batch))

                # Q_validation, loss_validation = self.Q_accuracy_and_loss(validation_batch, validation_gradient_batch_sizes)
                # print "[%d, %d] Q%s score (validation set) = %f" % (i, iteration,  self.output_size, Q_validation)
                # print "[%d, %d] loss (validation set) = %f" % (i, iteration, loss_validation)

    def predict(self, X):
        return self.sess.run(self.prediction, feed_dict={self.X: X, self.dropout_keep_prob: 1.0})

    def calc_accuracy(self, X, y):
        correct = []

        for b, (X_batch, y_batch) in enumerate(mnist_data.chunks([X, y], 128)):
            correct.append(self.sess.run(self.correct_prediction, feed_dict={self.X: X_batch, self.y: y_batch, self.dropout_keep_prob: 1.0}))

        return np.mean(np.concatenate(correct))

    @staticmethod
    def print_layer(layers, idx, name):
        if layers[-1][name].get_shape()[0].value is None:
            size = int(np.prod(layers[-1][name].get_shape()[1:]))
        else:
            size = int(np.prod(layers[-1][name].get_shape()))

        print("layer %2d: %10s: %s [size %s]" % (len(layers), name, layers[idx][name].get_shape(), "{:,}".format(size)))

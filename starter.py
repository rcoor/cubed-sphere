
from batch_factory.deepfold_batch_factory import BatchFactory

import argparse
import sys
import tempfile

import glob
import os
import time
import datetime

import numpy as np
import pandas as pd

from functools import reduce
import operator

from directional import conv_spherical_cubed_sphere, avg_pool_spherical_cubed_sphere
from batch_factory.deepfold_batch_factory import BatchFactory
import tensorflow as tf

FLAGS = None

class Deep_cnn(object):
    """deepnn builds the graph for a deep net for classifying residues.

    Args:
    x: An input tensor with the dimensions (batch_size, sides, radius, xi, eta, channels).
    y: Amount of classes to predict
    shape: Shape of the input tensor x

    Returns:
    A tuple (y, keep_prob). y is a tensor of shape (batch_size, n_classes), with values
    equal to the logits of classifying the digit into one of n-classes (the
    digits 0-20). keep_prob is a scalar placeholder for the probability of
    dropout.
    """
    def __init__(self, n_classes=21, shape=[None, 6, 24, 38, 38, 2]):
        self = self
        self.n_classes = n_classes
        self.shape = [None, 6,24,38,38,2]
        self.y_ = tf.placeholder(tf.float32, shape=[None, n_classes])
        self.x = tf.placeholder(tf.float32, shape=[None, 6,24,38,38,2])
        self.keep_prob = tf.placeholder(tf.float32)

    def build_graph(self):
        def conv_layer(input, channels_in, channels_out, name="conv"):
            with tf.name_scope(name):
                W = tf.Variable(tf.random_normal([3,3,3, channels_in, channels_out]), name="W")
                b = tf.Variable(tf.random_normal([channels_out]), name="b")
                conv = conv_spherical_cubed_sphere(input, W, strides=[1, 1, 1, 1, 1], padding="SAME")
                act = tf.nn.relu(conv + b)
                tf.summary.histogram("weights", W)
                tf.summary.histogram("biases", b)
                tf.summary.histogram("activations", act)
                return avg_pool_spherical_cubed_sphere(act, ksize=[1, 1, 3, 3, 1], strides=[1,1,2,2,1], padding="VALID")

        def reduce_dim(x):
            return reduce(operator.mul, x.shape.as_list()[1:], 1)

        def fc_layer(input, channels_in, channels_out, name="fc"):
            with tf.name_scope(name):
                W = tf.Variable(tf.random_normal([channels_in, channels_out]), name="W")
                b = tf.Variable(tf.random_normal([channels_out]), name="b")
                return tf.nn.relu(tf.matmul(input, W) + b)

        def out(input, channels_in, channels_out, name="out"):
            with tf.name_scope(name):
                W = tf.Variable(tf.random_normal([channels_in, channels_out]), name="W")
                b = tf.Variable(tf.random_normal([channels_out]), name="b")
                return tf.matmul(input, W) + b

        # Reshape to use within a convolutional neural net.
        x = tf.reshape(self.x, shape=[-1, 6, 24, 38, 38, 2])
        conv1 = conv_layer(x, 2, 16,  name="conv1")
        conv2 = conv_layer(conv1, 16, 32,  name="conv2")
        conv3 = conv_layer(conv2, 32, 64,  name="conv3")
        conv4 = conv_layer(conv3, 64, 128,  name="conv4")

        flattened = tf.reshape(conv4,  [-1, reduce_dim(conv4)])

        fc1 = fc_layer(flattened, reduce_dim(conv4), 2048)
        drop_out = tf.nn.dropout(fc1, self.keep_prob)
        fc2 = fc_layer(drop_out, 2048, 2048)
        return out(fc2, 2048, self.n_classes)

    def train_graph(self, train_batch_factory, validation_batch_factory, test_batch_factory):
        y_conv = self.build_graph()
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=y_conv)
            cross_entropy = tf.reduce_mean(cross_entropy)

        with tf.name_scope('learning_rate'):
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.1
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100, 0.96, staircase=True)

        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(0.001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cross_entropy, global_step=global_step)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)

        ### DEFINE SUMMARIES ###
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("cross_entropy", cross_entropy)
        merged = tf.summary.merge_all()

        time_str = datetime.datetime.now().isoformat()
        train_writer = tf.summary.FileWriter("./tmp/summary/train/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
        train_writer.add_graph(tf.get_default_graph())
        validation_writer = tf.summary.FileWriter("./tmp/summary/validation/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
        test_writer = tf.summary.FileWriter("./tmp/summary/test/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
        ### TRAIN MODEL ###
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #saver = tf.train.import_meta_graph('./model/model.cpkt.meta')
            if os.path.isfile("./model/model.cpkt"):
                saver.restore(sess, "./model/model.cpkt")
            # set variables
            prev_test_accuracy = 0.0

            for i in range(40000):
                batch, _ = train_batch_factory.next(10)
                _, summary = sess.run([train_step, merged], feed_dict={
                                    self.x: batch["data"], self.y_: batch["model_output"], self.keep_prob: 0.5})
                if i % 10 == 0:
                    # Train accuracy
                    summary, train_accuracy = sess.run([merged, accuracy], feed_dict={
                                                    self.x: batch["data"], self.y_: batch["model_output"], self.keep_prob: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                    train_writer.add_summary(summary, i)

                    # Validation accuracy
                    batch, _ = validation_batch_factory.next(10)
                    summary, validation_accuracy = sess.run([merged, accuracy],
                    feed_dict={
                                                            self.x: batch["data"], self.y_: batch["model_output"], self.keep_prob: 1.0})
                    print('step %d, validation accuracy %g' %
                        (i, validation_accuracy))
                    validation_writer.add_summary(summary, i)

                if i % 100 == 0:
                    # Test accuracy
                    batch, _ = test_batch_factory.next(75)
                    summary, test_accuracy = sess.run([merged, accuracy], feed_dict={self.x: batch["data"], self.y_: batch["model_output"], self.keep_prob: 1.0})
                    print('step %d, test accuracy %g' % (i, test_accuracy))
                    test_writer.add_summary(summary, i)
                    if test_accuracy > prev_test_accuracy:
                        prev_test_accuracy = test_accuracy
                        save_path = saver.save(sess, "model/model.cpkt")
                        print("New model saved to path: ", save_path)

    def load_graph(self):

        y_conv = self.build_graph()

        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        saver = tf.train.Saver()
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            #saver = tf.train.import_meta_graph('./model/model.cpkt.meta')
            saver.restore(sess, "model/model.cpkt")

            file_path = './atomistic_features_cubed_sphere/1A0E_protein_features.npz'

            with np.load(file_path) as data:
                for aa in data['aa_one_hot']:
                    inference_accuracy = y_conv.eval(feed_dict={self.x: aa, self.keep_prob: 1.0})
                    print(inference_accuracy)

# Set flags
flags = tf.app.flags

flags.DEFINE_string(
    "input_dir", "./atomistic_features_cubed_sphere/", "Input path")
flags.DEFINE_float("test_set_fraction", 0.25,
                   "Fraction of data set aside for testing")
flags.DEFINE_integer("validation_set_size", 10, "Size of validation set")
flags.DEFINE_string("logdir", "tmp/summary/", "Path to summary files")

FLAGS = flags.FLAGS


def main(_):
    ### GET DATA ###
    # get proteins feature file names and grid feature file names
    protein_feature_filenames = sorted(
        glob.glob(os.path.join(FLAGS.input_dir, "*protein_features.npz")))
    grid_feature_filenames = sorted(
        glob.glob(os.path.join(FLAGS.input_dir, "*residue_features.npz")))

    # Set range for validation and test set
    validation_end = test_start = int(
        len(protein_feature_filenames) * (1. - FLAGS.test_set_fraction))
    train_end = validation_start = int(
        validation_end - FLAGS.validation_set_size)

    # create object from BatchFactory class
    train_batch_factory = BatchFactory()

    # add the dataset X labels
    train_batch_factory.add_data_set("data",
                                     protein_feature_filenames[:train_end],
                                     grid_feature_filenames[:train_end])

    # add the dataset Y labels
    train_batch_factory.add_data_set("model_output",
                                     protein_feature_filenames[:train_end],
                                     key_filter=["aa_one_hot"])

    # create object from BatchFactory class
    validation_batch_factory = BatchFactory()

    # add the dataset X labels
    validation_batch_factory.add_data_set("data",
                                          protein_feature_filenames[
                                              validation_start:validation_end],
                                          grid_feature_filenames[validation_start:validation_end])

    # add the dataset Y labels
    validation_batch_factory.add_data_set("model_output",
                                          protein_feature_filenames[
                                              validation_start:validation_end],
                                          key_filter=["aa_one_hot"])

    # create object from BatchFactory class
    test_batch_factory = BatchFactory()

    # add the dataset X labels
    test_batch_factory.add_data_set("data",
                                    protein_feature_filenames[test_start:],
                                    grid_feature_filenames[test_start:])

    # add the dataset Y labels
    test_batch_factory.add_data_set("model_output",
                                    protein_feature_filenames[test_start:],
                                    key_filter=["aa_one_hot"])



    # Build the graph for the deep net
    deep_cnn = Deep_cnn()
    deep_cnn.build_graph()
    deep_cnn.train_graph(train_batch_factory, validation_batch_factory, test_batch_factory)



if __name__ == '__main__':
    print("starting")
    #tf.app.run()


protein_feature_filenames = sorted(
    glob.glob(os.path.join(FLAGS.input_dir, "*protein_features.npz")))
grid_feature_filenames = sorted(
    glob.glob(os.path.join(FLAGS.input_dir, "*residue_features.npz")))

# Set range for validation and test set
validation_end = test_start = int(
    len(protein_feature_filenames) * (1. - FLAGS.test_set_fraction))
train_end = validation_start = int(
    validation_end - FLAGS.validation_set_size)

# create object from BatchFactory class
train_batch_factory = BatchFactory()

# add the dataset X labels
train_batch_factory.add_data_set("data",
                                protein_feature_filenames[:train_end],
                                grid_feature_filenames[:train_end])

# add the dataset Y labels
train_batch_factory.add_data_set("model_output",
                                protein_feature_filenames[:train_end],
                                key_filter=["aa_one_hot"])

batch, _ = train_batch_factory.next(10)

print(batch["data"])

# Build the graph for the deep net
deep_cnn = Deep_cnn()
deep_cnn.build_graph()
deep_cnn.load_graph()

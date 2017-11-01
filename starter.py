
from batch_factory.deepfold_batch_factory import BatchFactory

import argparse
import sys
import tempfile

import glob
import os
import time
import datetime
import math

import numpy as np
import pandas as pd

from functools import reduce
import operator

from directional import conv_spherical_cubed_sphere, avg_pool_spherical_cubed_sphere
from batch_factory.deepfold_batch_factory import BatchFactory
import tensorflow as tf

FLAGS = None

class CNNModel(object):
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
    def __init__(self):
        # internal setting
        self.optimizer = tf.train.AdamOptimizer(0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
        self.shape = [None, 6,24,38,38,2]
        self.n_classes = 21

        # placeholders
        self.labels = tf.placeholder(tf.float32, shape=[None, self.n_classes])
        self.x = tf.placeholder(tf.float32, shape=[None, 6,24,38,38,2])
        self.keep_prob = tf.placeholder(tf.float32)

        # config
        self.batch_size = 10
        self.max_steps = 60000

    def build_graph(self):
        bias_initializer = tf.constant_initializer(0.0)

        def conv_layer(input, channels_in, channels_out, name="conv"):
            with tf.name_scope(name):

                W = tf.get_variable("W/{}".format(name), shape=[3,3,3, channels_in, channels_out], initializer=tf.truncated_normal_initializer(stddev=1.0, dtype=tf.float32))
                b = tf.get_variable("b/{}".format(name), shape=[channels_out], initializer=bias_initializer, dtype=tf.float32)

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
                W = tf.get_variable("W/{}".format(name), shape=[channels_in, channels_out], initializer=tf.truncated_normal_initializer(stddev=1.0, dtype=tf.float32))
                b = tf.get_variable("b/{}".format(name), shape=[channels_out], initializer=bias_initializer, dtype=tf.float32)
                return tf.nn.relu(tf.matmul(input, W) + b)

        def out(input, channels_in, channels_out, name="out"):
            with tf.name_scope(name):
                W = tf.get_variable("W/{}".format(name), shape=[channels_in, channels_out], initializer=tf.truncated_normal_initializer(stddev=1.0, dtype=tf.float32))
                b = tf.get_variable("b/{}".format(name), shape=[channels_out], initializer=bias_initializer, dtype=tf.float32)
                return tf.matmul(input, W) + b

        # Reshape to use within a convolutional neural net.
        x = tf.reshape(self.x, shape=[-1, 6, 24, 38, 38, 2])
        conv1 = conv_layer(x, 2, 16,  name="conv1")
        conv2 = conv_layer(conv1, 16, 32,  name="conv2")
        conv3 = conv_layer(conv2, 32, 64,  name="conv3")
        conv4 = conv_layer(conv3, 64, 128,  name="conv4")

        flattened = tf.reshape(conv4,  [-1, reduce_dim(conv4)])

        fc1 = fc_layer(flattened, reduce_dim(conv4), 2048, name="fc1")
        drop_out = tf.nn.dropout(fc1, self.keep_prob)
        fc2 = fc_layer(drop_out, 2048, 2048, name="fc2")
        return out(fc2, 2048, self.n_classes)

    def batch_factory(self):
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

        return {'train': train_batch_factory, 'validation': validation_batch_factory, 'test': test_batch_factory}

    def loss(self, logits, labels):
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            cross_entropy = tf.reduce_mean(cross_entropy)
            tf.add_to_collection('losses', cross_entropy)
            return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def accuracy(self, logits, labels):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.labels, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            return tf.reduce_mean(correct_prediction)

    def train(self, session):
        # Labels is a placeholder
        logits = self.build_graph()
        loss_op = self.loss(logits, self.labels)
        optimize_op = self.optimizer.minimize(loss_op)
        saver = tf.train.Saver(tf.all_variables())
        batch_factory = self.batch_factory()

        accuracy = self.accuracy(logits, self.labels)

        ### DEFINE SUMMARIES ###
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("cross_entropy", loss_op)
        merged = tf.summary.merge_all()

        time_str = datetime.datetime.now().isoformat()
        train_writer = tf.summary.FileWriter("./tmp/summary/train/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
        train_writer.add_graph(tf.get_default_graph())
        validation_writer = tf.summary.FileWriter("./tmp/summary/validation/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
        test_writer = tf.summary.FileWriter("./tmp/summary/test/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))

        ### TRAIN MODEL ###
        with session as sess:
            sess.run(tf.global_variables_initializer())
            if os.path.isfile("./model/model.ckpt"):
                saver.restore(sess, "./model/model.ckpt")

            # set variables
            best_test_accuracy = 0.0

            for step in range(self.max_steps):
                batch, _ = batch_factory['train'].next(self.batch_size)
                _, summary = sess.run([optimize_op, merged], feed_dict={self.x: batch["data"], self.labels: batch["model_output"], self.keep_prob: 0.5})
                if step % 10 == 0:
                    # Train accuracy
                    summary, train_accuracy = sess.run([merged, accuracy], feed_dict={self.x: batch["data"], self.labels: batch["model_output"], self.keep_prob: 1.0})
                    print('step %d, training accuracy %g' % (step, train_accuracy))
                    train_writer.add_summary(summary, step)

                    # Validation accuracy
                    batch, _ = batch_factory['validation'].next(10)
                    summary, validation_accuracy = sess.run([merged, accuracy],
                    feed_dict={self.x: batch["data"], self.labels: batch["model_output"], self.keep_prob: 1.0})
                    print('step %d, validation accuracy %g' %(step, validation_accuracy))
                    validation_writer.add_summary(summary, step)

                    # Save model if it has improved
                    checkpoint_path = os.path.join('model', 'model.ckpt')
                    saver.save(session, checkpoint_path, global_step=step)
                    print("New model saved to path: ", checkpoint_path)

    def predict(self, session, x):

        logits = self.build_graph()

        # Load model
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('model')
        saver.restore(session, ckpt.model_checkpoint_path)

        with session as sess:
            predicted_logits = sess.run([logits], feed_dict={self.x: x, self.keep_prob: 1.0})

        return predicted_logits


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
    # Train the graph
    with tf.Graph().as_default():
        model = CNNModel()
        session = tf.Session()
        model.train(session)

if __name__ == '__main__':
    print("starting")
    tf.app.run()

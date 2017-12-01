
from batch_factory.deepfold_batch_factory import BatchFactory

import argparse
import sys
import tempfile

import glob
import os
import time
import datetime
import math

import pickle
import numpy as np
import pandas as pd

from functools import reduce
import operator

from directional import conv_spherical_cubed_sphere, avg_pool_spherical_cubed_sphere
from batch_factory.deepfold_batch_factory import BatchFactory
import tensorflow as tf

from Bio.PDB import PDBParser, MMCIFParser, Polypeptide
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB import PDBList

FLAGS = None

# Set flags
flags = tf.app.flags

flags.DEFINE_string("input_dir", "./data/atomistic_features_cubed_sphere_train/", "Input path")
flags.DEFINE_float("test_set_fraction", 0.25,"Fraction of data set aside for testing")
flags.DEFINE_integer("validation_set_size", 10, "Size of validation set")
flags.DEFINE_string("logdir", "tmp/summary/", "Path to summary files")
flags.DEFINE_boolean("train", False, "Define if this is a training session")
flags.DEFINE_boolean("infer", False, "Define if this is a infering session")
flags.DEFINE_boolean("batch_size", 10, "batch size to train on")

FLAGS = flags.FLAGS

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
        self.bias_initializer = tf.constant_initializer(0.0)
        self.shape = [-1, 6, 24, 38, 38, 2]
        self.n_classes = 21

        # placeholders
        self.labels = tf.placeholder(tf.float32, shape=[None, self.n_classes])
        self.x = tf.placeholder(tf.float32, shape=[None, 6, 24, 38, 38, 2])
        self.keep_prob = tf.placeholder_with_default(tf.constant(1.0), shape=None)

        # config
        self.batch_size = 10
        self.max_steps = 700000

    def _weight_variable(self, name, shape, stddev=0.1):
        return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))

    def _reduce_dim(self, x):
            return reduce(operator.mul, x.shape.as_list()[1:], 1)

    def _fc_layer(self, input, channels_in, channels_out, name="fc"):
        with tf.variable_scope(name):
            W = self._weight_variable("weights", [channels_in, channels_out])
            b = tf.get_variable("b", shape=[channels_out], initializer=self.bias_initializer, dtype=tf.float32)
            return tf.nn.relu(tf.matmul(input, W) + b)

    def _out_layer(self, input, channels_in, channels_out, name="out"):
        with tf.variable_scope(name):
            W = self._weight_variable("weights", [channels_in, channels_out])
            b = tf.get_variable("b", shape=[channels_out], initializer=self.bias_initializer, dtype=tf.float32)
            return tf.matmul(input, W) + b

    def _conv_layer(self, input, channels_in, channels_out, name="conv"):
        with tf.variable_scope(name) as scope:
            W = self._weight_variable("weights", [3, 3, 3, channels_in, channels_out])
            b = tf.get_variable("b", shape=[channels_out], initializer=self.bias_initializer, dtype=tf.float32)
            convolution = conv_spherical_cubed_sphere(input, W, strides=[1, 1, 1, 1, 1], padding="SAME", name=name)
            activation = tf.nn.relu(convolution + b)
            return avg_pool_spherical_cubed_sphere(activation, ksize=[1, 1, 3, 3, 1], strides=[1,1,2,2,1], padding="VALID")

    def _build_graph(self):
        # Reshape to use within a convolutional neural net.
        x = tf.reshape(self.x, shape=[-1, 6, 24, 38, 38, 2])
        conv1 = self._conv_layer(x, 2, 16,  name="conv1")
        conv2 = self._conv_layer(conv1, 16, 32,  name="conv2")
        conv3 = self._conv_layer(conv2, 32, 64,  name="conv3")
        conv4 = self._conv_layer(conv3, 64, 128,  name="conv4")

        flattened = tf.reshape(conv4,  [-1, self._reduce_dim(conv4)])

        fc1 = self._fc_layer(flattened, self._reduce_dim(conv4), 2048, name="fc1")
        drop_out = tf.nn.dropout(fc1, self.keep_prob)
        fc2 = self._fc_layer(drop_out, 2048, 2048, name="fc2")
        return self._out_layer(fc2, 2048, self.n_classes)

    def _batch_factory(self):
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

    def _loss(self, logits, labels):
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            return tf.reduce_mean(cross_entropy)

    def _accuracy(self, logits, labels):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.labels, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            return tf.reduce_mean(correct_prediction)

    def train(self, session):
        # Labels is a placeholder
        logits = self._build_graph()
        loss_op = self._loss(logits, self.labels)
        optimize_op = self.optimizer.minimize(loss_op)
        saver = tf.train.Saver(tf.global_variables())
        batch_factory = self._batch_factory()

        accuracy = self._accuracy(logits, self.labels)

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

            # set variables
            best_validation_accuracy = 0.0
            for step in range(self.max_steps):
                batch, _ = batch_factory['train'].next(self.batch_size)
                _, summary = sess.run([optimize_op, merged], feed_dict={self.x: batch["data"], self.labels: batch["model_output"], self.keep_prob: 0.5})
                if step % 100 == 0:
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

                    if validation_accuracy > best_validation_accuracy:
                        best_validation_accuracy = validation_accuracy
                        # Save model if it has improved
                        checkpoint_path = os.path.join('model', 'model.ckpt')
                        saver.save(session, checkpoint_path)
                        print("New model saved to path: ", checkpoint_path)

    def _probabilities(self, logits):
        with tf.name_scope('probabilities'):
            return tf.nn.softmax(logits=logits)

    def predict(self, session, data):

        logits = self._build_graph()

        # Load model
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('model/')
        saver.restore(session, ckpt.model_checkpoint_path)

        logit_array = []

        ''' with session as sess:
            print(batch_factory.data_size())
            for i in range(batch_factory.data_size()):
                batch, _ = batch_factory.next(1)
                predicted_logits = sess.run([self._probabilities(logits)], feed_dict={self.x: batch["data"]})
                print(predicted_logits)
                logit_array.append(predicted_logits)
        # TODO: Here I need to have a tensor of size: (protein_size, patches, radius, xi, eta, channels) - can this work?
        print(predicted_logits) '''
        with session as sess:
            predicted_logits = sess.run([self._probabilities(logits)], feed_dict={self.x: data})


        return predicted_logits

def main(_):
    print("hej")
    tf.reset_default_graph()
    if FLAGS.train:
        with tf.Graph().as_default():
            model = CNNModel()
            session = tf.Session()
            model.batch_size = FLAGS.batch_size
            model.train(session)
            print(graph.get_shape)

    if FLAGS.infer:
        with tf.Graph().as_default():
            model = CNNModel()
            session = tf.Session()
            model.batch_size = 1

            ''' # get filenames
            protein_feature_filenames = sorted(glob.glob(os.path.join(FLAGS.input_dir, "*protein_features.npz")))
            print(protein_feature_filenames)
            for name in protein_feature_filenames:
                protein_name = os.path.basename(name).split("_")[0]
                print(protein_name)
                p = get_protein(FLAGS.input_dir, protein_name, max_batch_size=1)
                print(p)

                #predicted_logits = model.predict(session, get_protein(FLAGS.input_dir, protein_name, max_batch_size=1))
                #np.savez("protein_logits/{}".format(protein_name), predicted_logits) '''

            ''' batchFactory = get_protein(FLAGS.input_dir, "107L", max_batch_size=1)
            print(batchFactory.data_size())
            batch, _ = batchFactory.next(batchFactory.data_size())
            print(batch["data"][44-1])


            # Fetch structure from .cif file
            pdbl = PDBList()
            pdbl.retrieve_pdb_file("107L", pdir="./data/PDB/")
            structure = MMCIFParser().get_structure("107L", "{}{}.cif".format("./data/PDB/", "107L"))
            # Build structures
            ppb=PPBuilder()
            for pp in ppb.build_peptides(structure):
                seq = pp.get_sequence()
                print(seq[44-1])
                print(len(seq)) '''

if __name__ == '__main__':
    tf.app.run()

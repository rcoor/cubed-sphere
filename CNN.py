
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

flags.DEFINE_string(
    "input_dir", "./data/atomistic_features_cubed_sphere_train/", "Input path")
flags.DEFINE_float("test_set_fraction", 0.25,
                   "Fraction of data set aside for testing")
flags.DEFINE_integer("validation_set_size", 10, "Size of validation set")
flags.DEFINE_string("logdir", "tmp/summary/", "Path to summary files")
flags.DEFINE_boolean("train", False, "Define if this is a training session")
flags.DEFINE_boolean("infer", False, "Define if this is a infering session")
flags.DEFINE_integer("batch_size", 10, "batch size to train on")
flags.DEFINE_string("model", None, "Path to specific model, otherwise None.")

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

    def __init__(self,  checkpoint_path='model/', step=None, reg_fact=0.001, *args, **kwargs):
        # tf.reset_default_graph()
        # internal setting
        self.bias_initializer = tf.constant_initializer(0.0)
        self.shape = [-1, 6, 24, 38, 38, 2]
        self.n_classes = 21

        # config
        self.batch_size = 10
        self.max_epochs = 700000

        # build model
        self.logits = self._build_graph(*args, **kwargs)

        self.entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        self.optimizer = tf.train.AdamOptimizer(0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
        self.regularization = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if not v.name.startswith("b")]) * reg_fact
        self.loss = tf.reduce_mean(self.entropy) + self.regularization
        self.train_step = self.optimizer.minimize(self.loss)

        self.session = tf.Session()
        # self.saver = tf.train.Saver()
        ### Initialize variables ###
        tf.global_variables_initializer().run(session=self.session)
        print("Variables initialized")

    def _weight_variable(self, name, shape, stddev=0.1):
        return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))

    def _reduce_dim(self, x):
        return reduce(operator.mul, x.shape.as_list()[1:], 1)

    def _fc_layer(self, input, channels_in, channels_out, name="fc"):
        with tf.variable_scope(name):
            W = self._weight_variable("weights", [channels_in, channels_out])
            b = tf.get_variable(
                "b", shape=[channels_out], initializer=self.bias_initializer, dtype=tf.float32)
            return tf.nn.relu(tf.matmul(input, W) + b)

    def _out_layer(self, input, channels_in, channels_out, name="out"):
        with tf.variable_scope(name):
            W = self._weight_variable("weights", [channels_in, channels_out])
            b = tf.get_variable(
                "b", shape=[channels_out], initializer=self.bias_initializer, dtype=tf.float32)
            return tf.matmul(input, W) + b

    def _conv_layer(self, input,
                    channels_in,
                    channels_out,
                    ksize_r=3,
                    ksize_xi=3,
                    ksize_eta=3,
                    stride_r=2,
                    stride_xi=2,
                    stride_eta=2,
                    name="conv"):
        with tf.variable_scope(name) as scope:
            W = self._weight_variable("weights", [ksize_r, ksize_xi, ksize_eta, channels_in, channels_out])
            b = tf.get_variable(
                "b", shape=[channels_out], initializer=self.bias_initializer, dtype=tf.float32)
            convolution = conv_spherical_cubed_sphere(input, W, strides=[
                1, stride_r, stride_xi, stride_eta, 1],
                padding="VALID",
                name=name)
            return tf.nn.relu(convolution + b)

    def _pooling_layer(self,
                       input,
                       ksize_r=3,
                       ksize_xi=3,
                       ksize_eta=3,
                       stride_r=2,
                       stride_xi=2,
                       stride_eta=2,
                       name="pooling"):
        with tf.variable_scope(name) as scope:
            return avg_pool_spherical_cubed_sphere(input, ksize=[1, ksize_r, ksize_xi, ksize_eta, 1], strides=[1, stride_r, stride_xi, stride_eta, 1], padding="VALID")

    def _build_graph(self):
        # Reshape to use within a convolutional neural net.

        # placeholders
        self.labels = tf.placeholder(tf.float32, shape=[None, self.n_classes])
        self.x = tf.placeholder(tf.float32, shape=[None, 6, 24, 38, 38, 2])
        self.keep_prob = tf.placeholder_with_default(tf.constant(1.0), shape=None)

        x = tf.reshape(self.x, shape=[-1, 6, 24, 38, 38, 2])
        ### LAYER 1 ###
        conv1 = self._conv_layer(x,
                                 ksize_r=3,
                                 ksize_xi=5,
                                 ksize_eta=5,
                                 channels_out=16,
                                 channels_in=2,
                                 stride_r=1,
                                 stride_xi=2,
                                 stride_eta=2,
                                 name="conv1")

        pool1 = self._pooling_layer(conv1,
                                    ksize_r=1,
                                    ksize_xi=3,
                                    ksize_eta=3,
                                    stride_r=1,
                                    stride_xi=2,
                                    stride_eta=2)

        ### LAYER 2 ####
        conv2 = self._conv_layer(pool1, ksize_r=3,
                                 ksize_xi=3,
                                 ksize_eta=3,
                                 channels_in=16,
                                 channels_out=32,
                                 stride_r=1,
                                 stride_xi=1,
                                 stride_eta=1,  name="conv2")

        pool2 = self._pooling_layer(conv2,
                                    ksize_r=3,
                                    ksize_xi=3,
                                    ksize_eta=3,
                                    stride_r=2,
                                    stride_xi=2,
                                    stride_eta=2, name="pool2")

        ### LAYER 3 ####
        conv3 = self._conv_layer(pool2,
                                 ksize_r=3,
                                 ksize_xi=3,
                                 ksize_eta=3,
                                 channels_in=32,
                                 channels_out=64,
                                 stride_r=1,
                                 stride_xi=1,
                                 stride_eta=1,  name="conv3")

        pool3 = self._pooling_layer(conv3,
                                    ksize_r=1,
                                    ksize_xi=3,
                                    ksize_eta=3,
                                    stride_r=1,
                                    stride_xi=2,
                                    stride_eta=2, name="pool3")

        ### LAYER 4 ####
        conv4 = self._conv_layer(pool3,
                                 ksize_r=3,
                                 ksize_xi=3,
                                 ksize_eta=3,
                                 channels_in=64,
                                 channels_out=128,
                                 stride_r=1,
                                 stride_xi=1,
                                 stride_eta=1,  name="conv4")

        pool4 = self._pooling_layer(conv4,
                                    ksize_r=1,
                                    ksize_xi=3,
                                    ksize_eta=3,
                                    stride_r=1,
                                    stride_xi=1,
                                    stride_eta=1, name="pool4")

        print(pool4.get_shape)

        flattened = tf.reshape(pool4,  [-1, self._reduce_dim(pool4)])

        fc1 = self._fc_layer(
            flattened, self._reduce_dim(pool4), 2048, name="fc1")
        drop_out1 = tf.nn.dropout(fc1, self.keep_prob)

        fc2 = self._fc_layer(drop_out1, 2048, 2048, name="fc2")
        drop_out2 = tf.nn.dropout(fc2, self.keep_prob)

        out = self._out_layer(drop_out2, 2048, self.n_classes)
        return out

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


    def train(self):
        # Labels is a placeholder
        ''' loss_op = self._loss(logits, self.labels)
        optimize_op = self.optimizer.minimize(loss_op) '''

        batch_factory = self._batch_factory()


        ## DEFINE SUMMARIES ###
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("cross_entropy", self.entropy)
        merged = tf.summary.merge_all()

        time_str = datetime.datetime.now().isoformat()
        train_writer = tf.summary.FileWriter(
            "./tmp/summary/train/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
        train_writer.add_graph(tf.get_default_graph())
        validation_writer = tf.summary.FileWriter(
            "./tmp/summary/validation/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
        test_writer = tf.summary.FileWriter(
            "./tmp/summary/test/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))

        ### TRAIN MODEL ###
        with self.session.as_default():
            #self.logits = self._build_graph()
            iteration = 0

            # set variables
            best_validation_accuracy = 0.0
            for epoch in range(self.max_epochs):

                more_data = True
                while more_data:

                    more_data = (batch_factory['train'].feature_index != 0)

                    batch, _ = batch_factory['train'].next(self.batch_size)

                    feed_dict = dict({self.x: batch["data"], self.labels: batch["model_output"], self.keep_prob: 0.5})

                    _, loss_value, summary = self.session.run([self.train_step, self.loss, merged], feed_dict={self.x: batch["data"], self.labels: batch["model_output"], self.keep_prob: 0.5})

                    if iteration % 100 == 0:

                        Q_training_batch, loss = self.Q_accuracy_and_loss(batch)
                        print('step %d, of epoch %d, training loss %g, accuracy %g' %(iteration, epoch, loss, Q_training_batch))
                        train_writer.add_summary(summary, iteration)

                        # Validation accuracy
                        validation_batch, _ = batch_factory['validation'].next(batch_factory['validation'].data_size())

                        Q_validation_batch, loss = self.Q_accuracy_and_loss(validation_batch)
                        print('step %d, of epoch %d, training loss %g, accuracy %g' %(iteration, epoch, loss, Q_validation_batch))
                        validation_writer.add_summary(summary, iteration)

                        if Q_validation_batch > best_validation_accuracy:
                            best_validation_accuracy = Q_validation_batch
                            # Save model if it has improved
                            checkpoint_path = os.path.join('model', 'model.ckpt')
                            saver.save(self.session, checkpoint_path, global_step=step)
                            print("New model saved to path: ", checkpoint_path)

                        iteration += 1

    def _accuracy(self, logits, labels):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(
                tf.argmax(logits, 1), tf.argmax(self.labels, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            return tf.reduce_mean(correct_prediction)

    def Q_accuracy_and_loss(self, batch):
        y = batch["model_output"]
        y_argmax = np.argmax(y, 1)
        results = self.infer(y)

        y_, entropies = map(np.concatenate, zip(*results))

        predictions = np.argmax(y_, 1)
        identical = (predictions == y_argmax)

        Q_accuracy = np.mean(identical)

        regularization = self.session.run(self.regularization, feed_dict={})
        loss = np.mean(entropies) + regularization

        return Q_accuracy, loss

    def infer(self, data):
        return self.session.run([tf.nn.softmax(self.logits), self.entropy], feed_dict={self.x: data, self.keep_prob: 1.0})

    def restore(self, checkpoint_path, step=None):
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)

        if ckpt and ckpt.model_checkpoint_path:
            if step is None or step == -1:
                print("Restoring from: last checkpoint")
                self.saver.restore(self.session, tf.train.latest_checkpoint(checkpoint_path))
            else:
                checkpoint_file = checkpoint_path+("/model.ckpt-%d" % step)
                print("Restoring from:", checkpoint_file)
                self.saver.restore(self.session, checkpoint_file)
        else:
            print("Could not load file")



def main(_):
    print("hej")
    if FLAGS.train:
        model = CNNModel()
        model.batch_size = FLAGS.batch_size
        model.train()
        # print(graph.get_shape)

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

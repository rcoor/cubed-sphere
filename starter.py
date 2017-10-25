
from batch_factory.deepfold_batch_factory import BatchFactory

import argparse
import sys
import tempfile

import glob
import os
import time
import datetime

from functools import reduce
import operator

from directional import conv_spherical_cubed_sphere, avg_pool_spherical_cubed_sphere
from batch_factory.deepfold_batch_factory import BatchFactory
import tensorflow as tf

FLAGS = None


def deepnn(n_classes=21):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.
    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    ''' with tf.name_scope('reshape'):
        x_image = tf.reshape(x, shape=[-1, 6, 24, 38, 38, 2]) '''
    x_image = tf.placeholder(tf.float32, shape=[40, 6, 24, 38, 38, 2])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        filter_shape = [3, 5, 5, 2, 16]
        W_conv1 = weight_variable(filter_shape)
        b_conv1 = bias_variable(filter_shape[-1:])
        h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1,  [1,1,3,3,1], [1,1,2,2,1])
    print(h_pool1)
    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([3, 3, 3, 16, 32])
        b_conv2 = bias_variable([32])
        h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2, [1,3,3,3,1],  [1,2,2,2,1])

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([3, 3, 3, 32, 64])
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv3d(h_pool2, W_conv3) + b_conv3)

    # Second pooling layer.
    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv3, [1,1,3,3,1],  [1,1,2,2,1])

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv4'):
        W_conv4 = weight_variable([3, 3, 3, 64, 128])
        b_conv4 = bias_variable([128])
        h_conv4 = tf.nn.relu(conv3d(h_pool3, W_conv4) + b_conv4)

    # Second pooling layer.
    with tf.name_scope('pool4'):
        h_pool4 = max_pool_2x2(h_conv4,  [1,1,3,3,1],  [1,1,2,2,1])

        print(h_pool4.get_shape)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        # get shape of the last pool and reduce by product
        h_pool4_shape = h_pool4.shape.as_list()[1:]
        print(h_pool4_shape)
        dim_prod = reduce(operator.mul, h_pool4_shape, 1)

        W_fc1 = weight_variable([dim_prod, 2048])
        b_fc1 = bias_variable([2048])

        h_pool4_flat = tf.reshape(h_pool4, [-1, dim_prod])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)


    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([2048, 2048])
        b_fc2 = bias_variable([2048])

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc3'):
        W_fc3 = weight_variable([2048, n_classes])
        b_fc3 = bias_variable([n_classes])

        h_fc3 = tf.matmul(h_fc2, W_fc3) + b_fc3
        print(h_fc3.get_shape)

    return h_fc3


def conv3d(x, W):
    """conv3d returns a 3d convolution layer with full stride."""
    return conv_spherical_cubed_sphere(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x, filter_size=[1,1,3,3,1], strides=[1,1,2,2,1]):
    """max_pool_2x2 ford 3d convolutions downsamples a feature map by 2X."""
    ''' pools = []
    for patch in range(x.get_shape().as_list()[1]):
        print(x[:, patch, :, :, :, :].get_shape)
        pools.append(
            tf.nn.avg_pool3d(x[:, patch, :, :, :, :], ksize=filter_size, strides=filter_size, padding='SAME')
        )
        tf.stack(pools, axis=1)
        '''
    pool = avg_pool_spherical_cubed_sphere(x,
                                         ksize=filter_size,
                                         strides=strides,
                                         padding='VALID')
    return pool


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1, name="W")
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape, name="b")
    return tf.Variable(initial)

# Set flags
flags = tf.app.flags

flags.DEFINE_string(
    "input_dir", "./atomistic_features_cubed_sphere/", "Input path")
flags.DEFINE_float("test_set_fraction", 0.25,
                   "Fraction of data set aside for testing")
flags.DEFINE_integer("validation_set_size", 10, "Size of validation set")
flags.DEFINE_string("logdir", "tmp/summary/", "Path to summary files")

FLAGS = flags.FLAGS

deepnn()

''' def main(_):

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

    ### CREATE MODEL ###
    # Create the model
    x = tf.placeholder(tf.float32, shape=[None, 24, 76, 151, 2])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, shape=[None, 21])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('learning_rate'):
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.1
        learning_rate = tf.train.exponential_decay(
            starter_learning_rate, global_step, 100, 0.96, staircase=True)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(
            0.001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cross_entropy, global_step=global_step)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    ### DEFINE SUMMARIES ###
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.scalar("cross_entropy", cross_entropy)
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
                                  x: batch["data"], y_: batch["model_output"], keep_prob: 0.5})
            if i % 10 == 0:
                # Train accuracy
                summary, train_accuracy = sess.run([merged, accuracy], feed_dict={
                                                   x: batch["data"], y_: batch["model_output"], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
                train_writer.add_summary(summary, i)

                # Validation accuracy
                batch, _ = validation_batch_factory.next(10)
                summary, validation_accuracy = sess.run([merged, accuracy], feed_dict={
                                                        x: batch["data"], y_: batch["model_output"], keep_prob: 1.0})
                print('step %d, validation accuracy %g' %
                      (i, validation_accuracy))
                validation_writer.add_summary(summary, i)

            if i % 100 == 0:
                # Test accuracy
                batch, _ = test_batch_factory.next(75)
                summary, test_accuracy = sess.run([merged, accuracy], feed_dict={
                                                  x: batch["data"], y_: batch["model_output"], keep_prob: 1.0})
                print('step %d, test accuracy %g' % (i, test_accuracy))
                test_writer.add_summary(summary, i)
                if test_accuracy > prev_test_accuracy:
                    prev_test_accuracy = test_accuracy
                    save_path = saver.save(sess, "model/model.cpkt")
                    print("New model saved to path: ", save_path)

if __name__ == '__main__':
    tf.app.run() '''

import numpy as np
import sys
from model import Model
import tensorflow as tf

from optparse import OptionParser

parser = OptionParser()
parser.add_option("--data", dest="data_filename",
                  help="data file", metavar="FILE")
parser.add_option("--reshape", dest="reshape", nargs=3,
                  help="Optionally rehape dataset", type=int)
parser.add_option("--training-range", dest="training_range", nargs=2, default=(0,5600),
                  help="Which rows to use for training", type=int)
parser.add_option("--testing-range", dest="testing_range", nargs=2, default=(5605,5877),
                  help="Which rows to use for training", type=int)
parser.add_option("--model-checkpoint-path", dest="model_checkpoint_path", default="models",
                  help="Directory where models are stored")
parser.add_option("--read-from-checkpoint", action="store_true", dest="read_from_checkpoint",
                  help="Whether to read model from checkpoint")
parser.add_option("--mode", choices=['train', 'test'], dest="mode", default="train", type='choice',
                  help="Mode of operation: train or test")
parser.add_option("--crf-output-layer", action="store_true", dest="crf_output_layer", default=False,
                  help="Whether output layers should be a conditional random field")
parser.add_option("--layers", dest="layers", default=3, type=int,
                  help="Number of layers in the network")
parser.add_option("--batch-size", dest="batch_size", default=10, type=int,
                  help="Batch size used during training")
parser.add_option("--filter-depth", dest="filter_depth", default=10, type=int,
                  help="Filter size (channels)")
parser.add_option("--dump-frequency", dest="dump_frequency", default=10, type=int,
                  help="How often to write to screen and save models")
parser.add_option("--optimize-using-lbfgs", dest="optimize_using_lbfgs", action="store_true", default=False,
                  help="Use the LBFGS optimizer")
parser.add_option("--lbfgs-maxiter", dest="lbfgs_maxiter", type=int, default=50,
                  help="How many lbfgs iterations to run in each step")
parser.add_option("--regularization-factor", dest="regularization_factor", type=float, default=0.001,
                  help="L2 regularization factor")

(options, args) = parser.parse_args()

data = np.load(options.data_filename)
if options.reshape:
    data = data.reshape(options.reshape)
print("data shape: ", data.shape)

ss_offset = 22
ss_labels8 = ['C', 'B', 'E', 'G', 'I', 'H', 'S', 'T','NoSeq']
ss_labels3 = {'H': ['G','H','I'], 'E':['E','B'], 'C':['S','T','C'], 'NoSeq':['NoSeq']}

ss_column_indices = []
for label3 in ['H', 'E', 'C', 'NoSeq']:
    ss_column_indices.append([ss_offset+ss_labels8.index(ss) for ss in ss_labels3[label3]])

ss_columns = np.zeros([data.shape[0], data.shape[1], len(ss_labels3)])
for i, column_indices in enumerate(ss_column_indices):
    ss_columns[:,:,i] = np.sum(data[:,:, column_indices], axis=2)

# data = np.concatenate((data[:,:,:ss_offset], ss_columns), axis=2)
# data = np.concatenate((data[:,:,:ss_offset], ss_columns, data[:,:,35:]), axis=2)
data = np.concatenate((data[:,:,:ss_offset], data[:,:,35:]), axis=2), ss_columns
# data = data[:,:,:ss_offset], ss_columns
print("SS-reduced data set shape: ", data[0].shape, data[1].shape)

training_data = data[0][slice(*options.training_range)], data[1][slice(*options.training_range)] 
print("training data shape: ", training_data[0].shape, training_data[1].shape)

test_data = data[0][slice(*options.testing_range)], data[1][slice(*options.testing_range)] 
print("test data shape: ", test_data[0].shape, test_data[1].shape)

model = Model(input_size=data[0].shape[2],
              output_size=4,
              max_length=training_data[0].shape[1],
              crf_output_layer=options.crf_output_layer,
              layers=options.layers,
              filter_depth=options.filter_depth,
              regularization_factor = options.regularization_factor,
              optimize_using_lbfgs = options.optimize_using_lbfgs,
              lbfgs_maxiter = options.lbfgs_maxiter)

if options.read_from_checkpoint:

    model.restore(options.model_checkpoint_path)


if options.mode == 'train':
    X,y = training_data

    training_sequence_lengths = np.argmax((training_data[0][:,:,-1] == 1), axis=1)
    # training_sequence_lengths = np.full(training_data[0].shape[0], training_data[0].shape[1])

    model.train(X, y,
                model_checkpoint_path=options.model_checkpoint_path,
                sequence_lengths=training_sequence_lengths,
                num_passes=100000,
                dump_frequency=options.dump_frequency,
                batch_size=options.batch_size,
                test_X = test_data[0],
                test_y = test_data[1])

elif options.mode == 'test':
    X, y = test_data

    predictions = model.infer(X)

    if not options.crf_output_layer:
        predictions = tf.argmax(predictions, 2).eval()

    y_argmax = tf.argmax(y, 2).eval()
    
    # test_sequence_lengths = np.argmax((test_data[0][:,:,-1] == 1), axis=1)

    # matches = 0
    # total = 0
    # for i in range(len(predictions)):
    #     prediction = np.array(predictions[i][:test_sequence_lengths[i]])
    #     labels = y_argmax[i][:test_sequence_lengths[i]]
    #     print prediction
    #     print labels
    #     matches += np.count_nonzero(prediction==labels)
    #     total += len(prediction)
    # print "Q3 score: ", matches / float(total)

    # X_sel = (X[:,:,-1] != 1)
    # predictions_sel = predictions[X_sel]
    # y_sel = y_argmax[X_sel]
    # print "Q3 score: ", np.count_nonzero(y_sel == predictions_sel) / float(predictions_sel.size)
    print("Q3 score: ", model.Q3_accuracy(X,y))
    
    # else:

    #     print "Q3 score: ", np.count_nonzero(y_sel == predictions) / float(predictions_argmax_sel.size)
        
    # # predictions_argmax = tf.argmax(predictions, 2).eval()

    # # print X.shape
    # # print "???", (X[:,:,-1] != 1).shape
    # # print np.count_nonzero(X[X[:,:,-1] == 1]), X.size, (X[X[:,:,-1] == 1]).shape

    # # print predictions_argmax.shape
    # # for protein in predictions_argmax:
    # #     print protein
    # X_sel = (X[:,:,-1] != 1)
    # y_sel = y_argmax[X_sel]
    # # predictions_argmax_sel = predictions_argmax[X_sel]

    
    # matches = 0
    # total = 0
    # for i in range(len(predictions)):
    #     prediction = predictions[i][:test_sequence_lengths[i]]
    #     labels = y_argmax[i][:test_sequence_lengths[i]]
    #     matches += np.count_nonzero(prediction==labels)
    #     total += len(prediction)
    # # for i in range(X_sel.shape[0]):
    # #     y_argmax_sel = y_argmax[i][X_sel[i]]
    # #     predictions_argmax_sel = predictions_argmax[i][X_sel[i]]
    # #     print "y_sel: ", y_argmax_sel
    # #     print "predictions_argmax_sel: ", predictions_argmax_sel, predictions_argmax_sel.shape
    # #     matches += np.count_nonzero(y_argmax_sel == predictions_argmax_sel)
    # #     total += len(y_argmax_sel)
        
    # print "Q3 score: ", matches / float(total)
    
    # print y.shape
    # print X_sel.shape
    # print y_sel.shape
    # print y[X_sel].shape
    # print predictions_argmax_sel.shape
    
    # for i in range(X_sel.shape[0]):
    #     print y_sel[i,:]
    #     print predictions_argmax_sel[i,:]
    #     print
    
    # print "???", tf.argmax(predictions, 2).get_shape()
    # print predictions.shape
    # print tf.nn.in_top_k(predictions, y, 1)
    # print "Q3 score: ", np.count_nonzero(y_sel == predictions) / float(predictions_argmax_sel.size)
else:
    print("Unknown mode: ", options.mode)


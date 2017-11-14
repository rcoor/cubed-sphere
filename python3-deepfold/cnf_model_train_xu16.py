import numpy as np
import sys
from model import Model

from optparse import OptionParser

parser = OptionParser()
parser.add_option("--data", dest="data_filename",
                  help="data file", metavar="FILE")
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
parser.add_option("--dump-frequency", dest="dump_frequency", default=10, type=int,
                  help="How often to write to screen and save models")
parser.add_option("--optimize-using-lbfgs", dest="optimize_using_lbfgs", action="store_true", default=False,
                  help="Use the LBFGS optimizer")
parser.add_option("--lbfgs-maxiter", dest="lbfgs_maxiter", type=int, default=50,
                  help="How many lbfgs iterations to run in each step")
parser.add_option("--regularization-factor", dest="regularization_factor", type=float, default=0.01,
                  help="L2 regularization factor")

(options, args) = parser.parse_args()

data = np.load(options.data_filename)['data']

feature_array = data[:,:,:-4]
label_array = data[:,:,-4:]

# Strip off EmissionProb and PSIPRED data
# feature_array = feature_array[:,:,21:63]
feature_array = np.concatenate([feature_array[:,:,:21], feature_array[:,:,42:63]], axis=2)

print(feature_array[-1, -1])


training_data = feature_array[slice(*options.training_range)], label_array[slice(*options.training_range)]
print("training data shape: ", training_data[0].shape, training_data[1].shape)

test_data = feature_array[slice(*options.testing_range)], label_array[slice(*options.testing_range)]
print("test data shape: ", test_data[0].shape, test_data[1].shape)

model = Model(input_size=training_data[0].shape[2],
              output_size=4,
              max_length=training_data[0].shape[1],
              crf_output_layer=options.crf_output_layer,
              layers=options.layers,
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
    

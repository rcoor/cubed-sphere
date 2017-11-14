import numpy as np
import sys

from optparse import OptionParser

parser = OptionParser()
parser.add_option("--input", dest="input_data_filename",
                  help="data file", metavar="FILE")
parser.add_option("--output", dest="output_data_filename", default="xu16.npz",
                  help="data file", metavar="FILE")

(options, args) = parser.parse_args()

max_seq_length = 700
current_seq_length = None
features = []
labels = []
for line in open(options.input_data_filename):

    line = line.strip()

    if line == "":
        continue

    if current_seq_length is None:
        current_seq_length = int(line)
        features.append([])
        labels.append([])

    elif len(features[-1]) < current_seq_length:
        f_list = line.split()
        f_list.append('0.')
        f_list.insert(60, '0.')
        f_list.insert(40, '0.')
        f_list.insert(20, '0.')
        features[-1].append(f_list)
    elif len(labels[-1]) < current_seq_length:
        labels[-1].append(int(line))

        if len(labels[-1]) == current_seq_length:
            current_seq_length = None
    else:
        assert(false)        

dummy_indices = [20, 41, 62, 66]
padding_feature = np.zeros(len(features[0][0]))
padding_feature[dummy_indices] = 1
        
feature_array = np.zeros([len(features), max_seq_length, len(features[0][0])])
for i in range(len(features)):
    feature_array[i,:len(features[i]),:] = features[i][:max_seq_length]
    feature_array[i,len(features[i]):,:] = padding_feature


ss3_labels = ['H', 'E', 'C', 'NA'] 
ss8_labels = ['H', 'G', 'I', 'E', 'B', 'T', 'S', 'L', 'NA']
ss3_labels_lookup = {'H': ['G','H','I'], 'E':['E','B'], 'C':['S','T','L', 'NA']}
ss3_labels_lookup_inv = {'H':'H', 'G':'H', 'I':'H', 'E':'E', 'B':'E', 'T':'C', 'S':'C', 'L':'C'}
ss_output_dim = len(ss3_labels)

dummy_indices = [3]
padding_label = np.zeros(ss_output_dim)
padding_label[dummy_indices] = 1

label_array = np.zeros([len(labels), max_seq_length, ss_output_dim])
for i in range(len(labels)):
    label_ss8_indices = labels[i]
    label_ss3_indices = [ss3_labels.index(ss3_labels_lookup_inv[ss8_labels[label_index]]) for label_index in label_ss8_indices]
    # print label_ss3_indices
    # print label_array[i]
    label_array[i,:len(labels[i])] = np.eye(ss_output_dim)[label_ss3_indices]
    label_array[i,len(labels[i]):] = padding_label
    # label_array[i,:,label_ss3_indices] = 1

data = np.concatenate([feature_array, label_array], axis=2)

np.savez_compressed(options.output_data_filename, data=data)

print(data.shape)

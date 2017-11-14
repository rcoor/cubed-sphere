import os
import numpy as np
import re

from utils import re_float

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("stdout_file", metavar="STDOUT", nargs=1, type=str, help="the stdout file")
parser.add_argument("-q", "--qvalue", help="print index for max Q value", action="store_true")
parser.add_argument("-l", "--loss", help="print index for min loss", action="store_true")
parser.add_argument("--max-step", help="The maximal step to include", type=int, default=None)

parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

with open(args.stdout_file[0], 'r') as f:
    read_data = f.read()

Q_validation = np.array(re.findall("(?:\[\d+,\s*)?(\d+)(?:\])?\s*Q\d+ score \(validation set\)\s*[:=]\s*\(?(%s)" % re_float, read_data), dtype=float)
loss_validation = np.array(re.findall("(?:\[\d+,\s*)?(\d+)(?:\])?\s*loss \(validation set\)\s*[:=]\s*\(?(%s)" % re_float, read_data), dtype=float)


if args.max_step is not None:
    Q_validation = Q_validation[Q_validation[:,0]<args.max_step, :]
    loss_validation = loss_validation[loss_validation[:,0]<args.max_step, :]



Q_argmax = np.argmax(Q_validation[:,1])
Q_max = Q_validation[Q_argmax]

if loss_validation.shape[0] > 0:
    loss_argmin = np.argmin(loss_validation[:,1])
    loss_min = loss_validation[loss_argmin]

if args.verbose:
    print("step: %d" % Q_max[0])
    print("Q value: %f" % Q_max[1])
    
    if loss_validation.shape[0] > 0:
        print("loss: %f" % loss_validation[Q_argmax][1])
        print()
        print("step: %d" % loss_min[0])
        print("loss: %f" % loss_min[1])
        print("Q value: %f" % Q_validation[loss_argmin][1])

else:
    if args.qvalue:
        print(int(Q_max[0]))
    if args.loss:
        print(int(loss_min[0]))


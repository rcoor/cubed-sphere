import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re

from utils import re_float

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("stdout_file", metavar="STDOUT", nargs=1, type=str, help="the stdout file")
parser.add_argument("plot_file_loss", metavar="PLOT", nargs='?', type=str, default=None, help="the output plot file name for the loss plot (default: %(default)s)")
parser.add_argument("plot_file_Q", metavar="PLOT", nargs='?', type=str, default=None, help="the output plot file name for the Q scores (default: %(default)s)")
parser.add_argument("plot_file_losses", metavar="PLOT", nargs='?', type=str, default=None, help="the output plot file name for the two losses (default: %(default)s)")
parser.add_argument("--qymin", metavar="VAL", default=None, type=float, help="Minimal y-value for Q plot (default: %(default)s)")
parser.add_argument("--qymax", metavar="VAL", default=None, type=float, help="Maximal y-value for Q plot (default: %(default)s)")

args = parser.parse_args()

plot_file_loss = args.plot_file_loss if args.plot_file_loss is not None else (os.path.splitext(args.stdout_file[0])[0]+".loss.png")
plot_file_Q = args.plot_file_Q if args.plot_file_Q is not None else (os.path.splitext(args.stdout_file[0])[0]+".Q.png")
plot_file_losses = args.plot_file_losses if args.plot_file_losses is not None else (os.path.splitext(args.stdout_file[0])[0]+".losses.png")

with open(args.stdout_file[0], 'r') as f:
    read_data = f.read()

loss = np.array(re.findall("[Ll]oss\s*[:=]\s*(%s)" % (re_float), read_data), dtype=float)
Q_training = np.array(re.findall("(?:\[\d+,\s*)?(\d+)(?:\])?\s*Q\d+ score \(training batch\)\s*[:=]\s*\(?(%s)" % re_float, read_data), dtype=float)
Q_validation = np.array(re.findall("(?:\[\d+,\s*)?(\d+)(?:\])?\s*Q\d+ score \(validation set\)\s*[:=]\s*\(?(%s)" % re_float, read_data), dtype=float)

loss_training = np.array(re.findall("(?:\[\d+,\s*)?(\d+)(?:\])?\s*loss \(training batch\)\s*[:=]\s*\(?(%s)" % re_float, read_data), dtype=float)
loss_validation = np.array(re.findall("(?:\[\d+,\s*)?(\d+)(?:\])?\s*loss \(validation set\)\s*[:=]\s*\(?(%s)" % re_float, read_data), dtype=float)

# Plot the loss
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(len(loss)), loss, color="red", label="Loss")

N = 100
ma = np.convolve(loss, np.ones((N,))/N, mode='valid')
ax.plot(np.arange(len(ma))+N, ma, color="blue", label="Loss MA")

ax.set_xlabel("Mini batch")
ax.set_ylabel("loss")
ax.set_yscale('log')

ax.legend(loc="best")
fig.savefig(plot_file_loss)

# Plot the Q scores
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(*zip(*Q_training)[:2], color="green", label="Q score training batch")
ax.plot(*zip(*Q_validation)[:2], color="blue", label="Q score validation set")

ax.set_xlabel("Batches")
ax.set_ylabel("Q score")
ax.set_ylim([args.qymin, args.qymax])

ax.legend(loc="best")
fig.savefig(plot_file_Q)


# Plot the two losses
if loss_training.shape[0] > 0:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(*zip(*loss_training)[:2], color="green", label="loss training batch")
    ax.plot(*zip(*loss_validation)[:2], color="blue", label="loss score validation set")

    ax.set_xlabel("Batches")
    ax.set_ylabel("loss")
    ax.set_yscale('log')

    ax.legend(loc="best")
    fig.savefig(plot_file_losses)

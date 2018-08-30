#!/usr/bin/python

import sys
import matplotlib
# the following line is added immediately after import matplotlib
# and before import pylot. The purpose is to ensure the plotting
# works even under remote login (i.e. headless display)
matplotlib.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as pyplot
from csv import reader
import numpy as np
import argparse
import re
import os
import pdb

"""Plot MeanReward from log.csv.
To use this script to generate plot for MeanReward:
    python plotcurve.py -i /path/log.csv -o fig.png

usage: [-h] [-x X-AXIS] [-i INPUT] [-o OUTPUT] [--format FORMAT] [key [key ...]]

positional arguments:
  key                   keys of scores to plot, the default will be
                        MeanReward
  xvar                  x-axis of plot, the default will be episodes

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input filename of log, default will be standard
                        input
  -o OUTPUT, --output OUTPUT
                        output filename of figure, default will be standard
                        output
  --format FORMAT       figure format(png|pdf|ps|eps|svg)


"""

def init_xvars():
    d = {'_Episode': 'episode',
        'Time': 'time (min)',
        'Steps': 'steps',
        '_lr_multiplier': 'learning rate multiplier'}
    return d

def plot_average_return(keys, xlabel, xvars_dict, inputfile, outputfile):
    print("open file ", inputfile)
    with open(inputfile, 'r') as f:
        data = list(reader(f))
    # data[0] header
    if len(data) < 2:
        return

    if not keys:
        key = '_MeanReward'
    else:
        key = keys[0]

    if xlabel[0] in xvars_dict:
        adjx = xvars_dict[xlabel[0]]
    else:
        adjx = xlabel[0]

    rIdx = data[0].index(key)
    returns = np.array([d[rIdx] for d in data[1::]], dtype='float32')
    xIdx = data[0].index(xlabel[0])
    x_axis = np.array([d[xIdx] for d in data[1::]], dtype='float32')
    # plot
    pyplot.plot(x_axis, returns, linewidth=3.0)
    pyplot.title('average return over ' + str(adjx))
    # pyplot.xscale('log')
    pyplot.xlabel(adjx)
    pyplot.ylabel('average return')
    pyplot.savefig(outputfile, bbox_inches='tight')
    pyplot.clf()
    print("save to output file")


def main(argv):
    """
    main method of plotting curves.
    """
    cmdparser = argparse.ArgumentParser(
        "Plot MeanReward from log.csv.")
    cmdparser.add_argument(
        'key', nargs='*',
        help='key of scores to plot, the default is MeanReward')
    cmdparser.add_argument(
        '-x', 
        '--xvar',
        nargs='*',
        help='x-axis of plot, default will be episode')
    cmdparser.add_argument(
        '-i',
        '--inputs',
        nargs='*',
        help='input filename(s) of log.csv '
        'default will be standard input')
    cmdparser.add_argument(
        '-o',
        '--output',
        help='output filename of figure, '
        'default will be standard output')
    cmdparser.add_argument(
        '--format',
        help='figure format(png|pdf|ps|eps|svg)')
    cmdparser.add_argument(
        '-t',
        '--time',
        help='print returns-times or returns-samples '
        'default will be returns-samples')
    args = cmdparser.parse_args(argv)
    format = args.format
    if args.output:
        outputfile = open(args.output, 'wb')
        if not format:
            format = os.path.splitext(args.output)[1]
            if not format:
                format = 'png'
    else:
        outputfile = sys.stdout
    
    xlabel = args.xvar
    if not xlabel:
        xlabel = ['_Episode']
    xvars_dict = init_xvars()

    input_dir = "/home/tianbing/github/wall-e/src/log-files/"
    inputfiles = [input_dir + inputfile + "/log.csv" for inputfile in args.inputs]
    plot_average_return(args.key, xlabel, xvars_dict, inputfiles[0], outputfile)


if __name__ == "__main__":
    main(sys.argv[1:])

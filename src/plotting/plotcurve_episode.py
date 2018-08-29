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

"""Plot AverageReturn from progress.csv.
To use this script to generate plot for AverageReturn:
    python plotcurve.py -i progress.csv -o fig.png
    python plotcurve.py -i progress.csv -o fig.png AverageReturn

usage: [-h] [-i INPUT] [-o OUTPUT] [--format FORMAT] [key [key ...]]

positional arguments:
  key                   keys of scores to plot, the default will be
                        AverageReturn

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


def plot_average_return(keys, inputfile, outputfile):
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
    rIdx = data[0].index(key)
    returns = np.array([d[rIdx] for d in data[1::]], dtype='float32')
    eIdx = data[0].index('_Episode')
    eposides = np.array([d[eIdx] for d in data[1::]], dtype='float32')
    # plot
    pyplot.plot(eposides, returns, linewidth=3.0)
    pyplot.title('average return over episode')
    # pyplot.xscale('log')
    pyplot.xlabel('episode')
    pyplot.ylabel('average return')
    pyplot.savefig(outputfile, bbox_inches='tight')
    pyplot.clf()
    print("save to output file")


def main(argv):
    """
    main method of plotting curves.
    """
    cmdparser = argparse.ArgumentParser(
        "Plot AverageReturn from progress.csv.")
    cmdparser.add_argument(
        'key', nargs='*',
        help='key of scores to plot, the default is AverageReturn')
    cmdparser.add_argument(
        '-i',
        '--inputs',
        nargs='*',
        help='input filename(s) of progress.csv '
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

    input_dir = "/home/tianbing/github/PPO/src/log-files/"
    inputfiles = [input_dir + inputfile + "/log.csv" for inputfile in args.inputs]
    plot_average_return(args.key, inputfiles[0], outputfile)


if __name__ == "__main__":
    main(sys.argv[1:])

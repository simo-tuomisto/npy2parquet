# -*- coding: utf-8 -*-
"""This module converts numpy arrays listed in a dataframe into a parquet-file.
"""
import argparse
import pandas as pd
from npy2parquet.functions import df2parquet

parser = argparse.ArgumentParser(
    usage='This program will write numpy arrays listed in a CSV into a single parquet file')
parser.add_argument('csv', nargs=1,
                    help=('CSV that contains the names of the '
                          'npy files in one column'))
parser.add_argument('output', nargs=1, help='Output .parquet-file')
parser.add_argument('-n', '--npyfile-column', default='npyfile',
                    help=('Column that contains the names of the numpy arrays '
                          '(default: npyfile)'))
parser.add_argument('-d', '--data-column', default='data',
                    help=('Column that will store the numpy array contents '
                          '(default: data)'))
parser.add_argument('-b', '--batch-size', default=50, type=int,
                    help=('How many arrays should be stored in one parquet '
                          'row-group (default: 50)'))
parser.add_argument('-s', '--shuffle', default=False, action='store_true',
                    help=('Shuffle rows before storing the data'
                          '(default: False)'))
parser.add_argument('-r', '--seed', default=10, type=int,
                    help='Seed used for shuffling (default: 10)')
parser.add_argument('-f', '--overwrite', default=False, action='store_true',
                    help='Overwrite output file if it exists (default: False)')

args = parser.parse_args()

csv = args.csv[0]
output = args.output[0]

print(f'Creating dataset "{output}" based on "{csv}"')

csv = pd.read_csv(csv)

df2parquet(csv,
           output,
           npyfile_column=args.npyfile_column,
           data_column=args.data_column,
           batch_size=args.batch_size,
           shuffle=args.shuffle,
           seed=args.seed,
           overwrite=args.overwrite)

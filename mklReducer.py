
#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ignacio Arroyo-Fernandez'

import sys
import argparse
from operator import itemgetter
import pdb

acc = []
paths = []
# Add much results as you want. First add the corresponding argument option:
parser = argparse.ArgumentParser(description='Reducer for multiple MKL distributed jobs. Joint format, e.g. -pwr, is allowed.')
parser.add_argument('-f', type=str, dest = 'reducer_file', help='Specifies the file of outputs writen by the multiple mkl objects. If not specified, output will be printed to stdout.')
parser.add_argument('-m', dest = 'only_minimum', action='store_true', help='Include this option if you want to see only the minimum path. All other options (-p -c -w -k) are hidden.')
parser.add_argument('-p', dest = 'performances', action='store_true', help='... if you want to see the list of performances of the analyzed paths.')
parser.add_argument('-c', dest = 'paths', action='store_true', help='... if you want to see the analyzed paths.')
parser.add_argument('-w', dest = 'weights', action='store_true', help='... if you want to see the optimal weights for analyzed paths.')
parser.add_argument('-k', dest = 'kernel_params', action='store_true', help='... if you want to see the generated basis kernel parameters for the path.')
parser.add_argument('-e', dest = 'estimated_output', action='store_true', help='... if you want to see the estimated regression/labels for the test set.')
parser.add_argument('-r', dest = 'ranked', action='store_true', help='... if you want results to be presented only in ranking format (by performance).')

parser.set_defaults (only_minimum=None, performances=None, paths=None, weights=None, kernel_params=None,
                     estimated_output = None, ranked=None)
args = parser.parse_args()

if not args.only_minimum:
    print '#All paths:\n'
    r = []

d = {}
if args.reducer_file: # If a file containing outputs writen by the multiple mkl objects is specified as argument.
    f = open(args.reducer_file)
    ac = f.readlines()
else:
    ac = sys.stdin

for line in ac:
    a = line.strip().split(';')
    for att, value in args.__dict__.iteritems():
        if att == 'paths':
            if value: d['path']=a[1]
        elif att == 'weights':
            if value: d['weights'] = a[2]
        elif att == 'kernel_params':
            if value: d['kernel_params'] = a[3]
        elif att == 'estimated_output':
            if value: d['estimated_output'] = a[4]                  # Add another key from the input a[i+1]. Don't
        elif att == 'performances':                                 # forget using the same option name (e.g. weights)
            if value or args.ranked: d['performance']=float(a[0])   # both in 'elif' and in the corresponding dictionary
    if not args.only_minimum:                                       # new key, e.g. d['weights'].
        if args.ranked:
            r.append(d)
            d = {}
        else:
            acc.append((float(a[0]), a[1:]))
            print d
            d = {}

if args.ranked:
    for i in sorted(r, key=lambda k : k['performance'], reverse=True):
        print i,'\n'
else:
    a, b = max(acc)
    print '\n#The maximum performance path:\n ', 'Performance:', a,'\n','Parameters:', b

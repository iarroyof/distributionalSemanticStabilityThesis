
#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ignacio Arroyo Fernandez'

import sys
import argparse

acc = []
paths = []

parser = argparse.ArgumentParser(description='Reducer for multiple MKL distributed jobs.')
parser.add_argument('-r', type=str, dest = 'reducer_file', help='Specifies the file of outputs writen by the multiple mkl objects.')
args = parser.parse_args()

if args.reducer_file: # If a file containing outputs writen by the multiple mkl objects.
    with open(args.reducer_file) as ac:
        a = ac.readline().split(';')
        acc.append((float(a[0]), a[1]))
else: # If outputs are writen to stdin by the multiple mkl objects (piped mode)
    ac = sys.stdin
    for line in sys.stdin:
        a = line.strip().split(';')
        print a
        acc.append((float(a[0]), a[1]))

print '\nThe maximum performance path: ', max(acc)

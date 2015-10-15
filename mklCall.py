#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ignacio Arroyo Fernandez'

from mklObj import *
import pdb
from ast import literal_eval
import argparse

# [feats_train,
# feats_test,
# labelsTr,
# labelsTs] = load_multiclassToy('/home/iarroyof/shogun-data/toy/',  # Data directory
#                             'fm_train_multiclass_digits500.dat',   # Multi-class dataSet examples file name
#                             'label_train_multiclass_digits500.dat')# Multi-class Labels file name

# tr_ts_portion, fileTrain, fileLabels, dataRoute
[feats_train, feats_test, labelsTr, labelsTs] = load_binData(tr_ts_portion=0.75, fileTrain='fm_train_binary_92.dat',
                                                             fileLabels='label_train_twoclass.dat',
                                                             dataRoute='/home/iarroyof/shogun-data/toy/')

#[feats_train, feats_test, labelsTr, labelsTs] = generate_binToy()

# We have redesigned our classification problem to be binary. It is because of the semantic similarity problem proposed
# by SemEval. An input (combined) vector encodes distributional and word context information from a pair of sentences.
# We hypothesized this vector also encodes the similarity of such a pair. How ever, we thing this combination can be
# made in different ways, so we will test some of them which are inspired in signal processing theory.
binary = True
mkl_object = mklObj(binary=binary)

def mkPool(path):
    global feats_train
    global feats_test
    global labelsTr
    global labelsTs
    global mkl_object

    if path[0][0] is 'gaussian':
        a = 2*path[0][1][0]**2
        b = 2*path[0][1][1]**2
    else:
        a = path[0][1][0]
        b = path[0][1][1]
    # For now, similarity and dissimilarity are assumed to be exactly equally distributed, i.e. the amount of
    # semantically similar sentences is very likely the same than the amount of dissimilar ones over the corpus. Thus
    # the regularization parameters for the binary view of our problem are the same for both classes. In the case we
    # determine a counterapproach it is needed to determine the unbalancing proportion for including it bellow.
    if binary:
        mkl_object.mklC = [path[5], path[5]]
    else:
        mkl_object.mklC = path[5]

    mkl_object.weightRegNorm = path[4]
    mkl_object.fit_kernel(featsTr=feats_train,
                   targetsTr=labelsTr,
                   featsTs=feats_test,
                   targetsTs= labelsTs,
                   kernelFamily=path[0][0],
                   randomRange=[a, b],             # For homogeneous polynomial kernels these two parameter sets
                   randomParams=[(a + b)/2, 1.0],  # have not effect. For quadratic there isn't parameter distribution
                   hyper=path[3],       # With not effect when kernel family is polynomial and some
                   pKers=path[2])


    return mkl_object.testerr

#### Loading train and test data
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mklObject calling')
    parser.add_argument('-p', type=str, dest = 'current_path', help='Specifies the grid path to be tested by the mkl object.')
    #parser.add_argument('-t', type=int, dest = 'number_of_trials', metavar = 'N')
    args = parser.parse_args()
    path = list(literal_eval(args.current_path))
    print str(mkPool(path))+';'+str(path)


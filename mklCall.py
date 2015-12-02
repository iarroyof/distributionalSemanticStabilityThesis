#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ignacio Arroyo Fernandez'

from mklObj import *
#import pdb
from ast import literal_eval
from sys import stdout, stderr
import argparse
#//TODO: modify open_configuration_file() function for loading regression data
# files = open_configuration_file('mkl_object.conf')
# [feats_train, feats_test, labelsTr, labelsTs] = generate_binToy()
# [feats_train, feats_test, labelsTr, labelsTs] = load_binData(tr_ts_portion=0.75, fileTrain=files[0],
#                                                             fileLabels =files[1],
#                                                             dataRoute=files[2]) # Data directory
# We have redesigned our classification problem to be binary. It is because of the semantic similarity problem proposed
# by SemEval. An input (combined) vector encodes distributional and word context information from a pair of sentences.
# We hypothesized this vector also encodes the similarity of such a pair. How ever, we think this combination can be
# made in different ways, so we will test some of them which are inspired in signal processing theory.

# For now, similarity and dissimilarity are assumed to be exactly equally corpus-distributed, i.e. the amount of
# semantically similar sentences is very likely the same than the amount of dissimilar ones over the corpus. Thus
# the regularization parameters for the binary view of our problem are the same for both classes. In the case we
# determine a counterapproach it is needed to determine the unbalancing proportion for including it bellow.

# As we said above, unless there were a new approach our problem would be posed again. Nevertheless, we have
# experimented a new approach which is seeing our problem as a regression one. MKL logistic regression would be our
# ideal solution, however Shogun has not this option available. Thus we finally solve a MKL linear regression problem.
# For now, uniquely polynomial and Gaussian kernels are allowed. Other Shogun kernels are not able to handle sparse
# features. These based on explicit distance computation gave us an explicit type disagreement. Others give us a "dump
# segmentation fault". We consider now available kernels can give us sufficient empirical evidence. We have used the
# other ones for processing dense data and their behavior is to similar (at least for image classification). Thus the
# final performance rather depends on the learning hyperparameters and basis kernel parameters (e.g. C, l_p norm, kernel
# widths).

[feats_train, feats_test, labelsTr, labelsTs] = load_sparse_regressionData(fileTrain = '/home/iarroyof/sparse_train.mtx',
                                                                           fileTest = '/home/iarroyof/sparse_test.mtx',
                                                                           fileLabelsTr = '/home/iarroyof/labelSparse_train.mtx',
                                                                           fileLabelsTs = '/home/iarroyof/labelSparse_train.mtx')

problem_type = 'regression'
mkl_object = mklObj(problem=problem_type)

def mkPool(path):
    global feats_train; global feats_test; global labelsTr; global labelsTs; global mkl_object

    if path[0][0] is 'gaussian': a = 2*path[0][1][0]**2; b = 2*path[0][1][1]**2
    else: a = path[0][1][0]; b = path[0][1][1]

    if problem_type == 'binary' or problem_type == 'regression':
        mkl_object.mklC = [path[5], path[5]]
    elif problem_type == 'multiclass': mkl_object.mklC = path[5]
    mkl_object.weightRegNorm = path[4]
    mkl_object.fit_kernel(featsTr=feats_train,
                   targetsTr=labelsTr,
                   featsTs=feats_test,
                   targetsTs= labelsTs,
                   kernelFamily=path[0][0],
                   randomRange=[a, b],              # For homogeneous polynomial kernels these two parameter sets
                   randomParams=[(a + b)/2, 1.0],   # have not effect. For quadratic there isn't parameter distribution
                   hyper=path[3],                   # With not effect when kernel family is polynomial and some
                   pKers=path[2])

    return mkl_object.testerr, mkl_object.weights, mkl_object.sigmas, mkl_object.estimated_out
                                                                        # If possible, return the maximum performance
                                                                        # learned machine. It it were not possible,
                                                                        # return al learned machines and make the choice
                                                                        # of saving the best one, in order to retrieve
                                                                        # it when necessary.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mklObject calling')
    parser.add_argument('-p', type=str, dest = 'current_path', help='Specifies the grid path to be tested by the mkl object.')
    args = parser.parse_args()
    path = list(literal_eval(args.current_path))
    [performance, weights, kernel_params, output] = mkPool(path)
    #stderr.write('\n-----'+str(output)+'-----\n')
    #stderr.write('%s;%s;%s;%s;%s\n' % (performance, path, weights, kernel_params, output))
    stdout.write('%s;%s;%s;%s;%s\n' % (performance, path, weights, kernel_params, output))


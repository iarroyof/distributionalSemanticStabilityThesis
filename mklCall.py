#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ignacio Arroyo-Fernandez'

from mklObj import *
from ast import literal_eval
from sys import stdout
import argparse
#from sys import stderr
#import pdb

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
def mkl_learning_pool(path):
    global feats_train; global feats_test; global labelsTr; global labelsTs; global conf

    mkl_object = mklObj(problem=conf['problem_mode'])
    if path[0][0] is 'gaussian': a = 2*path[0][1][0]**2; b = 2*path[0][1][1]**2
    else: a = path[0][1][0]; b = path[0][1][1]

    if conf['problem_mode'] == 'binary' or conf['problem_mode'] == 'regression':
        mkl_object.mklC = [path[5], path[5]]
    elif conf['problem_mode'] == 'multiclass': mkl_object.mklC = path[5]

    mkl_object.weightRegNorm = path[4]
    mkl_object.fit_kernel(
                    featsTr=feats_train,
                    targetsTr=labelsTr,
                    featsTs=feats_test,
                    targetsTs= labelsTs,
                    kernelFamily=path[0][0],
                    randomRange=[a, b],              # For homogeneous polynomial kernels these two parameter sets
                    randomParams=[(a + b)/2, 1.0],   # have not effect. For quadratic there isn't parameter distribution
                    hyper=path[3],                   # With not effect when kernel family is polynomial and some
                    pKers=path[2])
    mkl_object.pattern_recognition(labelsTs)
    #return mkl_object.testerr, mkl_object.weights, mkl_object.sigmas, mkl_object.estimated_out
    return mkl_object.testerr, mkl_object.mkl_model, mkl_object.estimated_out                                                                    # If possible, return the maximum performance
                                                                        # learned machine. It it were not possible,
                                                                        # return al learned machines and make the choice
                                                                        # of saving the best one, in order to retrieve
                                                                        # it when necessary.

def mkl_pattern_recognition():
    """ This function loads a MKL pretrained model from specified file. All model parameters are loaded from that file,
    e.g. the kernel family, widths, subkernel weights, support, alphas, bias.
    In order to use loaded model for predicting, the mkl_object is instantiated in 'pattern_recognition' mode.
    """
    global feats_train; global feats_test; global labelsTs; global conf

    mkl_object = mklObj(problem = conf['problem_mode'], model_file = conf['model_file'], mode = conf['machine_mode'])
    mkl_object.fit_pretrained(feats_train, feats_test)
    mkl_object.pattern_recognition(targetsTs=labelsTs)

    return mkl_object.testerr, mkl_object.estimated_out

if __name__ == '__main__':
    conf = open_configuration_file('mkl_object.conf')
    [feats_train, feats_test, labelsTr, labelsTs] = load_regression_data(fileTrain = conf['training_file'],
                                                                           fileTest = conf['test_file'],
                                                                           fileLabelsTr = conf['training_labels_file'],
                                                                           fileLabelsTs = conf['test_labels_file'])
    if conf['machine_mode'] == 'learning':
        # Each training path is got either from stdin or from a path file. This part of the script is executed from
        # bash (mklParallel.sh) as many times as paths are present in the input.
        parser = argparse.ArgumentParser(description='mklObject calling')
        parser.add_argument('-p', type=str, dest = 'current_path', help='Specifies the grid path to be tested by the mkl object.')
        args = parser.parse_args()
        path = list(literal_eval(args.current_path))
        [performance, model, output] = mkl_learning_pool(path)

        stdout.write('%s;%s;%s;%s\n' % (performance, path, model, output))

        # This script (mklCall.py) is called multiple times in learning mode, so multiple outputs are written to the
        # stdout. These outputs are managed by the reducer script (mklReducer.py).
    elif conf['machine_mode'] == 'pattern_recognition':
        # The input file in pattern_recognition mode must be specified in mkl_object.conf file. Unlike the learning mode
        # no piped operation is used here, so the script is used once for a given test set. Even though the above, it is
        # possible testing several modes in pipe operation, but it is needed modifying this script for parsing these
        # models from stdin.
        [performance, output] = mkl_pattern_recognition()

        stdout.write('# Performance (determination coefficient): %s\n' % performance)
        for predicted in output:
            stdout.write('%s\n' % predicted)

        # No reducer script is used in pattern_recognition mode, so predicted values are directly printed to stdout.



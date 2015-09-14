#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ignacio Arroyo Fernandez'

from multiprocessing import Pool
#from functools import partial
import parmap as par

from mklObj import *
from gridObj import *

import pdb

""" ----------------------------------------------------------------------------------------------------
    MKL object Default definition:
    class mklObj:
        def __init__(self, weightRegNorm = 2, regPar = 2, epsilon = 1e-5,
                     threads = 2, mkl_epsilon = 0.001, binary = False, verbose = False):

    Kernel fitting function
    fit_kernel(self, featsTr,  targetsTr, featsTs, targetsTs,
                randomRange = [1, 50], randomParams = [1, 1],
                 hyper = 'linear', kernelFamily = 'guassian', pKers = 3):
    ----------------------------------------------------------------------------------------------------
"""

def mkPool(path):
    [feats_train,
        feats_test,
        labelsTr,
        labelsTs] = load_multiclassToy('/home/iarroyof/shogun-data/toy/',  # Data directory
                      'fm_train_multiclass_digits500.dat',          # Multi-class dataSet examples file name
                      'label_train_multiclass_digits500.dat')       # Multi-class Labels file name

    mkl_object = mklObj()
    if path[0][0] is 'gaussian':
        a = 2*path[0][1][0]**2
        b = 2*path[0][1][1]**2
    else:
        a = path[0][1][0]
        b = path[0][1][1] #; pdb.set_trace()
    #pdb.set_trace()
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

    del feats_train, feats_test, labelsTr,labelsTs
    return mkl_object.testerr



#mkl_kernel = mklObj()

#partial_mkPool = partial(mkPool, mkl_object = mkl_kernel, data = datax)
#### Loading train and test data
# 1) For multi-class problem loaded from file:
if __name__ == '__main__':
    perform = 1000
    minPath = {}
    p = Pool()
#### Loading the experimentation grid of parameters.
    grid = gridObj(file = 'gridParameterDic.txt')
    paths = grid.generateRandomGridPaths(trials = 3)
    [a, b, c] = paths
    #pdb.set_trace()
    r1 = p.apply_async(mkPool,[a])
    r2 = p.apply_async(mkPool,[b])
    r3 = p.apply_async(mkPool,[c])
    pdb.set_trace()
    print r1.get(timeout=50), r2.get(timeout=50), r3.get(timeout=50)
                                         # other powering forms.
    #if kernelO.testerr < perform:
     #   perform = kernelO.weights
      #  minPath = path
# mode = 'w'
# kernelO.filePrintingResults('mkl_output.txt', mode)
# kernelO.save_sigmas()

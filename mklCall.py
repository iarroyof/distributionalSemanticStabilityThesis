#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ignacio Arroyo Fernandez'

from mklObj import *
from multiprocessing import Pool
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
#### Loading train and test data
# 1) For multi-class problem loaded from file:
[feats_train,
 feats_test,
 labelsTr,
 labelsTs] = load_multiclassToy('/home/iarroyof/shogun-data/toy/',  # Data directory
                      'fm_train_multiclass_digits500.dat',          # Multi-class dataSet examples file name
                      'label_train_multiclass_digits500.dat')       # Multi-class Labels file name

# It is possible resetting the kernel for different principal parameters.
# //TODO: It is pending programming a method for loading from file a list of principal parameters:

kernelO = mklObj(mklC=10.0)

a = 2*0.5**2 # = 0.5
b = 2*10**2  # = 200
#### With n basis kernels
# //TODO: save widths each time when performance is better than prior experiment.
perform = 1000
minPath = {}

#### Loading the experimentation grid of parameters.
grid = []
with open('gridParameterDic.txt','r') as pointer:
    grid = eval(pointer.read())

def mkPool(parameterGrid):
    for path in parameterGrid:



for p in xrange(2, 15):
    kernelO.fit_kernel(featsTr=feats_train,
                   targetsTr=labelsTr,
                   featsTs=feats_test,
                   targetsTs=labelsTs,
                   kernelFamily=basisKernelFamily[0],
                   randomRange=[a, b],             # For homogeneous polynomial kernels these two parameter sets
                   randomParams=[(a + b)/2, 1.0],            # have not effect. For quadratic there isn't parameter distribution
                   hyper=widthDistribution[3],       # With not effect when kernel family is polynomial and some
                   pKers=p)                         # other powering forms.
    if kernelO.testerr < perform:
        perform = kernelO.weights
        minPath = path
# mode = 'w'
# kernelO.filePrintingResults('mkl_output.txt', mode)
# kernelO.save_sigmas()

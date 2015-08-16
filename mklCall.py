#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ignacio Arroyo Fernandez'

from mklObj import *

'''
    ----------------------------------------------------------------------------------------------------
    MKL object Default definition:
    class mklObj:
        def __init__(self, weightRegNorm = 2, regPar = 2, epsilon = 1e-5,
                     threads = 2, mkl_epsilon = 0.001, binary = False, verbose = False):

    Kernel fitting function
    fit_kernel(self, featsTr,  targetsTr, featsTs, targetsTs,
                randomRange = [1, 50], randomParams = [1, 1],
                 hyper = 'linear', kernelFamily = 'guassian', pKers = 3):
    ----------------------------------------------------------------------------------------------------------
'''
#### Loading train and test data
# 1) For multi-class problem loaded from file:
[feats_train,
 feats_test,
 labelsTr,
 labelsTs] = load_Toy('/home/iarroyof/shogun-data/toy/',  # Data directory
                      'fm_train_multiclass_digits500.dat',  # Multi-class dataSet examples file name
                      'label_train_multiclass_digits500.dat')  # Multi-class Labels file name

# It is possible resetting the kernel for different principal parameters.
# *** It is pending programming a method for loading from file a list of
# principal parameters:***
#### With m basis kernels:
basisKernelFamily = ['gaussian',
                     'inverseQuadratic',
                     'polynomial',
                     'power',
                     'rationalQuadratic',
                     'spherical',
                     'tstudent',
                     'wave',
                     'wavelet',
                     'cauchy',
                     'exponential']
#### With different kernel parameter distributions:
widthDistribution = ['linear',
                     'quadratic',
                     'log-gauss',
                     'gaussian',
                     'triangular',
                     'pareto',
                     'beta',
                     'gamma',
                     'weibull']

mode = 'w'
#### Instantiating the learnable kernel object
kernelO = mklObj()

#### With n basis kernels
# It is still missing methods for loading parameters from files****.
kernelO.fit_kernel(featsTr=feats_train,
                   targetsTr=labelsTr,
                   featsTs=feats_test,
                   targetsTs=labelsTs,
                   kernelFamily=basisKernelFamily[2],
                   randomRange=[50, 200],  # For homogeneous polynomial kernels these two parameter sets
                   randomParams=[50, 20],  # have not effect. For quadratic there isn't parameter distribution
                   hyper=widthDistribution[0],  # With not effect when kernel family is polynomial and
                   pKers=3)  # some other powering forms.
print kernelO.testerr, kernelO.weights
# kernelO.filePrintingResults('mkl_output.txt', mode)
# kernelO.save_sigmas()
kernelO.mkl.set_mkl_norm(3)

kernelO.fit_kernel(featsTr=feats_train,
                   targetsTr=labelsTr,
                   featsTs=feats_test,
                   targetsTs=labelsTs,
                   kernelFamily=basisKernelFamily[2],
                   randomRange=[50, 200],
                   randomParams=[50, 20],
                   hyper=widthDistribution[0],
                   pKers=3)

print kernelO.testerr, kernelO.weights
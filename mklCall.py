#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ignacio Arroyo Fernandez'

#from multiprocessing import Pool
#from functools import partial
#import parmap as par

from mklObj import *
from gridObj import *
import subprocess
import pdb, sys
from ast import literal_eval

[feats_train,
feats_test,
labelsTr,
labelsTs] = load_multiclassToy('/home/iarroyof/shogun-data/toy/',  # Data directory
                  'fm_train_multiclass_digits500.dat',          # Multi-class dataSet examples file name
                      'label_train_multiclass_digits500.dat')       # Multi-class Labels file name

mkl_object = mklObj()

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


    return mkl_object.testerr

#def gridGen(fileN, trials):
#    grid = gridObj(file = fileN)
        #print grid.generateRandomGridPaths(trials = int(args[2]))
#    for p in grid.generateRandomGridPaths(trials = trials): # the dict object is put to the output stream.
#       yield p

#### Loading train and test data
if __name__ == '__main__':

    path = list(literal_eval(sys.argv[1]))

    print mkPool(path)
#    perform = 0
#    minPath = []
#    weights = []
#    widths = []
#    acc = []
    #grid = gridObj(file = sys.argv[1])
    #paths = grid.generateRandomGridPaths(trials = sys.argv[2])
    #paths = gridGen(sys.argv[1], 2):#int(sys.argv[2])):
    
#    for path in paths:
        #pdb.set_trace()
#        print '\nA path: ', path
#        acc.append(mkPool(path))
#        if acc[-1] > perform:
#            perform = acc[-1]
#            minPath = path
#            widths = mkl_object.sigmas
#            weights = mkl_object.weights
#        print 'Accuracy: ', acc[-1]

#    print 'Minimum path: ', minPath
#    print 'Max accuracy: ', perform  
    
#    f = open('mkl_MinPath.txt', 'w')
#    f.write('--------- Random Grid Search results ----------------')
#    f.write('\nMinimum Path: ' + str(minPath))
#    f.write('\nMaximun Accuracy: ' + str(perform))
#    f.write('\n\nAll Accuracies: ' + str(acc))
#    f.write('\n\nWeights: ' + str(weights))
#    f.write('\n\nWidths: ' + str(widths))
#    f.close()

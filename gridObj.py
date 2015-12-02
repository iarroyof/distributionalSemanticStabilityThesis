#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ignacio Arroyo-Fernández'

from mklObj import customException
import random as rdn
from itertools import cycle
import os.path

class gridObj(object):
    """ This object class reads a dictionary of parameters from a file. These parameters are randomly selected for
     generating a grid search, for model selection.

    """

    def __init__(self, gridParams=None, file=None):

        """

        :type file: str
        """
        try:
            if file:
                if not gridParams and os.path.exists(file):
                    with open(file,'r') as pointer:
                        self.grid = eval(pointer.read())
                else:
                    raise customException('If you already gave me a name file (or a existent file name), please do not give me a parameter dictionary.')
            elif gridParams and not file:
                self.grid = gridParams

            else:
                raise customException('Give me a grid parameter dictionary explicitly or an existent name file (but not both them)...')
        except customException, (instance):
            assert isinstance(instance.parameter, object)
            print 'An error has occurred during \'gridObj\' initialization: ' + str(instance.parameter)

    def generateRandomGridPaths(self, trials = 2):
        grid_items = {}
        dic = []
        for item in self.grid:
            if trials > len(self.grid[item]): # Cycle padding for parameter lists shorter than the number of trials
                if not isinstance(self.grid[item], list):
                    keys = self.grid[item].keys()
                    rdn.shuffle(keys)
                    c = cycle(keys)
                    for i in xrange(trials):
                        dic.append(self.grid[item][next(c)])
                else:
                    keys = self.grid[item]
                    rdn.shuffle(keys)
                    c = cycle(keys)
                    for i in xrange(trials):
                        dic.append(next(c))

                grid_items[item] = dic # Returns a list of tuples. Each showing the item key and the
                dic = []                     # generated parameter list.
            else:
                if not isinstance(self.grid[item], list):
                    keys = rdn.sample(self.grid[item], trials)
                    for i in xrange(trials):
                        dic.append(self.grid[item][keys[i]])
                else:
                    keys = rdn.sample(self.grid[item], trials)
                    for i in xrange(trials):
                        dic.append(keys[i])

                grid_items[item] = dic # Returns a list of tuples. Each showing the item key and the
                dic = []

        # This code transposes the path list and removes the item names (unuseful for grid search)
        self.grid_paths =  [list(j) for j in zip(*[grid_items[i] for i in grid_items])]

        return self.grid_paths


# Object properties:
    @property
    def grid(self):
        """This is the parameter dictionary.
        :rtype : dict of dict
        """
        return self._grid

    @property
    def grid_paths(self):

        return self._grid_paths

    @property
    def trials(self):

        return self._trials


# Object setters:
    @grid_paths.setter
    def grid_paths(self, value):

        assert bool(value) # Verify if the any obtained grid path is not empty.
        self._grid_paths = value

    @trials.setter
    def trials(self, value):

        assert isinstance(value, int) and value > 0 # Verify if the number of trials is integer and grater than zero.
        self._trials = value

    @grid.setter
    def grid(self, value):
        """This method sets the grid parameter dictionary for random grid search. See the next formatting example:
        parameter_dictionary = {
        'inputSLM': {0:'1-gr_tfidf', 1:'2-gr_tfidf', 2:'3-gr_tfidf', 3:'1-gr', 4:'2-gr', 5:'3-gr'},
        'basisKernelFamily': {0:('gaussian', [0.5, 50]), 1:('inverseQuadratic', [0, 10]), 2:('polynomial', [1, 5]),
                                3:('power', [1, 5]), 4:('rationalQuadratic': [0, 10]), 5:('spherical', [0.5, 50]),
                                6:('tstudent', [0, 5]), 7:('wave', [0.5, 100]), 8:('wavelet', [0.5, 50]),
                                9:('cauchy', [0.5, 50]), 10:('exponential', [0.5, 50])},
        'basisKernelParameterDistribution': {0:'linear', 1:'quadratic', 2:'loggauss', 3:'gaussian', 4:'triangular',
                                                5:'pareto', 6:'beta', 7:'gamma', 8:'weibull'},
        'linearCombinationSize': [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40],
        'weightRegularizer': [0.5, 1.0, 1.5, 2.0, 5.0],
        'regularizationParameter': [0.1,  0.2,  0.4,  0.6,  0.8,  1.0,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.8,  2.0,
                                    2.5,  3.0,  3.5,  4.4,  4.5,  5.0,  7.0,  9.0,  10.0,  20.0,  30.0,  40.0,  50.0,
                                    100.0,  1000.0]
        }

        @type value: dict of dict
        """
        try:
            if not value:
                raise customException('The parameter dictionary is empty.')

            for param in value:
                if not param:
                    raise customException('A parameter sub-dictionary is empty: {0}'.format(param))

        except customException, (instance):
            print 'An error has occurred during \'gridObj\' initialization: ' + instance.parameter

        self._grid = value
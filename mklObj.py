#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
    
__author__ = 'Ignacio Arroyo-Fernandez'

from modshogun import *
from tools.load import LoadMatrix
from sklearn import r2_score
import random
from math import sqrt
import numpy
from os import getcwd
from sys import stderr
from pdb import set_trace as st

def open_configuration_file(fileName):
    """ Loads the input data configuration file. Lines which start with '#' are ignored. No lines different from
    configuration ones (even blank ones) at top are allowed. The amount of lines at top are exclusively either three or
    five (see below for allowed contents).

    The first line may be specifying the train data file in sparse market matrix format.
    The second line may be specifying the test data file in sparse market matrix format.
    The third line may be specifying the train labels file. An scalar by line must be associated as label of a vector
    in training data.
    The fourth line may be specifying the test labels file. An scalar by line must be associated as label of a vector
    in test data.
    The fifth line indicates options for the MKL object:
        First character : Problem type : valid_options = {r: regression, b: binary, m: multiclass}
        Second character: Machine mode : valid_options = {l: learning_mode, p: pattern_recognition_mode}
    Any other characters and amount of they will be ignored or caught as errors.
    For all configuration lines no other kind of content is allowed (e.g. comments in line ahead).
    Training data (and its labels) is optional. Whenever no five configuration lines are detected in this file,
    the first line will be considered as the test data file name, the second line as de test labels and third line as
    the MKL options. An error exception will be raised otherwise (e.g. no three or no five configuration lines).
    """
    with open(fileName) as f:
        configuration_lines = f.read().splitlines()
    problem_modes = {'r':'regression', 'b':'binary', 'm':'multiclass'}
    machine_modes = {'l':'learning', 'p':'pattern_recognition'}
    cls = 0     # Counted number of configuration lines from top.
    ncls = 5    # Number of configuration lines allowed.
    for line in configuration_lines:
        if not line.startswith('#'):
            cls += 1
        else:
            break

    if cls == ncls:
        mode = configuration_lines[4]
        configuration = {}
        if len(mode) == 2:
            try:
                configuration['problem_mode'] = problem_modes[mode[0]]
                configuration['machine_mode'] = machine_modes[mode[1]]
            except KeyError:
                sys.stderr.write('\nERROR: Incorrect configuration file. Invalid machine mode. See help for mklObj.open_configuration_file().')
    else:
        sys.stderr.write('\nERROR: Incorrect configuration file. Invalid number of lines. See help for mklObj.open_configuration_file().')
        exit()

    Null = ncls                                     # Null index
    if configuration['machine_mode'] == 'learning': # According to availability of training files, indexes are setted.
        trf = 0; tsf = 1; trlf = 2          # training_file, test_file, training_labels_file, test_labels_file, mode
        tslf = 3; mf = Null
        configuration_lines[ncls] = None
        del(configuration_lines[ncls+1:])   # All from the first '#' onwards is ignored.
    elif configuration['machine_mode'] == 'pattern_recognition':
        trf = 0; tsf = 1; trlf = Null       # training_file, test_file, test_labels_file, mode, model_file
        tslf = 2; mf = 3
        configuration_lines[ncls] = None
        del(configuration_lines[ncls+1:])

    configuration['training_file'] = configuration_lines[trf]
    configuration['test_file'] = configuration_lines[tsf]
    configuration['training_labels_file'] = configuration_lines[trlf]
    configuration['test_labels_file'] = configuration_lines[tslf]
    configuration['model_file'] = configuration_lines[mf]

    return configuration

# Loading toy multiclass data from files
def load_multiclassToy(dataRoute, fileTrain, fileLabels):
    """ :returns: [RealFeatures(training_data), RealFeatures(test_data), MulticlassLabels(train_labels),
    MulticlassLabels(test_labels)]. It is a set of Shogun training objects for raising a 10-class classification
    problem. This function is a modified version from http://www.shogun-toolbox.org/static/notebook/current/MKL.html
    Pay attention to input parameters because their documentations is valid for acquiring data for any multiclass
    problem with Shogun.

    :param dataRoute: The allocation directory of plain text file containing the train and test data.
    :param fileTrain: The name of the text file containing the train and test data. Each row of the file contains a
    sample vector and each column is a dimension of such a sample vector.
    :param fileLabels: The name of the text file containing the train and test labels. Each row must to correspond to
    each sample in fileTrain. It must be at the same directory specified by dataRoute.
    """
    lm = LoadMatrix()
    dataSet = lm.load_numbers(dataRoute + fileTrain)
    labels = lm.load_labels(dataRoute + fileLabels)

    return (RealFeatures(dataSet.T[0:3 * len(dataSet.T) / 4].T),  # Return the training set, 3/4 * dataSet
            RealFeatures(dataSet.T[(3 * len(dataSet.T) / 4):].T),  # Return the test set, 1/4 * dataSet
            MulticlassLabels(labels[0:3 * len(labels) / 4]),  # Return corresponding train and test labels
            MulticlassLabels(labels[(3 * len(labels) / 4):]))

# 2D Toy data generator
def generate_binToy(file_data = None, file_labels = None):
    """:return: [RealFeatures(train_data),RealFeatures(train_data),BinaryLabels(train_labels),BinaryLabels(test_labels)]
    This method generates random 2D training and test data for binary classification. The labels are {-1, 1} vectors.
    """
    num = 30
    num_components = 4
    means = numpy.zeros((num_components, 2))
    means[0] = [-1, 1]
    means[1] = [2, -1.5]
    means[2] = [-1, -3]
    means[3] = [2, 1]

    covs = numpy.array([[1.0, 0.0], [0.0, 1.0]])

    gmm = GMM(num_components)
    [gmm.set_nth_mean(means[i], i) for i in range(num_components)]
    [gmm.set_nth_cov(covs, i) for i in range(num_components)]
    gmm.set_coef(numpy.array([1.0, 0.0, 0.0, 0.0]))
    xntr = numpy.array([gmm.sample() for i in xrange(num)]).T
    xnte = numpy.array([gmm.sample() for i in xrange(5000)]).T
    gmm.set_coef(numpy.array([0.0, 1.0, 0.0, 0.0]))
    xntr1 = numpy.array([gmm.sample() for i in xrange(num)]).T
    xnte1 = numpy.array([gmm.sample() for i in xrange(5000)]).T
    gmm.set_coef(numpy.array([0.0, 0.0, 1.0, 0.0]))
    xptr = numpy.array([gmm.sample() for i in xrange(num)]).T
    xpte = numpy.array([gmm.sample() for i in xrange(5000)]).T
    gmm.set_coef(numpy.array([0.0, 0.0, 0.0, 1.0]))
    xptr1 = numpy.array([gmm.sample() for i in xrange(num)]).T
    xpte1 = numpy.array([gmm.sample() for i in xrange(5000)]).T

    if not file_data:

        return (RealFeatures(numpy.concatenate((xntr, xntr1, xptr, xptr1), axis=1)),  # Train Data
            RealFeatures(numpy.concatenate((xnte, xnte1, xpte, xpte1), axis=1)),  # Test Data
            BinaryLabels(numpy.concatenate((-numpy.ones(2 * num), numpy.ones(2 * num)))),  # Train Labels
            BinaryLabels(numpy.concatenate((-numpy.ones(10000), numpy.ones(10000)))))  # Test Labels
    else:

        data_set = numpy.concatenate((numpy.concatenate((xntr, xntr1, xptr, xptr1), axis=1),
                                      numpy.concatenate((xnte, xnte1, xpte, xpte1), axis=1)), axis = 1).T
        labels = numpy.concatenate((numpy.concatenate((-numpy.ones(2 * num), numpy.ones(2 * num))),
                                    numpy.concatenate((-numpy.ones(10000), numpy.ones(10000)))), axis = 1).astype(int)

        indexes = range(len(data_set))
        numpy.random.shuffle(indexes)
        fd = open(file_data, 'w')
        fl = open(file_labels, 'w')
        for i in indexes:
            fd.write('%f %f\n' % (data_set[i][0],data_set[i][1]))
            fl.write(str(labels[i])+'\n')

        fd.close()
        fl.close()
        #numpy.savetxt(file_data, data_set, fmt='%f')
        #numpy.savetxt(file_labels, labels, fmt='%d')

def load_binData(tr_ts_portion = None, fileTrain = None, fileLabels = None, dataRoute = None):
    if not dataRoute:
        dataRoute = getcwd()+'/'
    assert fileTrain and fileLabels # One (or both) of the input files are not given.
    assert (tr_ts_portion > 0.0 and tr_ts_portion <= 1.0) # The proportion of dividing the data set into train and test is in (0, 1]

    lm = LoadMatrix()
    dataSet = lm.load_numbers(dataRoute + fileTrain)
    labels = lm.load_labels(dataRoute + fileLabels)

    return (RealFeatures(dataSet.T[0:tr_ts_portion * len(dataSet.T)].T),  # Return the training set, 3/4 * dataSet
            RealFeatures(dataSet.T[tr_ts_portion * len(dataSet.T):].T),  # Return the test set, 1/4 * dataSet
            BinaryLabels(labels[0:tr_ts_portion * len(labels)]),  # Return corresponding train and test labels
            BinaryLabels(labels[tr_ts_portion * len(labels):]))

def load_regression_data(fileTrain = None, fileTest = None, fileLabelsTr = None, fileLabelsTs = None, sparse=False):
    """ This method loads data from sparse mtx file format ('CSR' preferably. See Python sci.sparse matrix
    format, also referred to as matrix market read and write methods). Label files should contain a column of
    these labels, e.g. see the contents of a three labels file:

     1.23
     -102.45
     2.2998438943

    Loading uniquely test labels is allowed (training labels are optional). In pattern_recognition mode no
    training labels are required. None is returned out for corresponding Shogun label object. Feature list
    returned:
    [features_tr, features_ts, labels_tr, labels_ts]
    Returned data is float type (dtype='float64'). This is the minimum data length allowed by Shogun given the 
    sparse distance functions does not allow other ones, e.g. short (float32).
    """
    assert fileTrain and fileTest and fileLabelsTs # Necessary test labels as well as test and train data sets specification.
    from scipy.io import mmread

    lm = LoadMatrix()
    if sparse:	
        sci_data_tr = mmread(fileTrain).asformat('csr').astype('float64').T
        features_tr = SparseRealFeatures(sci_data_tr)                       # Reformated as CSR and 'float64' type for
        sci_data_ts = mmread(fileTest).asformat('csr').astype('float64').T    # compatibility with SparseRealFeatures
        features_ts = SparseRealFeatures(sci_data_ts)
    else:
        features_tr = RealFeatures(lm.load_numbers(fileTrain).astype('float64'))
        features_ts = RealFeatures(lm.load_numbers(fileTest).astype('float64'))

    labels_ts = RegressionLabels(lm.load_labels(fileLabelsTs))

    if fileTrain and fileLabelsTr: # sci_data_x: Any sparse data type in the file.
        labels_tr = RegressionLabels(lm.load_labels(fileLabelsTr))
    else:
        labels_tr = None

    return features_tr, features_ts, labels_tr, labels_ts

# Exception handling:
class customException(Exception):
    """ This exception prevents training inconsistencies. It could be edited for accepting a complete
    dictionary of exceptions if desired.
    """
    def __init__(self, message):
        self.parameter = message

    def __str__(self):
        return repr(self.parameter)

# Basis kernel parameter generation:
def sigmaGen(self, hyperDistribution, size, rango, parameters):
    """ :return: list of float
    This module generates the pseudorandom vector of widths for basis Gaussian kernels according to a distribution, i.e.
    hyperDistribution =
                         {'linear',
                          'quadratic',
                          'loggauss'*,
                          'gaussian'*,
                          'triangular', # parameters[0] is the median of the distribution. parameters[1] has not effect.
                          'pareto',
                          'beta'*,
                          'gamma',
                          'weibull'}.
    Names marked with * require parameters, e.g. for 'gaussian', parameters = [mean, width]. The input 'size' is the
    amount of segments the distribution domain will be discretized out. The 'rango' input are the minimum and maximum
    values of the obtained distributed values. The 'parameters' of these weight vector distributions are set to common
    values of each distribution by default, but they can be modified.

    :param hyperDistribution: string
    :param size: It is the number of basis kernels for the MKL object.
    :param rango: It is the range to which the basis kernel parameters will pertain. For some basis kernels families
    this input parameter has not effect.
    :param parameters: It is a list of parameters of the distribution of the random weights, e.g. for a gaussian
    distribution with mean zero and variance 1, parameters = [0, 1]. For some basis kernel families this input parameter
    has not effect: {'linear', 'quadratic', 'triangular', 'pareto', 'gamma', 'weilbull', }

    .. seealso: fit_kernel() function documentation.
    """
    # Validating th inputs
    assert (isinstance(size, int) and size > 0)
    assert (rango[0] < rango[1] and len(rango) == 2)
    # .. todo: Revise the other linespaces of the other distributions. They must be equally consistent than the
    # .. todo: Gaussian one. Change 'is' when verifying equality between strings (PEP008 recommendation).
    sig = []
    if hyperDistribution == 'linear':
        line = numpy.linspace(rango[0], rango[1], size*2)
        sig = random.sample(line, size)
        return sig
    elif hyperDistribution == 'quadratic':
        sig = numpy.square(random.sample(numpy.linspace(int(sqrt(rango[0])), int(sqrt(rango[1]))), size))
        return sig
    elif hyperDistribution == 'gaussian':
        assert parameters[1] > 0 # The width is greater than zero?
        i = 0
        while i < size:
            numero = random.gauss(parameters[0], parameters[1])
            if rango[0] <= numero <= rango[1]:  # Validate the initial point of
                sig.append(numero)  # 'range'. If not met, loop does
                i += 1  # not end, but resets
                # If met, the number is appended
        return sig  # to 'sig' width list.
    elif hyperDistribution == 'triangular':
        assert rango[0] <= parameters[0] <= rango[1] # The median is in the range?
        sig = numpy.random.triangular(rango[0], parameters[0], rango[1], size)
        return sig
    elif hyperDistribution == 'beta':
        assert (parameters[0] >= 0 and parameters[1] >= 0) # Alpha and Beta parameters are non-negative?
        sig = numpy.random.beta(parameters[0], parameters[1], size) * (rango[1] - rango[0]) + rango[0]
        return sig
    elif hyperDistribution == 'pareto':
        return numpy.random.pareto(5, size=size) * (rango[1] - rango[0]) + rango[0]

    elif hyperDistribution == 'gamma':
        return numpy.random.gamma(shape=1, size=size) * (rango[1] - rango[0]) + rango[0]

    elif hyperDistribution == 'weibull':
        return numpy.random.weibull(2, size=size) * (rango[1] - rango[0]) + rango[0]

    elif hyperDistribution == 'loggauss':
        assert parameters[1] > 0 # The width is greater than zero?
        i = 0
        while i < size:
            numero = random.lognormvariate(parameters[0], parameters[1])
            if numero > rango[0] and numero < rango[1]:
                sig.append(numero)
                i += 1

        return sig
    else:
        print 'The entered hyperparameter distribution is not allowed: '+hyperDistribution
        #pdb.set_trace()

# Combining kernels
def genKer(self, featsL, featsR, basisFam, widths=[5.0, 4.0, 3.0, 2.0, 1.0], sparse = False):
    """:return: Shogun CombinedKernel object.
    This module generates a list of basis kernels. These kernels are tuned according to the vector ''widths''. Input
    parameters ''featsL'' and ''featsR'' are Shogun feature objects. In the case of a learnt RKHS, these both objects
    should be derived from the training SLM vectors, by means of the Shogun constructor realFeatures(). This module also
    appends basis kernels to a Shogun combinedKernel object.

    The kernels to be append are left in ''combKer'' object (see code), which is returned. We have analyzed some basis
    families available in Shogun, so possible string values of 'basisFam' are:

    basisFam = ['gaussian',
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
    """
    allowed_sparse = ['gaussian', 'polynomial'] # Change this assertion list and function if different kernels are needed.
    assert not (featsL.get_feature_class() == featsR.get_feature_class() == 'C_SPARSE') or basisFam in allowed_sparse # Sparse type is not compatible with specified kernel or feature types are different.
    kernels = []
    if basisFam == 'gaussian':
        for w in widths:
            k=GaussianKernel()
            #k.init(featsL, featsR)
            #st()
            kernels.append(k)
            kernels[-1].set_width(w)
            kernels[-1].init(featsL, featsR)
            #st()
    elif basisFam == 'inverseQuadratic':  # For this (and others below) kernel it is necessary fitting the
        if not sparse:
            dst = MinkowskiMetric(l=featsL, r=featsR, k=2)  # distance matrix at this moment k = 2 is for l_2 norm
        else:
            dst = SparseEuclideanDistance(l=featsL, r=featsR)
        for w in widths:
            kernels.append(InverseMultiQuadricKernel(0, w, dst))

    elif basisFam == 'polynomial':
        for w in widths:
            kernels.append(PolyKernel(0, w, False))

    elif basisFam == 'power':  # At least for images, the used norm does not make differences in performace
        if not sparse:
            dst = MinkowskiMetric(l=featsL, r=featsR, k=2)
        else:
            dst = SparseEuclideanDistance(l=featsL, r=featsR)
        for w in widths:
            kernels.append(PowerKernel(0, w, dst))

    elif basisFam == 'rationalQuadratic':  # At least for images, using 3-norm  make differences
        if not sparse:
            dst = MinkowskiMetric(l=featsL, r=featsR, k=2)  # in performance
        else:
            dst = SparseEuclideanDistance(l=featsL, r=featsR)
        for w in widths:
            kernels.append(RationalQuadraticKernel(0, w, dst))

    elif basisFam == 'spherical':  # At least for images, the used norm does not make differences in performace
        if not sparse:
            dst = MinkowskiMetric(l=featsL, r=featsR, k=2)
        else:
            dst = SparseEuclideanDistance(l=featsL, r=featsR)
        for w in widths:
            kernels.append(SphericalKernel(0, w, dst))

    elif basisFam == 'tstudent':  # At least for images, the used norm does not make differences in performace
        if not sparse:
            dst = MinkowskiMetric(l=featsL, r=featsR, k=2)
        else:
            dst = SparseEuclideanDistance(l=featsL, r=featsR)
        for w in widths:
            kernels.append(TStudentKernel(0, w, dst))

    elif basisFam == 'wave':  # At least for images, the used norm does not make differences in performace
        if not sparse:
            dst = MinkowskiMetric(l=featsL, r=featsR, k=2)
        else:
            dst = SparseEuclideanDistance(l=featsL, r=featsR)
        for w in widths:
            kernels.append(WaveKernel(0, w, dst))

    elif basisFam == 'wavelet' and not sparse:  # At least for images it is very low the performance with this kernel.
        for w in widths:  # It remains pending, for now, analysing its parameters.
            kernels.append(WaveletKernel(0, w, 0))

    elif basisFam == 'cauchy':
        if not sparse:
            dst = MinkowskiMetric(l=featsL, r=featsR, k=2)
        else:
            dst = SparseEuclideanDistance(l=featsL, r=featsR)
        for w in widths:
            kernels.append(CauchyKernel(0, w, dst))

    elif basisFam == 'exponential':  # For this kernel it is necessary specifying features at the constructor
        if not sparse:
            dst = MinkowskiMetric(l=featsL, r=featsR, k=2)
        else:
            dst = SparseEuclideanDistance(l=featsL, r=featsR)
        for w in widths:
            kernels.append(ExponentialKernel(featsL, featsR, w, dst, 0))

    elif basisFam == 'anova' and not sparse:  # This kernel presents a warning in training:
        """RuntimeWarning: [WARN] In file /home/iarroyof/shogun/src/shogun/classifier/mkl/MKLMulticlass.cpp line
           198: CMKLMulticlass::evaluatefinishcriterion(...): deltanew<=0.Switching back to weight norsm
           difference as criterion.
        """
        for w in widths:
            kernels.append(ANOVAKernel(0, w))

    else:
        raise NameError('Unknown Kernel family name!!!')
    
    combKer = CombinedKernel()
    #features_tr = CombinedFeatures()
    for k in kernels:
        combKer.append_kernel(k)
        #features_tr.append_feature_obj(featsL)    
    
    #combKer.init(features_tr, features_tr)
    #combKer.init(featsL,featsR)
    
    return combKer#, features_tr

# Defining the compounding kernel object
class mklObj(object):
    """Default self definition of the Multiple Kernel Learning object. This object uses previously defined methods for
    generating a linear combination of basis kernels that can be constituted from different families. See at
    fit_kernel() function documentation for details. This function trains the kernel weights. The object has other
    member functions offering utilities. See the next instantiation and using example:

        import mkl01 as mk

        kernel = mk.mklObj(weightRegNorm = 2,
                        mklC = 2,
                        SVMepsilon = 1e-5,
                        threads = 2,
                        MKLepsilon = 0.001,
                        probome = 'Multiclass',
                        verbose = False) # IMPORTANT: Don't use this feature (True) if you are working in pipe mode.
                                         # The object will print undesired outputs to the stdout.
    The above values are the defaults, so if they are suitable for you it is possible instantiating the object by simply
    stating: kernel = mk.mklObj(). Even it is possible modifying a subset of input parameters (keeping others as
    default): kernel = mk.mklObj(weightRegNorm = 1, mklC = 10, SVMepsilon = 1e-2). See the documentation of each setter
    below for allowed setting parameters without new instantiations.

    Now, once main parameters has been setted, fit the kernel:
        kernel.fit_kernel(featsTr =        feats_train,
                        targetsTr =      labelsTr,
                        featsTs =        feats_test,
                        targetsTs =      labelsTs,
                        kernelFamily =   'gaussian',
                        randomRange =    [50, 200],  # For homogeneous poly kernels these two parameter
                        randomParams =   [50, 20],   # sets have not effect. No basis kernel parameters
                        hyper =          'linear',   # Also with not effect when kernel family is polynomial
                        pKers =          3)          # and some other powering forms.

    Once the MKL object has been fitted, you can get what you need from it. See getters documentation listed below.
    """

    def __init__(self, weightRegNorm=2.0, mklC=2.0, SVMepsilon=1e-5, model_file = None,
                 threads=4, MKLepsilon=0.001, problem='regression', verbose=False, mode = 'learning', sparse = False):
        """Object initialization. This procedure is regardless of the input data, basis kernels and corresponding
        hyperparameters (kernel fitting).
        """
        mkl_problem_object = {'regression':(MKLRegression, [mklC, mklC]),
                              'binary': (MKLClassification, [mklC, mklC]),
                              'multiclass': (MKLMulticlass, mklC)}
        self.mode = mode
        self.sparse = sparse
        assert not model_file and mode != 'pattern_recognition' or (
                   model_file and mode == 'pattern_recognition')# Model file or pattern_recognition mode must be specified.
        self.__problem = problem
        self.verbose = verbose  # inner training process verbose flag
        self.Matrx = False  # Kind of returned learned kernel object. See getter documentation of these
        self.expansion = False  # object configuration parameters for details. Only modifiable by setter.
        self.__testerr = 0

        if mode == 'learning':
            try:
                self.mkl = mkl_problem_object[problem][0]()
                self.mklC = mkl_problem_object[problem][1]
            except KeyError:
                sys.stderr.write('Error: Given problem type is not valid.')
                exit()

            self.weightRegNorm = weightRegNorm  # Setting the basis' weight vector norm
            self.SVMepsilon = SVMepsilon  # setting the transducer stop (convergence) criterion
            self.MKLepsilon = MKLepsilon  # setting the MKL stop criterion. The value suggested by
            # Shogun examples is 0.001. See setter docs for details
        elif mode == 'pattern_recognition':
            [self.mkl, self.mkl_model] = self.load_mkl_model(file_name = model_file, model_type = problem)
            self.sigmas = self.mkl_model['widths']

        self.threads = threads  # setting number of training threads. Verify functionality!!

    def fit_pretrained(self, featsTr, featsTs):
        """ This method sets up a MKL machine by using parameters from self.mkl_model preloaded dictionary which
        contains preptrained model paremeters, e.g. weights and widths. 
        """
        self.ker = genKer(self, featsTr, featsTs, sparse = self.sparse,
                          basisFam = self.family_translation[self.mkl_model['family']], widths = self.sigmas)
        self.ker.set_subkernel_weights(self.mkl_model['weights'])  # Setting up pretrained weights to the
        self.ker.init(featsTr, featsTs)                            # new kernel

    # Self Function for kernel generation
    def fit_kernel(self, featsTr, targetsTr, featsTs, targetsTs, randomRange=[1, 50], randomParams=[1, 1],
                   hyper='linear', kernelFamily='gaussian', pKers=3):
        """ :return: CombinedKernel Shogun object.
        This method is used for training the desired compound kernel. See documentation of the 'mklObj'
        object for using example. 'featsTr' and 'featsTs' are the training and test data respectively.
        'targetsTr' and 'targetsTs' are the training and test labels, respectively. All they must be Shogun
        'RealFeatures' and 'MulticlassLabels' objects respectively.
        The 'randomRange' parameter defines the range of numbers from which the basis kernel parameters will be
        drawn, e.g. Gaussian random widths between 1 and 50 (the default). The 'randomParams' input parameter
        states the parameters of the pseudorandom distribution of the basis kernel parameters to be drawn, e.g.
        Gaussian-pseudorandom-generated weights with std. deviation equal to 1 and mean equal to 1 (the default).
        The 'hyper' input parameter defines the distribution of the pseudorandom-generated weights. See
        documentation of the sigmaGen() method of the 'mklObj' object to see a list of possible basis kernel
        parameter distributions. The 'kernelFamily' input parameter is the basis kernel family to be append to
        the desired compound kernel if you select, e.g., the default 'gaussian' family, all elements of the
        learned linear combination will be gaussians (each differently weighted and parametrized). See
        documentation of the genKer() method of the 'mklObj' object to see a list of allowed basis kernel
        families. The 'pKers' input parameter defines the size of the learned kernel linear combination, i.e.
        how many basis kernels to be weighted in the training and therefore, how many coefficients will have the
        Fourier series of data (the default is 3).

        .. note:: In the cases of kernelFamily = {'polynomial' or 'power' or 'tstudent' or 'anova'}, the input
        parameters {'randomRange', 'randomParams', 'hyper'} have not effect, because these kernel families do not
        require basis kernel parameters.

        :param featsTr: RealFeatures Shogun object conflating the training data.
        :param targetsTr: MulticlassLabels Shogun object conflating the training labels.
        :param featsTr: RealFeatures Shogun object conflating the test data.
        :param targetsTr: MulticlassLabels Shogun object conflating the test labels.
        :param randomRange: It is the range to which the basis kernel parameters will pertain. For some basis
         kernels families this input parameter has not effect.
        :param randomParams: It is a list of parameters of the distribution of the random weights, e.g. for a
         gaussian distribution with mean zero and variance 1, parameters = [0, 1]. For some basis kernel
         families this input parameter has not effect.
        :param hyper: string which specifies the name of the basis kernel parameter distribution. See
         documentation for sigmaGen() function for viewing allowed strings (names).
        :param kernelFamily: string which specifies the name of the basis kernel family. See documentation for
        genKer() function for viewing allowed strings (names).
        :param pKers: This is the number of basis kernels for the MKL object (linear combination).

        """
        # Inner variable copying:
        self._featsTr = featsTr
        self._targetsTr = targetsTr
        self._hyper = hyper
        self._pkers = pKers
        self.basisFamily = kernelFamily
        if self.verbose:  # Printing the training progress
            print '\nNacho, multiple <' + kernelFamily + '> Kernels have been initialized...'
            print "\nInput main parameters: "
            print "\nHyperarameter distribution: ", self._hyper, "\nLinear combination size: ", pKers, \
                '\nWeight regularization norm: ', self.weightRegNorm, \
                'Weight regularization parameter: ',self.mklC
            if self.__problem == 'multiclass':
                print "Classes: ", targetsTr.get_num_classes()
            elif self.__problem == 'binary':
                print "Classes: Binary"
            elif self.__problem == 'regression':
                print 'Regression problem'

        # Generating the list of subkernels. Creating the compound kernel. For monomial-nonhomogeneous (polynomial)
        # kernels the hyperparameters are uniquely the degree of each monomial, in the form of a sequence. MKL finds the
        # coefficient (weight) for each monomial in order to find a compound polynomial.
        if kernelFamily == 'polynomial' or kernelFamily == 'power' or \
                        kernelFamily == 'tstudent' or kernelFamily == 'anova':
            self.sigmas = range(1, pKers+1)
            self.ker = genKer(self, self._featsTr, self._featsTr, basisFam=kernelFamily, widths=self.sigmas, sparse = self.sparse)
        else:
        # We have called 'sigmas' to any basis kernel parameter, regardless if the kernel is Gaussian or not. So
        # let's generate the widths:
            self.sigmas = sorted(sigmaGen(self, hyperDistribution=hyper, size=pKers,
                                          rango=randomRange, parameters=randomParams))
            try:
                z = self.sigmas.index(0)
                self.sigmas[z] = 0.1
            except ValueError:
                pass

            try:  # Verifying if number of kernels is greater or equal to 2
                if pKers <= 1 or len(self.sigmas) < 2:
                    raise customException('Senseless MKLClassification use!!!')
            except customException, (instance):
                print 'Caugth: ' + instance.parameter
                print "-----------------------------------------------------"
                print """The multikernel learning object is meaningless for less than 2 basis
                     kernels, i.e. pKers <= 1, so 'mklObj' couldn't be instantiated."""
                print "-----------------------------------------------------"

            self.ker = genKer(self, self._featsTr, self._featsTr, basisFam=kernelFamily, widths=self.sigmas, sparse = self.sparse)
            if self.verbose:
                print 'Widths: ', self.sigmas
        # Initializing the compound kernel
        
#        combf_tr = CombinedFeatures()
#        combf_tr.append_feature_obj(self._featsTr)
#        self.ker.init(combf_tr, combf_tr)
        try:  # Verifying if number of  kernels was greater or equal to 2 after training
            if self.ker.get_num_subkernels() < 2:
                raise customException(
                    'Multikernel coefficients were less than 2 after training. Revise object settings!!!')
        except customException, (instance):
            print 'Caugth: ' + instance.parameter

        # Verbose for learning surveying
        if self.verbose:
            print '\nKernel fitted...'
        # Initializing the transducer for multiclassification
        features_tr = CombinedFeatures()
        features_ts = CombinedFeatures()
        for k in self.sigmas:
            features_tr.append_feature_obj(self._featsTr)
            features_ts.append_feature_obj(featsTs)
        
        self.ker.init(features_tr, features_tr)
        
        self.mkl.set_kernel(self.ker)
        self.mkl.set_labels(self._targetsTr)
        # Train to return the learnt kernel
        if self.verbose:
            print '\nLearning the machine coefficients...'
        # ------------------ The most time consuming code segment --------------------------
        self.crashed = False
        try:
            self.mkl.train()
        except SystemError:
            self.crashed = True

        self.mkl_model = self.keep_mkl_model(self.mkl, self.ker, self.sigmas) # Let's keep the trained model
        if self.verbose:                                                      # for future use.
            print 'Kernel trained... Weights: ', self.weights
        # Evaluate the learnt Kernel. Here it is assumed 'ker' is learnt, so we only need for initialize it again but
        # with the test set object. Then, set the initialized kernel to the mkl object in order to 'apply'.

        self.ker.init(features_tr, features_ts)   # Now with test examples. The inner product between training
        #st()
    def pattern_recognition(self, targetsTs):
        self.mkl.set_kernel(self.ker)           # and test examples generates the corresponding Gram Matrix.
        if not self.crashed:
            out = self.mkl.apply()  # Applying the obtained Gram Matrix
        else:
            out = RegressionLabels(-1.0*numpy.ones(targetsTs.get_num_labels()))

        self.estimated_out = list(out.get_labels())
        # ----------------------------------------------------------------------------------
        if self.__problem == 'binary':          # If the problem is either binary or multiclass, different
            evalua = ErrorRateMeasure()         # performance measures are computed.
            self.__testerr = 100 - evalua.evaluate(out, targetsTs) * 100
        elif self.__problem == 'multiclass':
            evalua = MulticlassAccuracy()
            self.__testerr = evalua.evaluate(out, targetsTs) * 100
        elif self.__problem == 'regression': # Determination Coefficient was selected for measuring performance
            #evalua = MeanSquaredError()
            #self.__testerr = evalua.evaluate(out, targetsTs)
            self.__testerr = r2_score(self.estimated_out,  list(targetsTs.get_labels()))

        # Verbose for learning surveying
        if self.verbose:
            print 'Kernel evaluation ready. The precision was: ', self.__testerr, '%'

    def keep_mkl_model(self, mkl, kernel, widths, file_name = None):
        """ Python reimplementated function for saving a pretrained MKL machine.
        This method saves a trained MKL machine to the file 'file_name'. If not 'file_name' is given, a
        dictionary 'mkl_machine' containing parameters of the given trained MKL object is returned.
        Here we assumed all subkernels of the passed CombinedKernel are of the same family, so uniquely the
        first kernel is used for verifying if the passed 'kernel' is a Gaussian mixture. If it is so, we insert
        the 'widths' to the model dictionary 'mkl_machine'. An error is returned otherwise.
        """
        mkl_machine = {}
        support=[]
        mkl_machine['num_support_vectors'] = mkl.get_num_support_vectors()
        mkl_machine['bias']=mkl.get_bias()
        for i in xrange(mkl_machine['num_support_vectors']):
            support.append((mkl.get_alpha(i), mkl.get_support_vector(i)))

        mkl_machine['support'] = support
        mkl_machine['weights'] = list(kernel.get_subkernel_weights())
        mkl_machine['family'] = kernel.get_first_kernel().get_name()
        mkl_machine['widths'] = widths

        if file_name:
            f = open(file_name,'w')
            f.write(str(mkl_machine)+'\n')
            f.close()
        else:
            return mkl_machine

    def load_mkl_model(self, file_name, model_type = 'regression'):
        """ This method receives a file name (if it is not in pwd, full path must be given) and a model type to
        be loaded {'regression', 'binary', 'multiclass'}. The loaded file must contain a t least a dictionary at
        its top. This dictionary must contain a key called 'model' whose value must be a dictionary, from which
        model parameters will be read. For example:
            {'key_0':value, 'key_1':value,..., 'model':{'family':'PolyKernel', 'bias':1.001,...}, key_n:value}
        Four objects are returned. The MKL model which is tuned to those parameters stored at the given file. A
        numpy array containing learned weights of a CombinedKernel. The widths corresponding to returned kernel
        weights and the kernel family. Be careful with the kernel family you are loading because widths no
        necessarily are it, but probably 'degrees', e.g. for the PolyKernel family.
        The Combined kernel must be instantiated outside this method, thereby loading to it corresponding
        weights and widths.
        """
        with open(file_name, 'r') as pointer:
            mkl_machine = eval(pointer.read())['learned_model']

        if model_type == 'regression':
            mkl = MKLRegression()           # A new two-class MKL object
        elif model_type == 'binary':
            mkl = MKLClassification()
        elif model_type == 'multiclass':
            mkl = MKLMulticlass()
        else:
            sys.stderr.write('ERROR: Unknown problem type in model loading.')
            exit()

        mkl.set_bias(mkl_machine['bias'])
        mkl.create_new_model(mkl_machine['num_support_vectors']) # Initialize the inner SVM
        for i in xrange(mkl_machine['num_support_vectors']):
            mkl.set_alpha(i, mkl_machine['support'][i][0])
            mkl.set_support_vector(i, mkl_machine['support'][i][1])
        mkl_machine['weights'] = numpy.array(mkl_machine['weights'])
        return mkl, mkl_machine

    # Getters (properties):
    @property
    def family_translation(self):
        """
        """
        self.__family_translation = {'PolyKernel':'polynomial', 'GaussianKernel':'gaussian',
                                     'ExponentialKernel':'exponential'}
        return self.__family_translation

    @property
    def mkl_model(self):
        """ This property stores the MKL model parameters learned by the self-object. These parameters can be
        stored into a file for future configuration of a non-trained MKL new MKL object. Also probably passed
        onwards for showing results.
        """
        return self.__mkl_model

    @property
    def estimated_out(self):
        """ This property is the mkl result after applying.
        """
        return self.__estimated_out

    @property
    def compoundKernel(self):
        """This method is used for getting the kernel object, i.e. the learned MKL object, which can be unwrapped
        into its matrix form instead of getting a Shogun object. Use the input parameters Matrix = True,
        expansion = False for getting the compound matrix of reals. For instance:
            mklObj.Matrix = True
            mklObj.expansion = False
            kernelMatrix = mklObj.compoundKernel
        Use Matrix = True, expansion = True for getting the expanded linear combination of matrices and weights
        separately, for instance:
            mklObj.Matrix = True
            mklObj.expansion = True
            basis, weights = mklObj.compoundKernel
        Use Matrix = False, expansion = False for getting the learned kernel Shogun object, for instance:
            mklObj.Matrix = False
            mklObj.expansion = False
            kernelObj = mklObj.compoundKernel

        .. warning:: Be careful with this latter variant of the method becuase of the large amount of needed
         physical memory.
        """

        if self.Matrx:
            kernels = []
            size = self.ker.get_num_subkernels()
            for k in xrange(0, size - 1):
                kernels.append(self.ker.get_kernel(k).get_kernel_matrix())
            ws = self.weights
            if self.expansion:
                return kernels, ws  # Returning the full expansion of the learned kernel.
            else:
                return sum(kernels * ws)  # Returning the matrix linear combination, in other words,
        else:  # a projector matrix representation.
            return self.ker  # If matrix representation is not required, only the Shogun kernel
            # object is returned.

    @property
    def sigmas(self):
        """This method is used for getting the current set of basis kernel parameters, i.e. widths, in the case
        of the gaussian basis kernel.
        :rtype : list of float
        """
        return self.__sigmas

    @property
    def verbose(self):
        """This is the verbose flag, which is used for monitoring the object training procedure.

        IMPORTANT: Don't use this feature (True) if you are working in pipe mode. The object will print undesired
        outputs to the stdout.

        :rtype : bool
        """
        return self._verbose

    @property
    def Matrx(self):
        """This is a boolean property of the object. Its aim is getting and, mainly, setting the kind of object
        we want to obtain as learned kernel, i.e. a Kernel Shogun object or a Kernel Matrix whose entries are
        reals. The latter could require large amounts of physical memory. See the mklObj.compoundKernel property
        documentation in this object for using details.
        :rtype :bool
        """
        return self.__Matrx

    @property
    def expansion(self):
        """This is a boolean property. Its aim is getting and, mainly, setting the mklObj object to return the
        complete expansion of the learned kernel, i.e. a list of basis kernel matrices as well as their
        corresponding coefficients. This configuration may require large amounts of physical memory. See the
        mklObj.compoundKernel property documentation in this object for using details.
        :rtype :bool
        .. seealso:: the code and examples and documentation about :@property:`compoundKernel`
        """
        return self.__expansion

    @property
    def weightRegNorm(self):
        """ The value of this property is the basis' weight vector norm, e.g. :math:`||\\beta||_p`, to be used as
        regularizer. It controls the smoothing among basis kernel weights of the learned multiple kernel combination. On
        one hand, If p=1 (the l_1 norm) the weight values B_i will be disproportionally between them, i.e. a few of them
        will be >> 0,some other simply > 0 and many of them will be zero or very near to zero (the vector B will be
        sparse). On the other hand, if p = 2 the weights B_i linearly distributed, i.e. their distribution shows an
        uniform tilt in such a way the differences between pairs of them are not significant, but rather proportional to
        the tilt of the distribution.

        To our knowledge, such a tilt is certainly not explicitly taken into account as regularization hyperparameter,
        although the parameter C \in [0, 1] is directly associated to it as scalar factor. Thus specifically for
        C \in [0, 1], it operates the vector B by forcing to it to certain orientation which describes a tilt
        m \in (0, 1)U(1, \infty) (with minima in the extremes of these subsets and maxima in their medians). Given that
        C \n [0, 1], the scaling effect behaves such that linearly depresses low values of B_i, whilst highlights their
        high values. The effect of C \in (1, \infty) is still not clearly studied, however it will be a bit different
        than the above, but keeping its scalar effect.

        Overall, as p tends to be >> 1 (or even p --> \\infty) the B_i values tend to be ever more uniformly
        distributed. More specific and complex regularization operators are explained in .. seealso:: Schölkopf, B., & Smola, A. J.
        (2002). Learning with kernels: Support vector machines, regularization, optimization, and beyond. MIT press.

        :rtype : vector of float
        """
        return self.__weightRegNorm

    # function getters
    @property
    def weights(self):
        """This method is used for getting the learned weights of the MKL object. We first get the kernel weights into
        a list object, before returning it. This is because 'get_subkernel_weights()' causes error while printing to an
        output file by means of returning a nonlist object.

        :rtype : list of float
        """
        self.__weights = list(self.ker.get_subkernel_weights())

        return self.__weights

    @property
    def SVMepsilon(self):
        """This method is used for getting the SVM convergence criterion (the minimum allowed error commited by
        the transducer in training).

        :rtype : float

        .. seealso:: See at page 22 of Sonnemburg et.al., (2006) Large Scale Multiple Kernel Learning.
        .. seealso:: @SVMepsilon.setter
        """
        return self.__SVMepsion

    @property
    def MKLepsilon(self):
        """This method is used for getting the MKL convergence criterion (the minimum allowed error committed by
        the MKL object in test).

        :rtype : float

        .. seealso:: See at page 22 of Sonnemburg et.al., (2006) Large Scale Multiple Kernel Learning.
        .. seealso:: @MKLepsilon.setter
        """
        return self.__MKLepsilon

    @property
    def mklC(self):
        """This method is used for setting regularization parameters. 'mklC' is a real value in multiclass problems,
        while in binary problems it must be a list of two elements. These must be different when the two classes are
        imbalanced, but must be equal for balanced densities in binary classification problems. For multiclass
        problems, imbalanced densities are not considered.

        :rtype : float

        .. seealso:: See at page 4 of Bagchi, (2014) SVM Classifiers Based On Imperfect Training Data.
        .. seealso:: @weightRegNorm property documentation for more details about C as regularization parameter.
        """
        return self.__mklC

    @property
    def threads(self):
        """ This property is used for getting and setting the number of threads in which the training procedure will be
        will be segmented into a single machine processor core.

        :rtype : int
        .. seealso:: @threads.setter documentation.
        """
        return self.__threads

    # Readonly properties:
    @property
    def problem(self):
        """This method is used for getting the kind of problem the mklObj object will be trained for. If binary == True,
        the you want to train the object for a two-class classification problem. Otherwise if binary == False, you want
        to train the object for multiclass classification problems. This property can't be modified once the object has
        been instantiated.

        :rtype : bool
        """
        return self.__problem

    @property
    def testerr(self):
        """This method is used for getting the test accuracy after training the MKL object. 'testerr' is a readonly
        object property.

        :rtype : float
        """
        return self.__testerr
    
    @property
    def sparse(self):
        """This method is used for getting the sparse/dense mode of the MKL object. 

        :rtype : float
        """
        return self.__sparse

    @property
    def crashed(self):
        """This method is used for getting the sparse/dense mode of the MKL object. 

        :rtype : float
        """
        return self.__crashed

    # mklObj (decorated) Setters: Binary configuration of the classifier cant be changed. It is needed to instantiate
    # a new mklObj object.

    @crashed.setter
    def crashed(self, value):

        assert isinstance(value, bool) # The model is not stored as a dictionary
        self.__crashed = value

    @mkl_model.setter
    def mkl_model(self, value):

        assert isinstance(value, dict) # The model is not stored as a dictionary
        self.__mkl_model = value

    @estimated_out.setter
    def estimated_out(self, value):

        self.__estimated_out = value
    
    @sparse.setter
    def sparse(self, value):

        self.__sparse = value

    @Matrx.setter
    def Matrx(self, value):
        """
        :type value: bool
        .. seealso:: @Matrx property documentation.
        """
        assert isinstance(value, bool)
        self.__Matrx = value

    @expansion.setter
    def expansion(self, value):
        """
        .. seealso:: @expansion property documentation
        :type value: bool
        """
        assert isinstance(value, bool)
        self.__expansion = value

    @sigmas.setter
    def sigmas(self, value):
        """ This method is used for setting desired basis kernel parameters for the MKL object. 'value' is a list of
        real values of 'pKers' length. In 'learning' mode, be careful to avoid mismatching between the number of basis kernels of the
        current compound kernel and the one you have in mind. A mismatch error could be arisen. In 'pattern_recognition'
        mode, this quantity is taken from the learned model, which is stored at disk.

        @type value: list of float
        .. seealso:: @sigmas property documentation
        """
        try:
            if self.mode == 'learning':
                if len(value) == self._pkers:
                    self.__sigmas = value
                else:
                    raise customException('Size of basis kernel parameter list mismatches the size of the combined\
                                       kernel. You can use len(CMKLobj.sigmas) to revise the mismatching.')
            elif self.mode == 'pattern_recognition':
                self.__sigmas = value
        except customException, (instance):
            print "Caught: " + instance.parameter

    @verbose.setter
    def verbose(self, value):
        """This method sets to True of False the verbose flag, which is used in turn for monitoring the object training
        procedure.

        @type value: bool
        """
        assert isinstance(value, bool)
        self._verbose = value

    @weightRegNorm.setter
    def weightRegNorm(self, value):
        """This method is used for changing the norm of the weight regularizer of the MKL object. Typically this
        changing is useful for retrain the model with other regularizer.

        @type value: float
        ..seealso:: @weightRegNorm property documentation.
        """
        assert (isinstance(value, float) and value >= 0.0)
        self.mkl.set_mkl_norm(value)
        self.__weightRegNorm = value

    @SVMepsilon.setter
    def SVMepsilon(self, value):
        """This method is used for setting the SVM convergence criterion (the minimum allowed error commited by
        the transducer in training). In other words, the low level of the learning process. The current basis
        kernel combination is tested as the SVM kernel. Regardless of each basis' weights.

        @type value: float
        .. seealso:: Page 22 of Sonnemburg et.al., (2006) Large Scale Multiple Kernel Learning.
        """
        assert (isinstance(value, float) and value >= 0.0)
        self.mkl.set_epsilon(value)
        self.__SVMepsion = value

    @MKLepsilon.setter
    def MKLepsilon(self, value):
        """This method is used for setting the MKL convergence criterion (the minimum allowed error committed by
        the MKL object in test). In other words, the high level of the learning process. The current basis
        kernel combination is tested as the SVM kernel. The basis' weights are tuned until 'MKLeps' is reached.

        @type value: float
        .. seealso:: Page 22 of Sonnemburg et.al., (2006) Large Scale Multiple Kernel Learning.
        """
        assert (isinstance(value, float) and value >= 0.0)
        self.mkl.set_mkl_epsilon(value)
        self.__MKLepsilon = value

    @mklC.setter
    def mklC(self, value):
        """This method is used for setting regularization parameters. These are different when the two classes
        are imbalanced and Equal for balanced densities in binary classification problems. For multiclass
        problems imbalanced densities are not considered, so uniquely the first argument is caught by the method.
        If one or both arguments are misplaced the default values are one both them.

        @type value: float (greater than zero. There exits the zero-norm, but it is not considered here.)
        .. seealso:: Page 4 of Bagchi,(2014) SVM Classifiers Based On Imperfect Training Data.
        """
        if self.__problem == 'binary' or self.__problem == 'regression':
            assert len(value) == 2
            assert (isinstance(value, (list, float)) and value[0] > 0.0 and value[1] > 0.0)
            self.mkl.set_C(value[0], value[1])
        elif self.__problem == 'multiclass':
            assert (isinstance(value, float) and value > 0.0)
            self.mkl.set_C(value)

        self.__mklC = value

    @threads.setter
    def threads(self, value):
        """This method is used for changing the number of threads we want to be running with a single machine core.
        These threads are not different parallel processes running in different machine cores.
        """
        assert (isinstance(value, int) and value > 0)
        self.mkl.parallel.set_num_threads(value)  # setting number of training threads
        self.__threads = value

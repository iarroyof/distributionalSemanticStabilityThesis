#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ignacio Arroyo-Fernandez'

from modshogun import *
from tools.load import LoadMatrix
import random
from math import sqrt
import numpy


# .. todo:: Specify and validate input and returning types for functions.
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
def generate_binToy():
    """:return: [RealFeatures(train_data),RealFeatures(train_data),BinaryLabels(train_labels),BinaryLabels(test_labels)]
    This method generates random 2D training and test data for binary classification. The labels are {-1, 1} vectors.
    """
    num = 30
    num_components = 4
    means = zeros((num_components, 2))
    means[0] = [-1, 1]
    means[1] = [2, -1.5]
    means[2] = [-1, -3]
    means[3] = [2, 1]

    covs = array([[1.0, 0.0], [0.0, 1.0]])

    gmm = GMM(num_components)
    [gmm.set_nth_mean(means[i], i) for i in range(num_components)]
    [gmm.set_nth_cov(covs, i) for i in range(num_components)]
    gmm.set_coef(array([1.0, 0.0, 0.0, 0.0]))
    xntr = array([gmm.sample() for i in xrange(num)]).T
    xnte = array([gmm.sample() for i in xrange(5000)]).T
    gmm.set_coef(array([0.0, 1.0, 0.0, 0.0]))
    xntr1 = array([gmm.sample() for i in xrange(num)]).T
    xnte1 = array([gmm.sample() for i in xrange(5000)]).T
    gmm.set_coef(array([0.0, 0.0, 1.0, 0.0]))
    xptr = array([gmm.sample() for i in xrange(num)]).T
    xpte = array([gmm.sample() for i in xrange(5000)]).T
    gmm.set_coef(array([0.0, 0.0, 0.0, 1.0]))
    xptr1 = array([gmm.sample() for i in xrange(num)]).T
    xpte1 = array([gmm.sample() for i in xrange(5000)]).T

    return (RealFeatures(concatenate((xntr, xntr1, xptr, xptr1), axis=1)),  # Train Data
            RealFeatures(concatenate((xnte, xnte1, xpte, xpte1), axis=1)),  # Test Data
            BinaryLabels(concatenate((-ones(2 * num), ones(2 * num)))),  # Train Labels
            BinaryLabels(concatenate((-ones(10000), ones(10000)))))  # Test Labels


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
                         ['linear'|
                          'quadratic'|
                          'log-gauss'|
                          'gaussian'|
                          'triangular'|
                          'pareto'|
                          'beta'|
                          'gamma'|
                          'weibull'].
    See http://pymotw.com/2/random/ for details about pseudorandom number generators in python. The input 'size' is the
    amount of segments the distribution domain will be discretized out. The 'rango' input are the minimum and maximum
    values of the obtained distributed values. The 'parameters' of these weight vector distributions are set to common
    values of each distribution by default, but they can be modified.

    :param hyperDistribution: string
    :param size: It is the number of basis kernels for the MKL object.
    :param rango: It is the range to which the basis kernel parameters will pertain. For some basis kernels families
    this input parameter has not effect.
    :param parameters: It is a list of parameters of the distribution of the random weights, e.g. for a gaussian
    distribution with mean zero and variance 1, parameters = [0, 1]. For some basis kernel families this input parameter
    has not effect.

    .. seealso: fit_kernel() function documentation.
    """
    # Validating inputs
    assert isinstance(size, int)
    assert (rango[0] < rango[1] and len(rango) == 2)
    # .. todo: Revise the other linespaces of the other distributions. They must be equally consistent than the
    # .. todo: Gaussian one. Change 'is' when verifying equality between strings (PEP008 recommendation).
    sig = []
    if hyperDistribution is 'linear':
        sig = random.sample(range(rango[0], rango[1]), size)
        return sig
    elif hyperDistribution is 'quadratic':
        sig = numpy.square(random.sample(numpy.linspace(sqrt(rango[0]), int(sqrt(rango[1]))), size))
        return sig
    elif hyperDistribution is 'gaussian':
        i = 0
        while i < size:
            numero = random.gauss(parameters[0], parameters[1])
            if numero > rango[0] and numero < rango[1]:  # Validate the initial point of
                sig.append(numero)  # 'range'. If not met, loop does
                i += 1  # not end, but resets
                # If met, the number is appended
        return sig  # to 'sig' width list.
    elif hyperDistribution == 'triangular':
        for i in xrange(size):
            sig.append(random.triangular(rango[0], rango[1]))
        return sig
    elif hyperDistribution == 'beta':
        i = 0
        while i < size:
            numero = random.betavariate(parameters[0], parameters[1]) * (rango[1] - rango[0]) + rango[0]
            sig.append(numero)
            i += 1
        return sig
    elif hyperDistribution == 'pareto':
        return numpy.random.pareto(5, size=size) * (rango[1] - rango[0]) + rango[0]

    elif hyperDistribution == 'gamma':
        return numpy.random.gamma(shape=1, size=size) * (rango[1] - rango[0]) + rango[0]

    elif hyperDistribution == 'weibull':
        return numpy.random.weibull(2, size=size) * (rango[1] - rango[0]) + rango[0]

    elif hyperDistribution == 'log-gauss':
        i = 0
        while i < size:
            numero = random.lognormvariate(parameters[0], parameters[1])
            if numero > rango[0] and numero < rango[1]:
                sig.append(numero)
                i += 1

        return sig
    else:
        print 'The entered hyperparameter distribution is not allowed.'


# Combining kernels
def genKer(self, featsL, featsR, basisFam, widths=[5, 4, 3, 2, 1]):
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

    kernels = []
    if basisFam == 'gaussian':
        for w in widths:
            kernels.append(GaussianKernel())
            kernels[len(kernels) - 1].set_width(w)

    elif basisFam == 'inverseQuadratic':  # For this (and others below) kernel it is necessary fitting the
        dst = MinkowskiMetric(l=featsL, r=featsR, k=2)  # distance matrix at this moment
        for w in widths:
            kernels.append(InverseMultiQuadricKernel(0, w, dst))

    elif basisFam == 'polynomial':
        for w in widths:
            kernels.append(PolyKernel(0, w, False))

    elif basisFam == 'power':  # At least for images, the used norm does not make differences in performace
        dst = MinkowskiMetric(l=featsL, r=featsR, k=2)
        for w in widths:
            kernels.append(PowerKernel(0, w, dst))

    elif basisFam == 'rationalQuadratic':  # At least for images, using 3-norm  make differences
        dst = MinkowskiMetric(l=featsL, r=featsR, k=2)  # in performace
        for w in widths:
            kernels.append(RationalQuadraticKernel(0, w, dst))

    elif basisFam == 'spherical':  # At least for images, the used norm does not make differences in performace
        dst = MinkowskiMetric(l=featsL, r=featsR, k=2)
        for w in widths:
            kernels.append(SphericalKernel(0, w, dst))

    elif basisFam == 'tstudent':  # At least for images, the used norm does not make differences in performace
        dst = MinkowskiMetric(l=featsL, r=featsR, k=2)
        for w in widths:
            kernels.append(TStudentKernel(0, w, dst))

    elif basisFam == 'wave':  # At least for images, the used norm does not make differences in performace
        dst = MinkowskiMetric(l=featsL, r=featsR, k=2)
        for w in widths:
            kernels.append(WaveKernel(0, w, dst))

    elif basisFam == 'wavelet':  # At least for images it is very low the performance with this kernel.
        for w in widths:  # It remains pending, for now, analysing its parameters.
            kernels.append(WaveletKernel(0, w, 0))

    elif basisFam == 'cauchy':
        dst = MinkowskiMetric(l=featsL, r=featsR, k=2)
        for w in widths:
            kernels.append(CauchyKernel(0, w, dst))

    elif basisFam == 'exponential':  # For this kernel it is necessary specifying features at the constructor
        dst = MinkowskiMetric(l=featsL, r=featsR, k=2)
        for w in widths:
            kernels.append(ExponentialKernel(featsL, featsR, w, dst, 0))

    elif basisFam == 'anova':  # This kernel presents a warning in training:
        """RuntimeWarning: [WARN] In file /home/iarroyof/shogun/src/shogun/classifier/mkl/MKLMulticlass.cpp line
           198: CMKLMulticlass::evaluatefinishcriterion(...): deltanew<=0.Switching back to weight norsm
           difference as criterion.
        """
        for w in widths:
            kernels.append(ANOVAKernel(0, w))

    else:
        raise NameError('Unknown Kernel family name!!!')

    combKer = CombinedKernel()
    for k in kernels:
        combKer.append_kernel(k)

    return combKer


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
                        binary = False,
                        verbose = False)

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

    def __init__(self, weightRegNorm=2.0, mklC=2.0, SVMepsilon=1e-5,
                 threads=2, MKLepsilon=0.001, binary=False, verbose=False):
        """Object initialization. This procedure is regardless of the input data, basis kernels and corresponding
        hyperparameters (kernel fitting).
        """
        self.__binary = binary
        if self.__binary:
            self.mkl = MKLClassification()  # MKL object (Binary)
        else:  # two classes are imbalanced. Equal for balanced densities
            self.mkl = MKLMulticlass()  # MKL object (Multiclass).

        self.mklC = mklC  # Setting regularization parameter. These are different when the
        self.weightRegNorm = weightRegNorm  # Setting the basis' weight vector norm
        self.SVMepsilon = SVMepsilon  # setting the transducer stop (convergence) criterion
        self.MKLepsilon = MKLepsilon  # setting the MKL stop criterion. The value suggested by
        # Shogun examples is 0.001. See setter docs for details
        self.threads = threads  # setting number of training threads. Verify functionality!!
        self.verbose = verbose  # inner training process verbose flag
        self.Matrx = False  # Kind of returned learned kernel object. See getter documentation of these
        self.expansion = False  # object configuration parameters for details. Only modifiable by setter.
        self.__testerr = 0

    # Self Function for kernel generation

    def fit_kernel(self, featsTr, targetsTr, featsTs, targetsTs, randomRange=[1, 50], randomParams=[1, 1],
                   hyper='linear', kernelFamily='guassian', pKers=3):
        """ :return: CombinedKernel Shogun object.
        This method is used for training the desired compound kernel. See documentation of the 'mklObj'
        object for using example. 'featsTr' and 'featsTs' are the training and test data respectively. 'targetsTr'
        and 'targetsTs' are the training and test labels, respectively. All they must be Shogun 'RealFeatures'
        and 'MulticlassLabels' objects respectively.
        The 'randomRange' parameter defines the range of numbers from which the basis kernel parameters will be
        drawn, e.g. Gaussian random widths between 1 and 50 (the default). The 'randomParams' input parameter
        states the parameters of the pseudorandom distribution of the basis kernel parameters to be drawn, e.g.
        Gaussian-pseudorandom-generated weights with std. deviation equal to 1 and mean equal to 1 (the default).
        The 'hyper' input parameter defines the distribution of the pseudorandom-generated weights. See
        documentation of the sigmaGen() method of the 'mklObj' object to see a list of possible basis kernel
        parameter distributions. The 'kernelFamily' input parameter is the basis kernel family to be append to the
        desired compound kernel if you select, e.g., the default 'gaussian' family, all elements of the learned
        linear combination will be gaussians (each differently weighted and parametrized). See documentation of
        the genKer() method of the 'mklObj' object to see a list of allowed basis kernel families. The 'pKers'
        input parameter defines the size of the learned kernel linear combination, i.e. how many basis kernels to
        be weighted in the training and therefore, how many coefficients will have the Fourier series of data (the
        default is 3).

        .. note:: In the cases of kernelFamily = {'polynomial' or 'power' or 'tstudent' or 'anova'}, the input
        parameters {'randomRange', 'randomParams', 'hyper'} have not effect, because these kernel families do not
        require basis kernel parameters.

        :param featsTr: RealFeatures Shogun object conflating the training data.
        :param targetsTr: MulticlassLabels Shogun object conflating the training labels.
        :param featsTr: RealFeatures Shogun object conflating the test data.
        :param targetsTr: MulticlassLabels Shogun object conflating the test labels.
        :param randomRange: It is the range to which the basis kernel parameters will pertain. For some basis kernels
        families this input parameter has not effect.
        :param randomParams: It is a list of parameters of the distribution of the random weights, e.g. for a gaussian
        distribution with mean zero and variance 1, parameters = [0, 1]. For some basis kernel families this input
        parameter has not effect.
        :param hyper: string which specifies the name of the basis kernel parameter distribution. See documentation for
        sigmaGen() function for viewing allowed strings (names).
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
                '\nWeight regularization norm: ', self.weightRegNorm
            if not self.__binary:
                print "Classes: ", targetsTr.get_num_classes()
            else:
                print "Classes: Binary"

            # Generating the list of subkernels. Creating the compound kernel
            # For monomial-nonhomogeneous (polynomial) kernels the hyperparameters are uniquely the degree of each monomial
            # in the form of a sequence. MKL finds the coefficient for each monomial in order to find a compound polynomial.
        if kernelFamily == 'polynomial' or kernelFamily == 'power' or \
                        kernelFamily == 'tstudent' or kernelFamily == 'anova':
            self.sigmas = list(range(0, pKers))
            self.ker = genKer(self, self._featsTr, self._featsTr, basisFam=kernelFamily, widths=self.sigmas)
        else:
            # We have called 'sigmas' to any basis kernel parameter, regardless if it is Gaussian or not. So generate the widths:
            self.sigmas = sorted(sigmaGen(self, hyperDistribution=hyper, size=pKers,
                                          rango=randomRange, parameters=randomParams))
            try:  # Verifying if number of kernels is greater or equal to 2
                if pKers <= 1 or len(self.sigmas) < 2:
                    raise customException('Senseless MKLClassification use!!!')
            except customException, (instance):
                print 'Caugth: ' + instance.parameter
                print "-----------------------------------------------------"
                print """The multikernel learning object is meaningless for less than 2 basis
                     kernels, i.e. pKers <= 1, so 'mklObj' couldn't be instantiated."""
                print "-----------------------------------------------------"

            self.ker = genKer(self, self._featsTr, self._featsTr, basisFam=kernelFamily, widths=self.sigmas)
            if self.verbose:
                print 'Widths: ', self.sigmas
        # Initializing the compound kernel
        self.ker.init(self._featsTr, self._featsTr)
        try:  # Verifying if number of  kernels was greater or equal to 2 after training
            if self.ker.get_num_subkernels() < 2:
                raise customException(
                    'Multikernel coeffients were less than 2 after training. Revise object settings!!!')
        except customException, (instance):
            print 'Caugth: ' + instance.parameter

        # Verbose for learning surveying
        if self.verbose:
            print '\nKernel fitted...'
        # Initializing the transducer for multiclassification
        self.mkl.set_kernel(self.ker)
        self.mkl.set_labels(self._targetsTr)
        # Train to return the learnt kernel
        if self.verbose:
            print '\nLearning the machine coefficients...'

        self.mkl.train()

        if self.verbose:
            print 'Kernel trained... Weights: ', self.weights
        # Evaluate the learnt Kernel. Here it is asumed 'ker' is learnt, so we only need for initialize it again but with
        # the test set object. Then, set the initialized kernel to the mkl object in order to 'apply'.
        self.ker.init(self._featsTr, featsTs)  # Now with test examples. The inner product between training
        self.mkl.set_kernel(self.ker)  # and test examples generates the corresponding Gramm Matrix.
        out = self.mkl.apply()  # Applying the obtained Gramm Matrix

        if self.__binary:  # If the problem is either binary or multiclass, different
            evalua = ErrorRateMeasure()  # performance measures are computed.
        else:
            evalua = MulticlassAccuracy()

        if self.__binary:
            self.__testerr = 100 - evalua.evaluate(out, targetsTs) * 100
        else:
            self.__testerr = evalua.evaluate(out, targetsTs) * 100
        # Verbose for learning surveying
        if self.verbose:
            print 'Kernel evaluation ready. The precision was: ', self.__testerr, '%'

    def save_sigmas(self, file='sigmasFile.txt', mode='w', note='Some note'):
        """This method saves the set of kernel parameters (e.g. gaussian widths) into a text file, which are
        associated to a basis kernel family. It could be used for loading a desired set of widths used in previous
        training epochs (i.e. when the F1-measure showed a maximum).
        By default, '../sigmasFile.txt' will be the corresponding directory and file name. You can set the mode of
        the file object, e.g. to 'a' for uniquely adding content. If you want adding some note to each saved sigma
        array, it could be used the 'note' input string.
        """
        f = open(file, mode)
        f.write('# ----------------- ' + note + ' ------------------')
        f.write("\n# Basis kernel family: " + self.basisFamily + '\n')
        for s in self.sigmas:
            f.write("%s, " % s)
        f.close()

    # Multi-kernel object training procedure file reporting.
    def filePrintingResults(self, fileName, mode):
        """This method is used for printing training results as well as used settings for each learned compounding
        kernel into a file for comparison. 'fileName' is the desired location of the file at your HD and 'mode' could
        be setted to 'a' for adding different training results to the same file. The default mode is 'w', which is
        used for creating or rewriting a file.
        """
        f = open(fileName, mode)
        if mode == 'w':
            f.write('                   Results\
            \n--------------------------------------------------------------------\n')
        else:
            f.write('\n--------------------------------------------------------------------\n')

        f.write("Basis kernel family: " + self.basisFamily)
        f.write("\nLinear combination size: " + str(self._pkers))
        f.write('\nHyperparameter distribution: ' + str(self._hyper))
        f.write('\nWeight regularization norm: ' + str(self.weightRegNorm))
        f.write('\nWeight regularization parameter: ' + str(self.mklC))
        f.write("\nWeights: ")
        ws = self.weights
        for item in ws:
            f.write("%s, " % item)
        f.write('\nWidths: ')
        for item in self.sigmas:
            f.write("%s, " % item)
        if self.__binary:
            f.write('\nTest error: ' + str(100 - self.__testerr * 100))
        else:
            f.write("\nClasses: " + str(self._targetsTr.get_num_classes()))
            f.write('\nTest error:' + str(self.__testerr * 100))
        f.close()

    # It is pending defining functions for run time attribute modification, e.g. set_verbose(), set_regPar(), etc. unwrapped

    # Getters (properties):
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

        .. warning:: Be careful with this latter variant of the method becuase of the large amount of needed physical
        memory.
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
        """This method is used for getting the current set of basis kernel parameters, i.e. widths, in the case of
        the gaussian basis kernel.
        :rtype : list of float
        """
        return self.__sigmas

    @property
    def verbose(self):
        """This is the verbose flag, which is used for monitoring the object training procedure.
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
        """This method is used for getting the learned weights of the MKL object.

        :rtype : list of float
        """
        self.__weights = self.ker.get_subkernel_weights()
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
    def binary(self):
        """This method is used for getting the kind of problem the mklObj object will be trained for. If binary == True,
        the you want to train the object for a two-class classification problem. Otherwise if binary == False, you want
        to train the object for multiclass classification problems. This property can't be modified once the object has
        been instantiated.

        :rtype : bool
        """
        return self.__binary

    @property
    def testerr(self):
        """This method is used for getting the test accuracy after training the MKL object. 'testerr' is a readonly
        object property.

        :rtype : float
        """
        return self.__testerr

    # mklObj (decorated) Setters: Binary configuration of the classifier cant be changed. It is needed to instantiate
    # a new mklObj object.
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
        """ This method is used for setting desired basis kernel parameters for the MKL object. 'value'
        is a list of real values of 'pKers' length. Be careful to avoid mismatching between the number of basis kernels
        of the current compound kernel and the one you have in mind. A mismatch error could be arisen.

        @type value: list of float
        .. seealso:: @sigmas property documentation
        """
        try:
            if len(value) == self._pkers and min(value) > 0:
                self.__sigmas = value
            else:
                raise customException('Size of basis kernel parameter list missmatches the size of the combined\
                                       kernel. You can use len(CMKLobj.get_sigmas()) to revise the missmatching.')
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

        @type value: float
        .. seealso:: Page 4 of Bagchi,(2014) SVM Classifiers Based On Imperfect Training Data.
        """
        if self.__binary:
            assert len(value) == 2
            assert (isinstance(value, (list, float)) and value[0] >= 0.0 and value[1] >= 0.0)
            self.mkl.set_C(value[0], value[1])
        else:
            assert (isinstance(value, float) and value >= 0.0)
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ignacio Arroyo-Fernandez'

from modshogun import *
from tools.load import LoadMatrix
import numpy

# Loading toy data from files
def load_Toy(dataRoute, fileTrain, fileLabels):
    lm = LoadMatrix()
    dataSet = lm.load_numbers(dataRoute + fileTrain)
    labels = lm.load_labels(dataRoute + fileLabels)

    return (RealFeatures(dataSet.T[0:3*len(dataSet.T)/4].T),    # Return the training set, 3/4 * dataSet
            RealFeatures(dataSet.T[(3*len(dataSet.T)/4):].T),   # Return the test set, 1/4 * dataSet
            MulticlassLabels(labels[0:3*len(labels)/4]),        # Return corresponding train and test labels
            MulticlassLabels(labels[(3*len(labels)/4):]))

# 2D Toy data generator
def generate_binToy():
    num=30;
    num_components=4
    means=zeros((num_components, 2))
    means[0]=[-1,1]
    means[1]=[2,-1.5]
    means[2]=[-1,-3]
    means[3]=[2,1]

    covs=array([[1.0,0.0],[0.0,1.0]])

    gmm=GMM(num_components)
    [gmm.set_nth_mean(means[i], i) for i in range(num_components)]
    [gmm.set_nth_cov(covs,i) for i in range(num_components)]
    gmm.set_coef(array([1.0,0.0,0.0,0.0]))
    xntr=array([gmm.sample() for i in xrange(num)]).T
    xnte=array([gmm.sample() for i in xrange(5000)]).T
    gmm.set_coef(array([0.0,1.0,0.0,0.0]))
    xntr1=array([gmm.sample() for i in xrange(num)]).T
    xnte1=array([gmm.sample() for i in xrange(5000)]).T
    gmm.set_coef(array([0.0,0.0,1.0,0.0]))
    xptr=array([gmm.sample() for i in xrange(num)]).T
    xpte=array([gmm.sample() for i in xrange(5000)]).T
    gmm.set_coef(array([0.0,0.0,0.0,1.0]))
    xptr1=array([gmm.sample() for i in xrange(num)]).T
    xpte1=array([gmm.sample() for i in xrange(5000)]).T

    return (RealFeatures(concatenate((xntr,xntr1,xptr,xptr1), axis=1)),     #Train Data
            RealFeatures(concatenate((xnte,xnte1,xpte,xpte1), axis=1)),     #Test Data
            BinaryLabels(concatenate((-ones(2*num), ones(2*num)))),         #Train Labels
            BinaryLabels(concatenate((-ones(10000), ones(10000)))) )        #Test Labels

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
    """ This module generates the pseudorandom vector of widths for basis Gaussian kernels according to a distribution,
    i.e.:
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
    """
    sig = []
    if hyperDistribution == 'linear':
        sig = random.sample(range(rango[0], rango[1]), size)
        return sig
    elif hyperDistribution == 'quadratic':
        sig = random.sample(range(rango[0],int(sqrt(rango[1]))), size)
        return numpy.array(sig)**2
    elif hyperDistribution == 'gaussian':
        i = 0
        while i < size:
            numero = random.gauss(parameters[0], parameters[1])
            if numero > rango[0] and numero < rango [1]:  # Validate the initial point of
                sig.append(numero)                        # 'range'. If not met, loop does
                i += 1                                    # not end, but resets
                                                          # If met, the number is appended
        return sig                                        # to 'sig' width list.
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
        return numpy.random.pareto(5, size = size) * (rango[1] - rango[0]) + rango[0]

    elif hyperDistribution == 'gamma':
        return numpy.random.gamma(shape = 1, size = size) * (rango[1] - rango[0]) + rango[0]

    elif hyperDistribution == 'weibull':
         return numpy.random.weibull(2, size = size) * (rango[1] - rango[0]) + rango[0]

    elif hyperDistribution == 'log-gauss':
        i = 0
        while i < size:
            numero = random.lognormvariate(parameters[0], parameters[1])
            if numero > rango[0] and numero < rango [1]:
                sig.append(numero)
                i += 1

        return sig
    else:
        print 'The entered hyperparameter distribution is not allowed.'

# Combining kernels
def genKer(self, featsL, featsR, basisFam, widths = [5,4,3,2,1]):
    """This module generates a list of basis kernels. These kernels are
    tuned according to the vector ''widths''. Input parameters ''featsL''
    and ''featsR'' are Shogun feature objects. In the case of a learnt RKHS,
    these both objects should be derived from the training SLM vectors,
    by means of the Shogun constructor realFeatures().
    This module also appends basis kernels to a Shogun combinedKernel object.
    The kernels to be append are left in ''combKer'' (see code) which is
    returned. We have analyzed some basis families available in Shogun,
    so possible values of 'basisFam' are:

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
            kernels[len(kernels)-1].set_width(w)

    elif basisFam == 'inverseQuadratic': # For this (and others below) kernel it is necessary fitting the
        dst = MinkowskiMetric (l = featsL, r = featsR, k = 2) # distance matrix at this moment
        for w in widths:
            kernels.append(InverseMultiQuadricKernel(0, w, dst))

    elif basisFam == 'polynomial':
        for w in widths:
            kernels.append(PolyKernel(0, w, False))

    elif basisFam == 'power': # At least for images, the used norm does not make differences in performace
        dst = MinkowskiMetric (l = featsL, r = featsR, k = 2)
        for w in widths:
            kernels.append(PowerKernel(0, w, dst))

    elif basisFam == 'rationalQuadratic': # At least for images, using 3-norm  make differences
        dst = MinkowskiMetric (l = featsL, r = featsR, k = 2) # in performace
        for w in widths:
            kernels.append(RationalQuadraticKernel(0, w, dst))

    elif basisFam == 'spherical': # At least for images, the used norm does not make differences in performace
        dst = MinkowskiMetric (l = featsL, r = featsR, k = 2)
        for w in widths:
            kernels.append(SphericalKernel(0, w, dst))

    elif basisFam == 'tstudent': # At least for images, the used norm does not make differences in performace
        dst = MinkowskiMetric (l = featsL, r = featsR, k = 2)
        for w in widths:
            kernels.append(TStudentKernel(0, w, dst))

    elif basisFam == 'wave': # At least for images, the used norm does not make differences in performace
        dst = MinkowskiMetric (l = featsL, r = featsR, k = 2)
        for w in widths:
            kernels.append(WaveKernel(0, w, dst))

    elif basisFam == 'wavelet': # At least for images it is very low the performance with this kernel.
        for w in widths:        # It remains pending, for now, analysing its parameters.
            kernels.append(WaveletKernel(0, w, 0))

    elif basisFam == 'cauchy':
        dst = MinkowskiMetric (l = featsL, r = featsR, k = 2)
        for w in widths:
            kernels.append(CauchyKernel(0, w, dst))

    elif basisFam == 'exponential': # For this kernel it is necessary specifying features at the constructor
        dst = MinkowskiMetric (l = featsL, r = featsR, k = 2)
        for w in widths:
            kernels.append(ExponentialKernel(featsL, featsR, w, dst, 0))

    elif basisFam == 'anova': # This kernel presents a warning in training:
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
class mklObj (object):
    """Default self definition of the Multiple Kernel Learning object. This object uses previously
     defined methods for generating a linear combination of basis kernels that can be constituted from
     different families. See at fit_kernel() function documentation for details. This function trains
     the kernel weights. The object has other member functions offering utilities. See the next
     instantiation and using example:

        import mkl01 as mk

        kernel = mk.mklObj(weightRegNorm = 2,
                        regPar = 2,
                        svm_epsilon = 1e-5,
                        threads = 2,
                        mkl_epsilon = 0.001,
                        binary = False,
                        verbose = False)
    The above values are the defaults, so if they are suitable for you it is possible instantiating
    the object by simply stating: kernel = mk.mklObj(). Even it is possible modifying a subset of
    input parameters (keeping others as default): kernel = mk.mklObj(weightRegNorm = 1, regPar = 10,
    svm_epsilon = 1e-2). See the documentation of each setter below for allowed setting parameters
    without new instantiations.

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
    def __init__(self, weightRegNorm = 2, mklC = 2, SVMepsilon = 1e-5,
                 threads = 2, MKLepsilon = 0.001, binary = False, verbose = False):
        """Object initialization. This procedure is regardless of the input data, basis kernels and
        corresponding hyperparameters (kernel fitting)."""
        self.binary = binary
        if self.binary:
            self.mkl = MKLClassification()  # MKL object (Binary)
        else:                               # two classes are imbalanced. Equal for balanced densities
            self.mkl = MKLMulticlass()      # MKL object (Multiclass).

        self.mklC = mklC                    # Setting regularization parameter. These are different when the
        self.weightRegNorm = weightRegNorm  # Setting the basis' weight vector norm
        self.SVMepsilon = SVMepsilon        # setting the transducer stop (convergence) criterion
        self.MKLepsilon = MKLepsilon        # setting the MKL stop criterion. The value suggested by
                                            # Shogun examples is 0.001. See setter docs for details
        self.threads =threads 	            # setting number of training threads. Verify functionality!!
        self.verbose = verbose              # inner training process verbose flag
        self.Matrx = False                  # Kind of returned learned kernel object. See getter documentation of these
        self.expansion = False              # object configuration parameters for details. Only modifiable by setter.
# All obtained objects below become class attributes, so they are available any moment.
# Self Function for kernel generation

    def fit_kernel(self, featsTr,  targetsTr, featsTs, targetsTs, randomRange = [1, 50], randomParams = [1, 1],
                   hyper = 'linear', kernelFamily = 'guassian', pKers = 3):
        """ This method is used for training the desired compound kernel. See documentation of the 'mklObj'
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
        Note: In the cases of kernelFamily = {'polynomial' or 'power' or 'tstudent' or 'anova'}, the input
        parameters {'randomRange', 'randomParams', 'hyper'} have not effect, because these kernel families do not
        require basis kernel parameters.
        """
        # Inner variable copying:
        self._featsTr = featsTr
        self._targetsTr = targetsTr
        self._hyper = hyper
        self._pkers = pKers
        self.basisFamily = kernelFamily
        if self.verbose:	# Printing the training progress
            print '\nNacho, multiple <' + kernelFamily + '> Kernels have been initialized...'
            print "\nInput main parameters: "
            print "Hyperarameter distribution: ", self._hyper, "\nLinear combination size: ", pKers
            if not self.binary:
                print "Classes: ", targetsTr.get_num_classes()
            else:
                print "Classes: Binary"

# Generating the list of subkernels. Creating the compound kernel
# For monomial-nonhomogeneous (polynomial) kernels the hyperparameters are uniquely the degree of each monomial
# in the form of a sequence. MKL finds the coefficient for each monomial in order to find a compound polynomial.
        if kernelFamily == 'polynomial' or kernelFamily == 'power' or \
                        kernelFamily == 'tstudent' or kernelFamily ==  'anova':
            self.sigmas = list(range(0,pKers))
            self.ker = genKer(self, self._featsTr, self._featsTr, basisFam = kernelFamily, widths = self.sigmas)
        else:
# We have called 'sigmas' to any kernel parameter, regardless if it is Gaussian or not. So generate the widths:
            self.sigmas = sorted(sigmaGen(self, hyperDistribution = hyper, size = pKers,
                               rango = randomRange, parameters = randomParams)) #; pdb.set_trace()
            try: # Verifying if number of kernels is greater or equal to 2
                if pKers <= 1 or len(self.sigmas) < 2:
                    raise customException('Senseless MKLClassification use!!!')
            except customException, (instance):
                print 'Caugth: ' + instance.parameter
                print "-----------------------------------------------------"
                print """The multikernel learning object is meaningless for less than 2 basis
                     kernels, i.e. pKers <= 1, so 'mklObj' couldn't be instantiated."""
                print "-----------------------------------------------------"

            self.ker = genKer(self, self._featsTr, self._featsTr, basisFam = kernelFamily, widths = self.sigmas)
            if self.verbose:
                print self.weights
        # Initializing the compound kernel
        self.ker.init(self._featsTr, self._featsTr)
        try: # Verifying if number of  kernels was greater or equal to 2 after training
            if self.ker.get_num_subkernels() < 2:
                raise customException('Multikernel coeffients were less than 2 after training. Revise object settings!!!')
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
        self.ker.init(self._featsTr, featsTs)	# Now with test examples. The inner product between training
        self.mkl.set_kernel(self.ker)		    # and test examples generates the corresponding Gramm Matrix.
        out = self.mkl.apply()			        # Applying the obtained Gramm Matrix

        if self.binary:			    # If the problem is either binary or multiclass, different
            evalua = ErrorRateMeasure()	# performance measures are computed.
        else:
            evalua = MulticlassAccuracy()

        if self.binary:
            self.testerr = 100-evalua.evaluate(out, targetsTs)*100
        else:
            self.testerr = evalua.evaluate(out, targetsTs)*100
# Verbose for learning surveying
        if self.verbose:
            print 'Kernel evaluation ready. The precision was: ', self.testerr, '%'

    def save_sigmas(self, file = 'sigmasFile.txt', mode = 'w', note = 'Some note'):
        """This method saves the set of kernel parameters (e.g. gaussian widths) into a text file, which are
        associated to a basis kernel family. It could be used for loading a desired set of widths used in previous
        training epochs (i.e. when the F1-measure showed a maximum).
        By default, '../sigmasFile.txt' will be the corresponding directory and file name. You can set the mode of
        the file object, e.g. to 'a' for uniquely adding content. If you want adding some note to each saved sigma
        array, it could be used the 'note' input string.
        """
        f = open(file, mode)
        f.write('# ----------------- '+ note +' ------------------')
        f.write("\n# Basis kernel family: " + self.basisFamily + '\n')
        for s in self.sigmas:
            f.write("%s, " % s)
        f.close()

# Multi-kernel object training procedure file reporting.
    def filePrintingResults (self, fileName, mode):
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
        f.write('\nHyperparameter distribution: ' + str(self._hyper) )
        f.write('\nWeight regularization norm: ' + str(self.weightRegNorm))
        f.write('\nWeight regularization parameter: ' + str(self.mklC))
        f.write("\nWeights: ")
        ws = self.weights
        for item in ws:
            f.write("%s, " % item)
        f.write('\nWidths: ')
        for item in self.sigmas:
            f.write("%s, " % item)
        if self.binary:
            f.write('\nTest error: ' + str(100 - self.testerr*100) )
        else:
            f.write("\nClasses: " + str(self._targetsTr.get_num_classes()) )
            f.write('\nTest error:' + str( self.testerr*100 ))
        f.close()
# It is pending defining functions for run time attribute modification, e.g. set_verbose(), set_regPar(),
# etc. unwrapped

# Getters (properties):
    @property
    def compoundKernel (self):
        '''This method is used for getting the kernel object, i.e. the learned MKL object, which can be unwrapped
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
        Be careful with this latter variant of the method becuase of the large amount of needed physical memory.
        '''

        if self.Matrix:
            kernels = []
            size = self.ker.get_num_subkernels()
            for k in xrange(0, size-1):
                kernels.append(self.ker.get_kernel(k).get_kernel_matrix())
            ws = self.weights
            if self.expansion:
                return kernels, ws #Returning the full expansion of the learned kernel.
            else:
                return sum(kernels*ws) # Returning the matrix linear combination, in other words,
        else:                               # a projector matrix representation.
            return self.ker                 # If matrix representation is not required, only the Shogun kernel
                                            # object is returned.

    @property
    def testerr (self):
        """This method is used for getting the test accuracy after training the MKL object.
        """
        return self.__testerr

    @property
    def sigmas (self):
        '''This method is used for getting the current set of basis kernel parameters.
        '''
        return self.__sigmas

    @property
    def verbose(self):
        """This method sets to True of False the verbose flag, which is used for monitoring the
        object training procedure.
        """
        return self._verbose

    @property
    def Matrx (self):
        '''This is a boolean property of the object. Its aim is getting and, mainly, setting the kind of object
        we want to obtain as learned kernel, i.e. a Kernel Shogun object or a Kernel Matrix whose entries are
        reals. The latter could require large amounts of physical memory. See the mklObj.compoundKernel property
        documentation in this object for using details.'''
        return self.__Matrx

    @property
    def expansion(self):
        '''This is a boolean property. Its aim is getting and, mainly, setting the mklObj object to return the
        complete expansion of the learned kernel, i.e. a list of basis kernel matrices as well as their
        corresponding coefficients. This configuration may require large amounts of physical memory. See the
        mklObj.compoundKernel property documentation in this object for using details.'''
        return self.__expansion

    @property
    def weightRegNorm(self):

        return self.__weightRegNorm
# function getters
    @property
    def weights(self):
        """This method is used for getting the learned weights of the MKL object.
        """
        self.__weights = self.ker.get_subkernel_weights()
        return self.__weights

    @property
    def SVMepsilon(self):
        """This method is used for setting the SVM convergence criterion (the minimum allowed error commited by
        the transducer in training). In other words, the low level of the learning process. The current basis
        kernel combination is tested as the SVM kernel. Regardless of each basis' weights. See at page 22 of
        Sonnemburg et.al., (2006) Large Scale Multiple Kernel Learning.
        """
        return self.__SVMepsion

    @property
    def MKLepsilon(self):
        """This method is used for setting the MKL convergence criterion (the minimum allowed error committed by
        the MKL object in test). In other words, the high level of the learning process. The current basis
        kernel combination is tested as the SVM kernel. The basis' weights are tuned until 'MKLeps' is reached.
        See at page 22 of Sonnemburg et.al., (2006) Large Scale Multiple Kernel Learning.
        """
        return self.__MKLepsilon

    @property
    def mklC(self):
        """This method is used for setting regularization parameters. These are different when the two classes
        are imbalanced and Equal for balanced densities in binary classification problems. For multiclass
        problems imbalanced densities are not considered, so uniquely the first argument is caught by the method.
        If one or both arguments are misplaced the default values are one both them. See at page 4 of Bagchi,
        (2014) SVM Classifiers Based On Imperfect Training Data.
        """
        return self.__mklC

    @property
    def binary(self):
        """This method is used for setting regularization parameters. These are different when the two classes
        are imbalanced and Equal for balanced densities in binary classification problems. For multiclass
        problems imbalanced densities are not considered, so uniquely the first argument is caught by the method.
        If one or both arguments are misplaced the default values are one both them. See at page 4 of Bagchi,
        (2014) SVM Classifiers Based On Imperfect Training Data.
        """
        return self.__binary

    @property
    def threads(self):
        return self.__threads

# mklObj (decorated) Setters: Binary configuration of the classifier cant be changed. It is needed to instantiate
# a new mklObj object.
    @Matrx.setter
    def Matrx(self, value):
        """ :type value: bool
        """
        self.__Matrx = value

    @expansion.setter
    def expansion(self, value):
        """ :type value: bool
        """
        self.__expansion = value

    @sigmas.setter
    def sigmas(self, value):
        """ @type value: list This method is used for setting desired basis kernel parameters for the MKL object. 'value'
        is a list of real values of 'pKers' length. Be careful to avoid mismatching between the number of basis kernels
        of the current compound kernel and the one you have in mind. A mismatch error could be arisen.
        """
        try:
            if len(value) == self._pkers:
                self.__sigmas = value
            else:
                raise customException('Size of basis kernel parameter list missmatches the size of the combined\
                                       kernel. You can use len(CMKLobj.get_sigmas()) to revise the missmatching.')
        except customException, (instance):
            print "Caught: " + instance.parameter

    @verbose.setter
    def verbose(self, value):
        """This method sets to True of False the verbose flag, which is used for monitoring the
        object training procedure.
        @type value: bool
        @type self: object
        """
        self._verbose = value

    @binary.setter
    def binary(self, value):
        """This method sets to True of False the verbose flag, which is used for monitoring the
        object training procedure.
        @type value: bool
        @type self: object
        """
        self.__binary = value
# mkl object setters

    @testerr.setter
    def testerr(self, value):
        assert isinstance(value, float)
        self.__testerr = value

    @weightRegNorm.setter
    def weightRegNorm(self, value):
        """This method is used for changing the norm of the weight regularizer of the MKL object. Tipically this
        change is useful for retrain the model with other regularizer.
        """
        self.mkl.set_mkl_norm(value)
        self.__weightRegNorm = value

    @SVMepsilon.setter
    def SVMepsilon(self, value):
        """This method is used for setting the SVM convergence criterion (the minimum allowed error commited by
        the transducer in training). In other words, the low level of the learning process. The current basis
        kernel combination is tested as the SVM kernel. Regardless of each basis' weights. See at page 22 of
        Sonnemburg et.al., (2006) Large Scale Multiple Kernel Learning.
        """
        self.mkl.set_epsilon(value)
        self.__SVMepsion = value

    @MKLepsilon.setter
    def MKLepsilon(self, value):
        """This method is used for setting the MKL convergence criterion (the minimum allowed error committed by
        the MKL object in test). In other words, the high level of the learning process. The current basis
        kernel combination is tested as the SVM kernel. The basis' weights are tuned until 'MKLeps' is reached.
        See at page 22 of Sonnemburg et.al., (2006) Large Scale Multiple Kernel Learning.
        """
        self.mkl.set_mkl_epsilon(value)
        self.__MKLepsilon = value

    @mklC.setter
    def mklC(self, value):
        """This method is used for setting regularization parameters. These are different when the two classes
        are imbalanced and Equal for balanced densities in binary classification problems. For multiclass
        problems imbalanced densities are not considered, so uniquely the first argument is caught by the method.
        If one or both arguments are misplaced the default values are one both them. See at page 4 of Bagchi,
        (2014) SVM Classifiers Based On Imperfect Training Data.
        """
        if self.binary:
            self.mkl.set_C(value[0], value[1])
        else:
            self.mkl.set_C(value)

        self.__mklC = value

    @threads.setter
    def threads(self, value):
        """This method is used for changing the number of threads we want to be running with a single machine core.
        These threads are not different parallel processes running in different machine cores.
        """
        self.mkl.parallel.set_num_threads(value) 	# setting number of training threads
        self.__threads = value


'''
#### Loading train and test data
# 1) For multiclass problem loaded from file:
[traindata,
 testdata,
 trainlab,
 testlab] = mk.load_Toy('/home/iarroyof/shogun-data/toy/',         # Data directory
                     'fm_train_multiclass_digits500.dat',       # Multiclass dataSet examples file name
                     'label_train_multiclass_digits500.dat')    # Multiclass Labels file name
# 2) For generated binary problem:
#[traindata, testdata, trainlab, testlab] = generate_binToy()

#### Casting train and test data for Shogun feature objects (uncomment and comment labels according to your
#### problem) ******Include following steps in load_Toy() for preventing loading shogun twice.
feats_train = RealFeatures(traindata) 	# train examples
#labelsTr = BinaryLabels(trainlab)	    # train binary lables
labelsTr = MulticlassLabels(trainlab)	# train multiclass labels
feats_test = RealFeatures(testdata)  	# test examples
#labelsTs = BinaryLabels(testlab)	    # test binary labels
labelsTs = MulticlassLabels(testlab)	# test multiclass labels

#### Instantiating the learnable kernel object
kernelO = mk.mklObj(verbose = True, threads = 4)
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

#### With n basis kernels
# Falta hacer funciones para habilitar la opción de cargar sigmas desde archivo.
kernelO.fit_kernel(featsTr = feats_train,
                   targetsTr = labelsTr,
                   featsTs = feats_test,
                   targetsTs = labelsTs,
                   kernelFamily = basisKernelFamily[2],
                   randomRange = [50, 200], # For homogeneous polynomial kernels these two parameter sets
                   randomParams = [50, 20], # have not effect. For quadratic there are not parameter distribution
                   hyper = widthDistribution[0], # With not effect when kernel family is polynomial and some
                   pKers = 3)                    # other powering forms.

#kernelO.filePrintingResults('mkl_output.txt', mode)
#kernelO.save_sigmas()
'''

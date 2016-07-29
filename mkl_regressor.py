from modshogun import *
import Gnuplot, Gnuplot.funcutils
from numpy import *
from sklearn.metrics import r2_score
from scipy.stats import expon
import sys

class mkl_regressor():

    def __init__(self, num_widths = 0, svm_C = 0.01, mkl_C = 1.0, svm_norm = 1, mkl_norm = 1, degree = 2):
        self.svm_C = svm_C
        self.mkl_C = mkl_C
        self.svm_norm = svm_norm
        self.mkl_norm = mkl_norm
        self.degree = degree
 
        if num_widths:
            self.widths = {}
            for w in xrange(num_widths):
                #if median_width:
                self.widths[w] = abs(expon(scale=50, loc = median_width))
                #else:
                #    self.widths.append(expon(scale=100))
            self.num_widths = num_widths

        #else:
        #    self.widths = [1, 2, 3, 4]

    def set_params(**params):
        try:
            self.degree = params['degree']
            self.svm_C = params['svm_C']
            self.mkl_C = params['mkl_C']
            self.mkl_norm = params['mkl_norm']
            self.svm_norm = params['svm_norm']
            self.widths = params['widths']
            self.num_widths = params['num_widths']
            self.median_width = params['median_width']
        except KeyError:
            pass
            
    def fit(self, X, y, **kwargs):
        try:
            self.degree = kwargs['degree']
            self.svm_C = kwargs['svm_c']
            self.mkl_C = kwargs['mkl_c']
            self.mkl_norm = kwargs['mkl_norm']
            self.svm_norm = kwargs['svm_norm']
            self.widths = kwargs['widths']
            self.num_widths = kwargs['num_widths']
            self.median_width = kwards['median_width']
        except KeyError:
            pass

        self.feats_train = SparseRealFeatures(X)
        labels_train = RegressionLabels(y)
        self.kernels  = CombinedKernel()        


        if self.num_widths or self.median_width: # for a given dictionary of widths
            if self.num_widths:
                self.widths = {}
                for w in xrange(self.num_widths):
                    self.widths[w] = numpy.random.uniform
            
            assert isinstance(self.widths, dict) # for a given dictionary of width distributions
            for key, generator in self.widths:
                kernel = GaussianKernel()
                kernel.set_width(abs(generator(low=self.median_width*0.5, high=self.median_width*1.5, size=None)))
                kernel.init(self.feats_train,self.feats_train)
                self.kernels.append_kernel(kernel)
                del kernel

            
        else: 
            assert isinstance(self.widths, list) and self.widths # for a given list of widths
            sys.stderr.write("widths: %s" % self.widths)
            for width in self.widths:
                kernel = GaussianKernel()
                kernel.set_width(width)
                kernel.init(self.feats_train,self.feats_train)
                self.kernels.append_kernel(kernel)
                del kernel

        if self.degree: # if a degree is given the kernel additionally comes in play (None for nonpoly kernel), so p = p + 1
            kernel = PolyKernel(10, self.degree)            
            self.kernels.append_kernel(kernel)
            del kernel

        self.kernels.init(self.feats_train, self.feats_train)

        binary_svm_solver = SVRLight() # seems to be optional, with LibSVR it does not work.
        self.mkl = MKLRegression(binary_svm_solver)

        self.mkl.set_C(self.svm_C, self.svm_C)
        self.mkl.set_C_mkl(self.mkl_C)
        self.mkl.set_mkl_norm(self.mkl_norm)
        self.mkl.set_mkl_block_norm(self.svm_norm)

        self.mkl.set_kernel(self.kernels)
        self.mkl.set_labels(labels_train)
        try:
            self.mkl.train(self.feats_train)
            sys.std_error.write("Estimator: %s" % self.mkl)
        except:
            print "Training Error"
        self.kernel_weights = self.kernels.get_subkernel_weights()
        #self.num_widths = self.kernels.get_num_subkernels()
        self.estimator = self.mkl

    def predict(self, X):
        feats_test = SparseRealFeatures(X)
        self.kernels.init(self.feats_train, feats_test) # test for test

        return self.estimator.apply_regression().get_labels()

    def get_params(self):
        """ self.degree = kwargs['degree']
            self.svm_C = kwargs['svm_c']
            self.mkl_C = kwargs['mkl_c']
            self.mkl_norm = kwargs['mkl_norm']
            self.svm_norm = kwargs['svm_norm']
            self.widths = kwargs['widths']
            self.kernel_weights = self.kernels.get_subkernel_weights()"""

        return {'mkl_c':self.mkl_C, 
                'svm_c':self.svm_C, 
                'mkl_norm': self.mkl_norm, 
                'svm_norm': self.svm_norm,
                'degree': self.degree, 
                'widths': self.widths,
                'weights': self.kernel_weights}
    
    def score(estimator, X, y):

        predicted = estimator.predict(X)
        return r2_score(predicted, y)

    

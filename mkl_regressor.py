from modshogun import *
import Gnuplot, Gnuplot.funcutils
from numpy import *
from sklearn.metrics import r2_score
from scipy.stats import expon
import sys

class mkl_regressor():

    def __init__(self, wd1 = 0.1, wd2 = 0.1, wd3 = 0.1, wd4 = 0.1, wd5 = 0.1, wd6 = 0.1, wd7 = 0.1, wd8 = 0.1, 
                wd9 = 0.1, wd10 = 0.1, wd11 = 0.1, wd12 = 0.1, wd13 = 0.1, wd14 = 0.1, wd15 = 0.1, wd16 = 0.1, 
                weights = [0], num_widths = 2, svm_c = 0.01, mkl_c = 1.0, svm_norm = 1, mkl_norm = 1, degree = 2):
        self.svm_c = svm_c
        self.mkl_c = mkl_c
        self.svm_norm = svm_norm
        self.mkl_norm = mkl_norm
        self.degree = degree
        self.num_widths = num_widths
    #    if num_widths:
    #        self.widths = []
    #        for w in xrange(num_widths):
    #            #if median_width:
    #            self.widths[w] = abs(expon(scale=50, loc = median_width))
    #            #else:
    #            #    self.widths.append(expon(scale=100))
    #        self.num_widths = num_widths#

    #    else:
    #        self.widths = []
    #        assert isinstance(widths, list)
    #        for val in widths:
    #            self.widths.append(val)
        #self.widths = []
        if self.num_widths > 16:
            self.num_widths = 16
        for i in xrange(self.num_widths):
                exec('self.wd' + str(i + 1) + ' = wd' + str(i + 1))

        self.kernel_weights = zeros([1, self.num_widths])

    def set_params(self, **params):
        try:
            self.degree = params['degree']
            self.svm_c = params['svm_c']
            self.mkl_c = params['mkl_c']
            self.mkl_norm = params['mkl_norm']
            self.svm_norm = params['svm_norm']
            self.num_widths = params['num_widths'],
            #self.wd1 = params['wd1']; self.wd2 = params['wd2']; self.wd3 = params['wd3']; self.wd3 = params[' wd3'] 
            #self.wd4 = params['wd4']; self.wd5 = params[' wd5']; self.wd6 = params['wd6']; self.wd7 = params[' wd7']
            #self.wd8 = params['wd8']; self.wd9 = params[' wd9']; self.wd10 = params['wd10']; self.wd11 = params[' wd11']
            #self.wd12 = params['wd12']; self.wd13 = params['wd13']; self.wd14 = params['wd14']; self.wd15 = params['wd15'];
            #self.wd16 = params['wd16']
            #self.widths = []
            if self.num_widths > 16:
                self.num_widths
            
            for i in xrange(self.num_widths):
                exec('self.wd' + str(i + 1) + "= kwargs['wd'" + str(i + 1) + ']')
            #for i in xrange(self.num_widths):
            #    exec('self.widths.append(wd' + str(i + 1) + ')')
            #    if not self.widths[i]:
            #        break
            #self.kernel_weights = zeros(1,len(self.widths))
            self.median_width = params['median_width']
        except KeyError:
            pass
    
    def fit(self, X, y, **kwargs):
        try:
            self.degree = kwargs['degree']
            self.svm_c = kwargs['svm_c']
            self.mkl_c = kwargs['mkl_c']
            self.mkl_norm = kwargs['mkl_norm']
            self.svm_norm = kwargs['svm_norm']
            self.num_widths = kwargs['num_widths']
            #self.widths = []
            if self.num_widths > 16:
                self.num_widths
            for i in xrange(self.num_widths):
                exec('self.wd' + str(i + 1) + "= kwargs['wd'" + str(i + 1) + ']')
                #if not self.widths[i]:
                #    break
            self.median_width = kwards['median_width']
         
        except KeyError:
            pass

        assert self.num_widths > 1

        #self.kernel_weights = zeros([1,len(self.widths)])
        self.feats_train = SparseRealFeatures(X).T
        labels_train = RegressionLabels(y)
        self.kernels  = CombinedKernel()        

        #if self.num_widths or self.median_width: # for a given dictionary of widths
            #if self.num_widths:
            #    self.widths = {}
            #    for w in xrange(self.num_widths):
            #        self.widths[w] = numpy.random.uniform
            
            #assert isinstance(self.widths, dict) # for a given dictionary of width distributions
         #   for w in self.widths:
         #       kernel = GaussianKernel()
         #       kernel.set_width(w)
         #       kernel.init(self.feats_train,self.feats_train)
         #       self.kernels.append_kernel(kernel)
         #       del kernel
         #   sys.stderr.write("widths dic: %s" % self.widths)
            
        #else: 
        #assert isinstance(self.widths, list) and self.widths # for a given list of widths
        #sys.stderr.write("widths list: %s" % self.widths)
        for width in xrange(self.num_widths):
            kernel = GaussianKernel()
            exec("kernel.set_width(self.wd" + str(width+1) + ")" ) 
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

        self.mkl.set_C(self.svm_c, self.svm_c)
        self.mkl.set_C_mkl(self.mkl_c)
        self.mkl.set_mkl_norm(self.mkl_norm)
        self.mkl.set_mkl_block_norm(self.svm_norm)

        self.mkl.set_kernel(self.kernels)
        self.mkl.set_labels(labels_train)
        #try:
        self.mkl.train(self.feats_train)
        sys.std_error.write("Estimator: %s" % self.mkl)
        #except:
        #    print "Training Error"
        self.kernel_weights = self.kernels.get_subkernel_weights()
        #self.num_widths = self.kernels.get_num_subkernels()
        self.estimator = self.mkl

    def predict(self, X):
        feats_test = SparseRealFeatures(X).T
        self.kernels.init(self.feats_train, feats_test) # test for test

        return self.estimator.apply_regression().get_labels()

    def get_params(self, deep = False):
        """ self.degree = kwargs['degree']
            self.svm_c = kwargs['svm_c']
            self.mkl_c = kwargs['mkl_c']
            self.mkl_norm = kwargs['mkl_norm']
            self.svm_norm = kwargs['svm_norm']
            self.widths = kwargs['widths']
            self.kernel_weights = self.kernels.get_subkernel_weights()"""
        params = {'mkl_c':self.mkl_c,
                'svm_c':self.svm_c,
                'mkl_norm': self.mkl_norm,
                'svm_norm': self.svm_norm,
                'degree': self.degree,
                'weights': self.kernel_weights}
        for i in xrange(self.num_widths):
                exec("params['wd" + str(i + 1) + "'] = self.wd" + str(i + 1) )    
            
        return params 
    
    def score(estimator, X, y):

        predicted = estimator.predict(X)
        return r2_score(predicted, y)

    

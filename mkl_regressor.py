from modshogun import *
from numpy import *
from sklearn.metrics import r2_score
from scipy.stats import randint
from scipy import stats
from scipy.stats import randint as sp_randint
from scipy.stats import expon
import sys

class mkl_regressor():

    def __init__(self, widths = None, kernel_weights = None, svm_c = 0.01, mkl_c = 1.0, svm_norm = 1, mkl_norm = 1, degree = 2, 
                    median_width = None, width_scale = 20.0, min_size=2, max_size = 10, kernel_size = None):
        self.svm_c = svm_c
        self.mkl_c = mkl_c
        self.svm_norm = svm_norm
        self.mkl_norm = mkl_norm
        self.degree = degree
        self.widths = widths
        self.kernel_weights = kernel_weights       
        self.median_width = median_width
        self.width_scale = width_scale
        self.min_size = min_size
        self.max_size = max_size
        self.kernel_size = kernel_size
                 
    def combine_kernel(self):

        self._kernels_  = CombinedKernel()
        for width in self.widths:
            kernel = GaussianKernel()
            kernel.set_width(width)
            kernel.init(self.feats_train, self.feats_train)
            self._kernels_.append_kernel(kernel)
            del kernel

        kernel = PolyKernel(10, self.degree)
        kernel.init(self.feats_train, self.feats_train)
        self._kernels_.append_kernel(kernel)
        del kernel
        self._kernels_.init(self.feats_train, self.feats_train)

    def fit(self, X, y, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        labels_train = RegressionLabels(y.reshape((len(y), )))

        self.feats_train = RealFeatures(X.T)
        self.combine_kernel()

        binary_svm_solver = SVRLight() # seems to be optional, with LibSVR it does not work.
        self.mkl = MKLRegression(binary_svm_solver)

        self.mkl.set_C(self.svm_c, self.svm_c)
        self.mkl.set_C_mkl(self.mkl_c)
        self.mkl.set_mkl_norm(self.mkl_norm)
        self.mkl.set_mkl_block_norm(self.svm_norm)

        self.mkl.set_kernel(self._kernels_)
        self.mkl.set_labels(labels_train)
        try:
            self.mkl.train()
        except SystemError as inst:
            if "Assertion" in str(inst):
                sys.stderr.write("""WARNING: Bad parameter combination: [svm_c %f mkl_c %f mkl_norm %f svm_norm %f, degree %d] \n widths %s \n
                                    MKL error [%s]""" % (self.svm_c, self.mkl_c, self.mkl_norm, self.svm_norm, self.degree, self.widths, str(inst)))
                pass
            
        self.kernel_weights = self._kernels_.get_subkernel_weights()
        self.__loaded = False

    def predict(self, X):
        self.feats_test = RealFeatures(X.T)
        ft = None
        if not self.__loaded:
            self._kernels_.init(self.feats_train, self.feats_test) # test for test
            self.mkl.set_kernel(self._kernels_)
        else:
            ft = CombinedFeatures()
            ft.append_feature_obj(self.feats_test)

        return self.mkl.apply_regression(ft).get_labels()

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)

        return self

    def get_params(self, deep=False):

        return {param: getattr(self, param) for param in dir(self) if not param.startswith('__') and not callable(getattr(self,param))}    

    def score(self,  X_t, y_t):

        predicted = self.predict(X_t)
        return r2_score(predicted, y_t)    
#median_width = None, width_scale = 20.0, min_size=2, max_size = 10, kernel_size = None):
    def param_vector(self):
        """Gives a vector of weights which distribution is linear. The 'median' value is used both as location parameter and
            for scaling parameter. If not size of the output vector is given, a random size between 'min_size' and 'max_size' is
            returned."""
        if not self.kernel_size:
            self.kernel_size = randint.rvs(low = self.min_size, high = self.max_size, size = 1)

        self.widths = linspace(start = self.median_width * 0.001, stop = self.median_width * self.width_scale, num = self.kernel_size)

class expon_vector(stats.rv_continuous):
    
    def __init__(self, loc = 1.0, scale = None, min_size=2, max_size = 10, size = None):
        self.loc = loc
        self.scale = scale
        self.min_size = min_size
        self.max_size = max_size
        self.size = size

    def rvs(self):
    
        if not self.size:        
            self.size = randint.rvs(low = self.min_size, high = self.max_size, size = 1)
        if self.scale:
            return expon.rvs(loc  = self.loc * 0.09, scale = self.scale, size = self.size)
        else:
            return expon.rvs(loc = self.loc * 0.09, scale = self.loc * 8.0, size = self.size)
    


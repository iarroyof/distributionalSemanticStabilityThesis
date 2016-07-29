from modshogun import *
import Gnuplot, Gnuplot.funcutils
from numpy import *
from sklearn.metrics import r2_score

class mkl_regressor():

    def __init__(self, widths = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0], kernel_weights = [0.01, 0.1, 1.0,], svm_c = 0.01, mkl_c = 1.0, svm_norm = 1, mkl_norm = 1, degree = 2):
        self.svm_c = svm_c
        self.mkl_c = mkl_c
        self.svm_norm = svm_norm
        self.mkl_norm = mkl_norm
        self.degree = degree
        self.widths = widths
        self.kernel_weights = kernel_weights
                

    def fit(self, X, y, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)        
       
        self.feats_train = RealFeatures(X.T)
        labels_train = RegressionLabels(y.reshape((len(y), )))
        self._kernels_  = CombinedKernel()
        for width in self.widths:
            kernel = GaussianKernel()
            kernel.set_width(width)
            kernel.init(self.feats_train,self.feats_train)
            self._kernels_.append_kernel(kernel)
            del kernel

        kernel = PolyKernel(10, self.degree)            
        self._kernels_.append_kernel(kernel)
        del kernel

        self._kernels_.init(self.feats_train, self.feats_train)

        binary_svm_solver = SVRLight() # seems to be optional, with LibSVR it does not work.
        self.mkl = MKLRegression(binary_svm_solver)

        self.mkl.set_C(self.svm_c, self.svm_c)
        self.mkl.set_C_mkl(self.mkl_c)
        self.mkl.set_mkl_norm(self.mkl_norm)
        self.mkl.set_mkl_block_norm(self.svm_norm)

        self.mkl.set_kernel(self._kernels_)
        self.mkl.set_labels(labels_train)
        self.mkl.train()
        self.kernel_weights = self._kernels_.get_subkernel_weights()

    def predict(self, X):
        self.feats_test = RealFeatures(X.T)
        self._kernels_.init(self.feats_train, self.feats_test) # test for test
        self.mkl.set_kernel(self._kernels_)
        return self.mkl.apply_regression().get_labels()

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)

        return self

    def get_params(self, deep=False):

        return {param: getattr(self, param) for param in dir(self) if not param.startswith('__') and not callable(getattr(self,param))}    

    def score(self,  X_t, y_t):

        predicted = self.predict(X_t)
        return r2_score(predicted, y_t)    

    

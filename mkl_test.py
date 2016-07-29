from modshogun import *
#from numpy import random, array, loadtxt
#from scipy.sparse import csr_matrix
#from tools.load import LoadMatrix 
import Gnuplot, Gnuplot.funcutils
from numpy import *
from sklearn.metrics import r2_score

#lm = LoadMatrix()

# Create the input sparse matrix
#r=array([0,2,0,2,1,2,1,2])
#c=array([0,0,1,1,2,2,3,3])
#data=array([0.950836978354, 2.40814562724, 0.441304200555, 3.7801571175, 0.4498, 3.5287, 1.4163, 0.2434]) 
#train_data = csr_matrix((data, (r, c)), shape=(3, 4)).astype('float64').T

# Convert data to Shogun sparse features

def scorer(estimator, X, y):

    predicted = estimator.predict(X)
    return r2_score(predicted, y)

class mkl_regressor():

    def __init__(self, widths = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0], svm_C = 0.01, mkl_C = 1.0, svm_norm = 1, mkl_norm = 1, degree = 2):
        self.svm_C = svm_C
        self.mkl_C = mkl_C
        self.svm_norm = svm_norm
        self.mkl_norm = mkl_norm
        self.degree = degree
        self.widths = widths

    def fit(self, X, y, **kwargs):
        try:
            self.degree = kwargs['degree']
            self.svm_C = kwargs['svm_c']
            self.mkl_C = kwargs['mkl_c']
            self.mkl_norm = kwargs['mkl_norm']
            self.svm_norm = kwargs['svm_norm']
            self.widths = kwargs['widths']
        except KeyError:
            pass
        
        self.feats_train = SparseRealFeatures(X)
        labels_train = RegressionLabels(y)
        self.kernels  = CombinedKernel()
        for width in self.widths:
            kernel = GaussianKernel()
            kernel.set_width(width)
            kernel.init(self.feats_train,self.feats_train)
            self.kernels.append_kernel(kernel)
            del kernel

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
        self.mkl.train(self.feats_train)
        self.kernel_weights = self.kernels.get_subkernel_weights()

    def predict(self, X):
        feats_test = SparseRealFeatures(X)
        self.kernels.init(self.feats_train, feats_test) # test for test

        return self.mkl.apply_regression().get_labels()

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
    

if __name__ == "__main__":

    labels = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/STS.gs.OnWN.txt")
    data = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/vectors_H10/pairs_eng-NO-test-2e6-nonempty_OnWN_d2v_H10_sub_m5w8.mtx").T

    labels_t = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/STS.gs.FNWN.txt")
    data_t = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/vectors_H10/pairs_eng-NO-test-2e6-nonempty_FNWN_d2v_H10_sub_m5w8.mtx").T

    mkl = mkl_regressor()
    mkl.fit(data, labels)
    preds = mkl.predict(data_t)

    print "R^2: ", scorer(mkl, data_t, labels_t)
    print "Parameters: ",  mkl.get_params()
    pred, real = zip(*sorted(zip(preds, labels_t), key=lambda tup: tup[1]))


    g = Gnuplot.Gnuplot()   
    g.plot(Gnuplot.Data(pred, with_="lines"), Gnuplot.Data(real, with_="linesp") )

    

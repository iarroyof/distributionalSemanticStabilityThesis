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
        print ">> samples", self.feats_train.get_num_vectors(), "dims",self.feats_train.get_num_features(), "L ", labels_train.get_num_labels()

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

if __name__ == "__main__":

#    labels = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/STS.gs.OnWN.txt")
#    data = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/vectors_H10/pairs_eng-NO-test-2e6-nonempty_OnWN_d2v_H10_sub_m5w8.mtx").T

#    labels_t = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/STS.gs.FNWN.txt")
#    data_t = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/vectors_H10/pairs_eng-NO-test-2e6-nonempty_FNWN_d2v_H10_sub_m5w8.mtx").T
    from sklearn.grid_search import RandomizedSearchCV as RS
    from scipy.stats import randint as sp_randint
    from scipy.stats import expon

    labels = array([2.0,0.0,2.0,1.0,3.0,2.0])
    labels = labels.reshape((len(labels), 1))
    data = array([[1.0,2.0,3.0],[1.0,2.0,9.0],[1.0,2.0,3.0],[1.0,2.0,0.0],[0.0,2.0,3.0],[1.0,2.0,3.0]])
    labels_t = array([1.,3.,4])
    labels_t = labels_t.reshape((len(labels_t), 1))
    data_t = array([[20.0,30.0,40.0],[10.0,20.0,30.0],[10.0,20.0,40.0]])
    k = 3
    print ">> Shapes: labels %s; Data %s\n\tlabelsT %s; DataT %s" % (labels.shape, data.shape, labels_t.shape, data_t.shape)

    param_grid = [ {'svm_c': expon(scale=100, loc=5),
                    'mkl_c': expon(scale=100, loc=5),
                    'degree': sp_randint(0, 32),
                    #'widths': [array([4.0,6.0,8.9,3.0]), array([4.0,6.0,8.9,3.0,2.0, 3.0, 4.0]), array( [100.0, 200.0, 300.0, 400.0]) 
                    'widths': [array([expon, expon])] 
                  }]
    
    mkl = mkl_regressor()
    rs = RS(mkl, param_distributions = param_grid[0], n_iter = 10, n_jobs = 24, cv = k)#, scoring="r2", verbose=True)
    rs.fit(data, labels)
    preds = rs.predict(data_t)
    #mkl.fit(data, labels)
    #preds = mkl.predict(data_t)

    print "R^2: ", rs.score(data_t, labels_t)
    print "Parameters: ",  rs.best_params_
    pred, real = zip(*sorted(zip(preds, labels_t), key=lambda tup: tup[1]))


    g = Gnuplot.Gnuplot()   
    g.plot(Gnuplot.Data(pred, with_="lines"), Gnuplot.Data(real, with_="linesp") )

    

from modshogun import *
from numpy import *
from sklearn.metrics import r2_score
from scipy.stats import randint
from scipy import stats
from scipy.stats import randint as sp_randint
from scipy.stats import expon
import sys, os

class mkl_regressor():

    def __init__(self, widths = array([0.01, 0.1, 1.0, 10.0, 50.0, 100.0]), kernel_weights = [0.01, 0.1, 1.0,], 
                        svm_c = 0.01, mkl_c = 1.0, svm_norm = 1, mkl_norm = 1, degree = 2):
        self.svm_c = svm_c
        self.mkl_c = mkl_c
        self.svm_norm = svm_norm
        self.mkl_norm = mkl_norm
        self.degree = degree
        self.widths = widths
        self.kernel_weights = kernel_weights                
    
    def combine_kernel(self):

        self._kernels_  = CombinedKernel()
        f = CombinedFeatures()
        for width in self.widths:
            kernel = GaussianKernel()
            kernel.set_width(width)
            kernel.init(self.feats_train,self.feats_train)
            self._kernels_.append_kernel(kernel)
            del kernel

        kernel = PolyKernel(10, self.degree)            
        self._kernels_.append_kernel(kernel)
        del kernel
        f.append_feature_obj(self.feats_train)
        self.feats_train = f; del f
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
        self.mkl.train()
        self.kernel_weights = self._kernels_.get_subkernel_weights()
        self.__loaded = False

    def predict(self, X):
        """ For prediction from here, once the model was loaded from outside, test if saving the kernel separetely and loading it again."""
        self.feats_test = CombinedFeatures()
        self.feats_test.append_feature_obj(RealFeatures(X.T))

        if not self.__loaded:
            self._kernels_.init(self.feats_train, self.feats_test) # test for test
            self.mkl.set_kernel(self._kernels_)

        return self.mkl.apply_regression(self.feats_test).get_labels()

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)

        return self

    def get_params(self, deep=False):

        return {param: getattr(self, param) for param in dir(self) if not param.startswith('__') and not callable(getattr(self,param))}    

    def score(self,  X_t, y_t):

        predicted = self.predict(X_t)
        return r2_score(predicted, y_t)    
    
    def serialization_matrix (self, file_name, sl='save'):
        from os.path import basename, dirname
        from bz2 import BZ2File as bz

        if sl == 'save': mode = "wb"
        else:            mode = "rb"

        try:
            #fstream = SerializableAsciiFile(dirname(file_name) + "/mtx_" + basename(file_name), mode)
            f = bz2.BZ2File(dirname(file_name) + "/mtx_" + basename(file_name), mode)
        except:
            sys.stderr.write("Error serializing kernel matrix.")

        if sl == 'save':
            #self.feats_train.save_serializable(fstream)
            #os.unlink(file_name)
            pickle.dump(self.mkl, f, protocol=2)
        else:
            #self.feats_train = RealFeatures()
            #self.feats_train.load_serializable(fstream)
            self.mkl = pickle.load(f)
            self.__loaded = True

    def save(self, file_name = None):
        """ Python reimplementated function for saving a pretrained MKL machine.
        This method saves a trained MKL machine to the file 'file_name'. If not 'file_name' is given, a
        dictionary 'mkl_machine' containing parameters of the given trained MKL object is returned.
        Here we assumed all subkernels of the passed CombinedKernel are of the same family, so uniquely the
        first kernel is used for verifying if the passed 'kernel' is a Gaussian mixture. If it is so, we insert
        the 'widths' to the model dictionary 'mkl_machine'. An error is returned otherwise.
        """
        mkl_machine = {}
        support = []
        mkl_machine['num_support_vectors'] = self.mkl.get_num_support_vectors()
        mkl_machine['bias'] = self.mkl.get_bias()
        for i in xrange(mkl_machine['num_support_vectors']):
            support.append((self.mkl.get_alpha(i), self.mkl.get_support_vector(i)))

        mkl_machine['support'] = support
        mkl_machine['weights'] = list(self._kernels_.get_subkernel_weights())
        mkl_machine['family'] = self._kernels_.get_first_kernel().get_name()
        mkl_machine['widths'] = self.widths
        mkl_machine['params'] = self.get_params()
 
        if file_name:
            with open(file_name,'w') as f:
                f.write(str(mkl_machine)+'\n')
            serialization_matrix(file_name, "save")    
        else:
            return mkl_machine


    def load(self, file_name):
        """ This method receives a file name (if it is not in pwd, full path must be given). The loaded file 
        must contain at least a dictionary at its top. This dictionary must contain a key called 'model' whose 
        value must be a dictionary, from which model parameters will be read. For example:
            {'key_0':value, 'key_1':value,..., 'model':{'family':'PolyKernel', 'bias':1.001,...}, key_n:value}
        Four objects are returned. The MKL model which is tuned to those parameters stored at the given file. A
        numpy array containing learned weights of a CombinedKernel. The widths corresponding to returned kernel
        weights and the kernel family. Be careful with the kernel family you are loading because widths no
        necessarily are it, but probably 'degrees' for the 'family':'PolyKernel' key-value.
        The Combined kernel must be instantiated outside this method, thereby loading to it corresponding
        weights and widths.
        """
        with open(file_name, 'r') as pointer:
            mkl_machine = eval(pointer.read())['learned_model']

        self.mkl.set_bias(mkl_machine['bias'])
        self.mkl.create_new_model(mkl_machine['num_support_vectors']) # Initialize the inner SVM
        for parameter, value in mkl_machine['params'].items():
            setattr(self, parameter, value)

        serialization_matrix(file_name, "load")
        self.mkl.set_kernel(self._kernels_) 
        for i in xrange(mkl_machine['num_support_vectors']):
            self.mkl.set_alpha(i, mkl_machine['support'][i][0])
            self.mkl.set_support_vector(i, mkl_machine['support'][i][1])
        #mkl_machine['weights'] = numpy.array(mkl_machine['weights'])
        return self
   
        #return mkl, mkl_machine

class expon_vector(stats.rv_continuous):
    
    def __init__(self, loc = 1.0, scale = None, min_size=2, max_size = 10, size = None):
        self.loc = loc
        self.scale = scale
        self.min_size = min_size
        self.max_size = max_size
        self.size = size # Only for initialization

    def rvs(self):
        if not self.size:
            self.size = randint.rvs(low = self.min_size, high = self.max_size, size = 1)
        if self.scale:
            return expon.rvs(loc = self.loc * 0.09, scale = self.scale, size = self.size)
        else:
            return expon.rvs(loc = self.loc * 0.09, scale = self.loc * 8.0, size = self.size)
    

if __name__ == "__main__":

    import Gnuplot, Gnuplot.funcutils
    from sklearn.grid_search import RandomizedSearchCV as RS
    from sklearn.externals import joblib
    import pickle
#    labels = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/STS.gs.OnWN.txt")
#    data = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/vectors_H10/pairs_eng-NO-test-2e6-nonempty_OnWN_d2v_H10_sub_m5w8.mtx")

#    labels_t = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/STS.gs.FNWN.txt")
#    data_t = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/vectors_H10/pairs_eng-NO-test-2e6-nonempty_FNWN_d2v_H10_sub_m5w8.mtx")
#    from sklearn.grid_search import RandomizedSearchCV as RS


    labels = array([2.0,0.0,2.0,1.0,3.0,2.0])
    labels = labels.reshape((len(labels), 1))
    data = array([[1.0,2.0,3.0],[1.0,2.0,9.0],[1.0,2.0,3.0],[1.0,2.0,0.0],[0.0,2.0,3.0],[1.0,2.0,3.0]])
    labels_t = array([1.,3.,4])
    labels_t = labels_t.reshape((len(labels_t), 1))
    data_t = array([[20.0,30.0,40.0],[10.0,20.0,30.0],[10.0,20.0,40.0]])
    
    model_file = None# "/almac/ignacio/data/mkl_models/mkl_0.asc"

    if not model_file:
        k = 3
        N = 2
        m = 10.0
        print ">> Shapes: labels %s; Data %s\n\tlabelsT %s; DataT %s" % (labels.shape, data.shape, labels_t.shape, data_t.shape)
        params = {'svm_c': expon(scale=100, loc=5),
                    'mkl_c': expon(scale=100, loc=5),
                    'degree': sp_randint(0, 24),
                    'widths': expon_vector(loc = m, min_size = 2, max_size = 10) }
        param_grid = []
        for i in xrange(N):
            param_grid.append(params)
        i = 0
        for params in param_grid:
            mkl = mkl_regressor()
            rs = RS(mkl, param_distributions = params, n_iter = 20, n_jobs = 24, cv = k, scoring="mean_squared_error")#"r2")
            rs.fit(data, labels)
            #with open('/almac/ignacio/data/mkl_models/mkl_%d.pkl' % i, 'w') as f:
            #    pickle.dump(rs._kernels_, f, protocol=2)
            rs.best_estimator_.save('/almac/ignacio/data/mkl_models/mkl_%d.asc' % i)

            preds = rs.best_estimator_.predict(data_t)
            print "R^2: ", r2_score(preds, labels_t)
            print "Parameters: ",  rs.best_params_
            i += 1
            pred, real = zip(*sorted(zip(preds, labels_t), key=lambda tup: tup[1]))
            g = Gnuplot.Gnuplot()   
            g.plot(Gnuplot.Data(pred, with_="lines"), Gnuplot.Data(real, with_="linesp") )

    else:
        g = Gnuplot.Gnuplot()        
        mkl = mkl_regressor()
        #new_mkl = MKLRegression()
        #fstream = SerializableAsciiFile(model_file, "r")
        #status = new_mkl.load_serializable(fstream)
        #sys.stderr.write("MKL model loading status: %s" % status)
        #os.unlink(model_file)
        #new_mkl.train()
        mkl.load(model_file)
        preds = mkl.predict(data_t)
        #preds = list(mkl.apply_regression(data_t.T).get_labels())
        if labels_t:
            print "R^2: ", r2_score(preds, labels_t)
        #print "Parameters: ",   new_mkl.get_params()
            pred, real = zip(*sorted(zip(preds, labels_t), key=lambda tup: tup[1]))
        else: 
            pred = preds; real = range(len(pred))

        g.plot(Gnuplot.Data(pred, with_="lines"), Gnuplot.Data(real, with_="linesp") )

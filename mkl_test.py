from modshogun import *
from numpy import *
from sklearn.metrics import r2_score
from scipy.stats import randint
from scipy import stats
from scipy.stats import randint as sp_randint
from scipy.stats import expon
import sys, os
import Gnuplot, Gnuplot.funcutils

#from pdb import set_trace as st

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

        self.__kernels  = CombinedKernel()
        for width in self.widths:
            kernel = GaussianKernel()
            kernel.set_width(width)
            kernel.init(self.__feats_train, self.__feats_train)
            self.__kernels.append_kernel(kernel)
            del kernel

        kernel = PolyKernel(10, self.degree)            
        kernel.init(self.__feats_train, self.__feats_train)
        self.__kernels.append_kernel(kernel)
        del kernel
        self.__kernels.init(self.__feats_train, self.__feats_train)

    def fit(self, X, y, **params):

        for parameter, value in params.items():
            setattr(self, parameter, value)        
        labels_train = RegressionLabels(y.reshape((len(y), )))
        
        self.__feats_train = RealFeatures(X.T)
        self.combine_kernel()

        binary_svm_solver = SVRLight() # seems to be optional, with LibSVR it does not work.
        self.__mkl = MKLRegression(binary_svm_solver)

        self.__mkl.set_C(self.svm_c, self.svm_c)
        self.__mkl.set_C_mkl(self.mkl_c)
        self.__mkl.set_mkl_norm(self.mkl_norm)
        self.__mkl.set_mkl_block_norm(self.svm_norm)

        self.__mkl.set_kernel(self.__kernels)
        self.__mkl.set_labels(labels_train)
        try:
            self.__mkl.train()
        except SystemError as inst:
            if "Assertion" in str(inst):
                sys.stderr.write("""WARNING: Bad parameter combination: [svm_c %f mkl_c %f mkl_norm %f svm_norm %f, degree %d] \n widths %s \n
                                    MKL error [%s]""" % (self.svm_c, self.mkl_c, self.mkl_norm, self.svm_norm, self.degree, self.widths, str(inst)))
                pass
        self.kernel_weights = self.__kernels.get_subkernel_weights()
        self.__loaded = False

    def predict(self, X):

        self.__feats_test = RealFeatures(X.T)
        ft = None
        if not self.__loaded:
            self.__kernels.init(self.__feats_train, self.__feats_test) # test for test
            self.__mkl.set_kernel(self.__kernels)
        else:
            ft = CombinedFeatures()
            for i in xrange(self.__mkl.get_kernel().get_num_subkernels()):
                ft.append_feature_obj(self.__feats_test)

        return self.__mkl.apply_regression(ft).get_labels()

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)

        return self

    def get_params(self, deep=False):

        return {param: getattr(self, param) for param in dir(self) if not param.startswith('__') and not '__' in param and not callable(getattr(self,param))}    

    def score(self,  X_t, y_t):

        predicted = self.predict(X_t)
        return r2_score(predicted, y_t)    
   
    def serialize_model (self, file_name, sl="save"):
        from os.path import basename, dirname
        from bz2 import BZ2File
        import pickle

        if sl == "save": mode = "wb"
        elif sl == "load": mode = "rb"
        else: sys.stderr.write("Bad option. Only 'save' and 'load' are available.")

        f = BZ2File(file_name + ".bin", mode)
        if not f: 
            sys.stderr.write("Error serializing kernel matrix.")
            exit()
        
        if sl == "save":
            pickle.dump(self.__mkl, f, protocol=2)
        elif sl == "load":
            mkl = self.__mkl = pickle.load(f)
            self.__loaded = True  
        else: sys.stderr.write("Bad option. Only 'save' and 'load' are available.")

    def save(self, file_name = None):
        """ Python reimplementated function for saving a pretrained MKL machine.
        This method saves a trained MKL machine to the file 'file_name'. If not 'file_name' is given, a
        dictionary 'mkl_machine' containing parameters of the given trained MKL object is returned.
        Here we assumed all subkernels of the passed CombinedKernel are of the same family, so uniquely the
        first kernel is used for verifying if the passed 'kernel' is a Gaussian mixture. If it is so, we insert
        the 'widths' to the model dictionary 'mkl_machine'. An error is returned otherwise.
        """
        self._support = []
        self._num_support_vectors = self.__mkl.get_num_support_vectors()
        self._bias = self.__mkl.get_bias()
        for i in xrange(self._num_support_vectors):
            self._support.append((self.__mkl.get_alpha(i), self.__mkl.get_support_vector(i)))
        
        self._kernel_family = self.__kernels.get_first_kernel().get_name()
 
        if file_name:
            with open(file_name,'w') as f:
                f.write(str(self.get_params())+'\n')
            self.serialize_model(file_name, "save")    
        else:
            return self.get_params()

    def load(self, file_name):
        """ This method receives a 'file.model' file name (if it is not in pwd, full path must be given). The loaded file 
        must contain at least a dictionary at its top. This dictionary must contain keys from which model 
        parameters will be read (including weights, C, etc.). For example:
            {'bias': value, 'param_1': value,...,'support_vectors': [(idx, value),(idx, value)], param_n: value}
        The MKL model is tuned to those parameters stored at the given file.  Other file with double extension must
        be jointly with the model file: '*file.model.bin' where the kernel matrix is encoded together with the kernel
        machine.
        """
        # Load machine parameters
        with open(file_name, 'r') as pointer:
            mkl_machine = eval(pointer.read())
        # Set loaded parameters
        for parameter, value in mkl_machine.items():
            setattr(self, parameter, value)
        # Load the machine itself
        self.serialize_model(file_name, "load") # Instantiates the loaded MKL.
        return self   

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
            return expon.rvs(loc = self.loc * 0.09, scale = self.scale, size = self.size)
        else:
            return expon.rvs(loc = self.loc * 0.09, scale = self.loc * 8.0, size = self.size)
 
def param_vector(self):
        """Gives a vector of weights which distribution is linear. The 'median' value is used both as location parameter and
            for scaling parameter. If not size of the output vector is given, a random size between 'min_size' and 'max_size' is
            returned."""
        if not self.kernel_size:
            self.kernel_size = randint.rvs(low = self.min_size, high = self.max_size, size = 1) 

def test_predict(data, machine = None, file=None, labels = None, out_file=None):
    g = Gnuplot.Gnuplot()
    if type(machine) is str:
        if "mkl_regerssion" == machine:
            machine_ = mkl_regressor()
            machine_.load(model_file)
		# elif other machine types ...
        else:
            print "Error machine type"
            exit()
     # elif other machine types ...  
    else:
        machine_ = machine

    preds = machine_.predict(data)

    if labels is not None:
        r2 = r2_score(preds, labels)
        print "R^2: ", r2
        pred, real = zip(*sorted(zip(preds, labels), key=lambda tup: tup[1]))

    else:
        pred = preds; real = range(len(pred))
    
    if out_file:
        output = {}
        output['learned_model'] = out_file
        output['estimated_output'] = preds
        output['best_params'] = machine_.get_params()
        output['performance'] = r2
        with open(out_file, "a") as f:
            f.write(str(output)+'\n')
    
    print "Machine Parameters: ",  machine_.get_params()
    g.plot(Gnuplot.Data(pred, with_="lines"), Gnuplot.Data(real, with_="linesp") )

if __name__ == "__main__":
    from sklearn.grid_search import RandomizedSearchCV as RS
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
    
    #name_components = shatter_file_name()
    model_file = None# "/almac/ignacio/data/mkl_models/mkl_0.model"
    out_file = "mkl_outs/mkl_idx_corpus_source_repr_dims_op_other.out"
    
    e = False #True
    p = True
    X = True
    Y = True
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
            rs.best_estimator_.save('/almac/ignacio/data/mkl_models/mkl_%d.model' % i)

            if e: # If user wants to save estimates
                #ests = rs.best_estimator_.predict(data)
                test_predict(data = data, machine = rs.best_estimator_, labels = labels, out_file = out_file)
                if p: # If user wants to predict and save just after training.
                    assert not X is None # Provide test data
                   #preds = rs.best_estimator_.predict(data_t)
                    if Y: # Get performance
                        test_predict(data = data_t, machine = rs.best_estimator_, labels = labels_t, out_file = out_file + ".pred")
                    else: # Only predictions
                        test_predict(data = data_t, machine = rs.best_estimator_, out_file = out_file + ".pred")

        sys.stderr.write("\n:>> Finished!!\n" )
    else:
        idx = 0
        test_predict(data = data_t, machine = "mkl_regerssion", file="/almac/ignacio/data/mkl_models/mkl_%d.asc" % idx, 
                        labels = labels_t, out_file = out_file)

        sys.stderr.write("\n:>> Finished!!\n" )

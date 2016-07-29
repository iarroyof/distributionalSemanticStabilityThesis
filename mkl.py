from modshogun import *
import Gnuplot, Gnuplot.funcutils
from numpy import *
from sklearn.metrics import r2_score
from mkl_regressor import *
from scipy.stats import randint as sp_randint
from sklearn.grid_search import  RandomizedSearchCV as RS
from scipy.stats import expon, uniform
import sys

from sklearn.cross_validation import cross_val_score

def scorer(estimator, X, y):

    predicted = estimator.predict(X)
    return r2_score(predicted, y)


if __name__ == "__main__":

    #labels = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/STS.gs.OnWN.txt")
    #data = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/vectors_H10/pairs_eng-NO-test-2e6-nonempty_OnWN_d2v_H10_sub_m5w8.mtx").astype("float64")
    labels = array([2.,3.,2.])
    data = array([[1.,2.,3.,4.],[1.,2.,3.,4.],[1.,2.,3.,4.]])
    print data.shape
    #labels_t = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/STS.gs.FNWN.txt")
    #data_t = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/vectors_H10/pairs_eng-NO-test-2e6-nonempty_FNWN_d2v_H10_sub_m5w8.mtx").astype("float64")
    labels_t = array([1.,3.,4])
    data_t = array([[1.,2.,3.,4.],[1.,2.,3.,4.],[1.,2.,3.,4.]])
    print data_t.shape
    k = 2
    median_wd = 0.01


    """     self.degree = kwargs['degree']
            self.svm_c = kwargs['svm_c']
            self.mkl_c = kwargs['mkl_c']
            self.mkl_norm = kwargs['mkl_norm']
            self.svm_norm = kwargs['svm_norm']
            self.widths = kwargs['widths']
            self.kernel_weights = self.kernels.get_subkernel_weights()
            self.num_widths
"""


    param_grid = [ {'svm_c': expon(scale=100, loc=5), 
                    'mkl_c': expon(scale=100, loc=5),                    
                    'degree': sp_randint(0, 32), 
                    'num_widths': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]}] #,
                    #'num_widths': sp_randint(0, 5) } ]


    for params in param_grid:

        mkl = mkl_regressor()
        for i in xrange(16):
            params['wd' + str(i)] = uniform(scale=50, loc = median_wd)
            
        rs = RS(mkl, param_distributions = params, n_iter = 10, n_jobs = 24, cv = k, scoring = scorer)

        #try:
        rs.fit(X=data, y=labels)
        #except:
        #    sys.stderr.write("\n:>> Fitting Error:\n" )

    print "State: ", rs.random_state, "Widths: ", rs.estimator.widths

    print "Scores: ", rs.scores
    print "Average: ", rs.best_score_

    preds = rs.predict(data_t).tolist()

    print "R^2: ", scorer(rs.best_estimator_, data_t, labels_t)
    print "Parameters: ",  rs.best_params_

    pred, real = zip(*sorted(zip(preds, labels_t), key=lambda tup: tup[1]))


    g = Gnuplot.Gnuplot()   
    g.plot(Gnuplot.Data(pred, with_="lines"), Gnuplot.Data(real, with_="linesp") )

    

from modshogun import *
import Gnuplot, Gnuplot.funcutils
from numpy import *
from sklearn.metrics import r2_score
from mkl_regressor import *
from scipy.stats import randint as sp_randint
from sklearn.grid_search import  RandomizedSearchCV as RS
from scipy.stats import expon
import sys

from sklearn.cross_validation import cross_val_score

#def scorer(estimator, X, y):

#    predicted = estimator.predict(X)
#    return r2_score(predicted, y)


if __name__ == "__main__":

    labels = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/STS.gs.OnWN.txt")
    data = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/vectors_H10/pairs_eng-NO-test-2e6-nonempty_OnWN_d2v_H10_sub_m5w8.mtx").T

    labels_t = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/STS.gs.FNWN.txt")
    data_t = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/vectors_H10/pairs_eng-NO-test-2e6-nonempty_FNWN_d2v_H10_sub_m5w8.mtx").T

    k = 7
    p = 5
    distributions = {}
    for _ in xrange(p):
        distributions[str(_)] = expon

    """     self.degree = kwargs['degree']
            self.svm_C = kwargs['svm_c']
            self.mkl_C = kwargs['mkl_c']
            self.mkl_norm = kwargs['mkl_norm']
            self.svm_norm = kwargs['svm_norm']
            self.widths = kwargs['widths']
            self.kernel_weights = self.kernels.get_subkernel_weights()
            self.num_widths
"""


    param_grid = [ {'svm_c': expon(scale=100, loc=5), 
                    'mkl_c': expon(scale=100, loc=5),                    
                    'degree': sp_randint(0, 32), 
                    'widths': distributions,
                    'num_widths': sp_randint(0, 5) } ]


    for params in param_grid:

        mkl = mkl_regressor()
        rs = RS(mkl, param_distributions = params, n_iter = 10, n_jobs = 24, cv = k, scoring = 'mean_squared_error')

        #try:
        rs.fit(data, labels)
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

    

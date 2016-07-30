from modshogun import *
import Gnuplot, Gnuplot.funcutils
from numpy import *
from sklearn.metrics import r2_score
#from  file_names_util import *
from mkl_regressor import *
def set_weigth_bounds(median_distance):
    """0.02*mean, 12*mean    """
    return median_distance * 0.01, median_distance * 20.0

if __name__ == "__main__":

    labels = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/STS.gs.OnWN.txt")
    data = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/vectors_H10/pairs_eng-NO-test-2e6-nonempty_OnWN_d2v_H10_sub_m5w8.mtx")

    labels_t = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/STS.gs.FNWN.txt")
    data_t = loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/vectors_H10/pairs_eng-NO-test-2e6-nonempty_FNWN_d2v_H10_sub_m5w8.mtx")

    from sklearn.grid_search import RandomizedSearchCV as RS
    from scipy.stats import randint as sp_randint
    from scipy.stats import expon

    #data_items = shatter_file_name(args.x)

    k = 7
    N = 20
    #if args.M:
    #    min_wigth, max_weight = set_weigth_bounds(float(args.M))
    #else:
    p = 16
    min_wigth = 0.0009
    max_weight = 0.9
    
    param_grid = [ {'svm_c': expon(scale=100, loc=5),
                    'mkl_c': expon(scale=100, loc=5),
                    'degree': sp_randint(0, 24),
                  }]


    param_grid[0]['widths'] = []
    for s in xrange(p): # 16 weights maximun
        param_grid[s]['widths'] = []
        param_grid[s]['widths'].append(random.uniform(low = min_wigth, high = max_weight, size = s+1))
        if s < (p - 1):
            param_grid.append(param_grid[0])

    for i in xrange(N):
        for params in param_grid:

            mkl = mkl_regressor()
            rs = RS(mkl, param_distributions = params, n_iter = 15, n_jobs = 16, cv = k)
            rs.fit(data, labels)

        preds = rs.best_estimator_.predict(data_t)

        print "R^2: ", rs.best_estimator_.score(data_t, labels_t)
        print "Parameters: ",  rs.best_params_
        pred, real = zip(*sorted(zip(preds, labels_t), key=lambda tup: tup[1]))

        g = Gnuplot.Gnuplot()
        g.plot(Gnuplot.Data(pred, with_="lines"), Gnuplot.Data(real, with_="linesp") )

from modshogun import *
from numpy import *
from sklearn.metrics import r2_score
from scipy.stats import randint
from scipy.stats import randint as sp_randint
from scipy.stats import expon
from mkl_regressor import *

if __name__ == "__main__":

    import Gnuplot, Gnuplot.funcutils
    from sklearn.grid_search import RandomizedSearchCV as RS
    from argparse import ArgumentParser as ap

    parser = ap(description='This script trains/applies a SVR over any input dataset of numerical representations. The main aim is to determine a set of learning parameters')
    parser.add_argument("-x", help="Input file name (train vectors)", metavar="input_file", required=True)
    parser.add_argument("-y", help="""Regression labels file. Do not specify this argument if you want to uniauely predict over any test set. In this case, you must to specify
                                    the SVR model to be loaded as the parameter of the option -o.""", metavar="regrLabs_file", default = None)
    parser.add_argument("-X", help="Input file name (TEST vectors)", metavar="test_file", default = None)
    parser.add_argument("-Y", help="Test labels file.", metavar="testLabs_file", default = None)
    parser.add_argument("-n", help="Number of tests to be performed.", metavar="tests_amount", default=1)
    parser.add_argument("-o", help="""The operation the input data was derived from. Options: {'conc', 'convss', 'sub'}. In the case you want to give a precalculated center for
                                    width randomization, specify the number. e.g. '-o 123.654'. A filename can be specified, which is the file where a SVR model is sotred,
                                    e.g. '-o filename.model'""", metavar="operat{or,ion}")
    parser.add_argument("-u", help="Especify C regulazation parameter. For a list '-u C:a_b', for a value '-u C:a'.", metavar="fixed_params", default = None)
    parser.add_argument("-K", help="Kernel type custom specification. Uniquely valid if -u is not none.  Options: gaussian, linear, sigmoid.", metavar="kernel", default = None)
    parser.add_argument("-s", help="Toggle if you will process sparse input format.", action="store_true", default = False)
    parser.add_argument("-k", help="k-fold cross validation for the randomized search.", metavar="k-fold_cv", default=None)
    parser.add_argument("-p", help="Minimum number of basis kernels.", metavar="min_amount", default=2)
    parser.add_argument("-P", help="Maximum number of basis kernels.", metavar="max_amount", default=10)
    parser.add_argument("-m", help="Median width for generating width vectors.", metavar="median", default=0.01)
    args = parser.parse_args()

    labels = loadtxt(args.y) #loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/STS.gs.OnWN.txt")
    data = loadtxt(args.x) #loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/vectors_H10/pairs_eng-NO-test-2e6-nonempty_OnWN_d2v_H10_sub_m5w8.mtx")
    
    if args.X:
        labels_t = loadtxt(args.Y) #loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/STS.gs.FNWN.txt")
    if args.Y:
        data_t = loadtxt(args.X) #loadtxt("/almac/ignacio/data/sts_all/pairs-NO_2013/vectors_H10/pairs_eng-NO-test-2e6-nonempty_FNWN_d2v_H10_sub_m5w8.mtx")

    k = int(args.k)
    N = int(args.n)
    min_p = int(args.p)
    max_p = int(args.P)
    median_w = float(args.m)
    # median_width = None, width_scale = 20.0, min_size=2, max_size = 10, kernel_size = None
    print ">> Shapes: labels %s; Data %s\n\tlabelsT %s; DataT %s" % (labels.shape, data.shape, labels_t.shape, data_t.shape)
    params = {'svm_c': expon(scale=100, loc=5),
                    'mkl_c': expon(scale=100, loc=5),
                    'degree': sp_randint(0, 24),
                    #'widths': expon_vector(loc = median_w, min_size = min_p, max_size = max_p) 
                    'width_scale': [5, 10, 20, 30, 40, 50, 100],
                    'median_width': expon(scale=20, loc=median_w),
                    'kernel_size': [2, 3, 4, 5, 6, 7, 8, 9, 10]
                }
    param_grid = []
    for i in xrange(N):
        param_grid.append(params)

    for params in param_grid:
        mkl = mkl_regressor()
        rs = RS(mkl, param_distributions = params, n_iter = 20, n_jobs = 16, cv = k, scoring="mean_squared_error", verbose = 1)#"r2")
        rs.fit(data, labels)
        preds = rs.best_estimator_.predict(data_t)

        print "R^2: ", r2_score(preds, labels_t)
        print "Parameters: ",  rs.best_params_
        
        pred, real = zip(*sorted(zip(preds, labels_t), key=lambda tup: tup[1]))
        g = Gnuplot.Gnuplot()   
        g.plot(Gnuplot.Data(pred, with_="lines"), Gnuplot.Data(real, with_="linesp") )

    
from mkl_test import *
import pickle
from numpy import array

labels = array([2.0,0.0,2.0,1.0,3.0,2.0]).reshape((6, 1))
data = array([[1.0,2.0,3.0],[1.0,2.0,9.0],[1.0,2.0,3.0],[1.0,2.0,0.0],[0.0,2.0,3.0],[1.0,2.0,3.0]])
data_t = array([[1.5,2.1,3.5],[2.0,1.0,1.0],[0.5,2.2,0.0],[1.0,3.0,10.0]])
data_v = array([[0.0,0.001,0.35],[0.0,1.0,0.0],[0.5,20.2,1.0],[10.0,0.0,8.0]])
model_file = "/almac/ignacio/data/mkl_models/mkl_0.asc"

mkl = mkl_regressor(widths=array([2.0, 5.0, 10.0]), svm_c = 5.0, mkl_c = 5.0, mkl_norm = 2, degree = 0)
mkl.fit(data, labels)
mkl.save(model_file)

print "weights: ", mkl.kernel_weights
print "Estimated: ", mkl.predict(data)
print "Given: ", labels

print "predicted0: ", mkl.predict(data_t)

mkl2 = mkl_regressor()
mkl2.load(model_file)
print "Predicted:", mkl2.predict(data_t)
print "Validated:", mkl2.predict(data_v)

from matplotlib import *
from numpy import loadtxt
from scipy.signal import correlate
from matplotlib.pyplot import *
from ast import literal_eval

def read_results(file_name):
    o = open(file_name, 'r').readlines()
    outs = []
    for r in o:
        outs.append(literal_eval(r))
    return outs

label_file = "/home/iarroyof/data/labels_fnwn_test.dat"
output_file = "/home/iarroyof/data/output_test_fnwn_sub.txt"

labs = loadtxt(label_file)[:,0]
sample = range(0, len(labs))
est_o = read_results(output_file)[0]['estimated_output']
labels = sorted(zip(labs, sample), key = lambda tup: tup[0])

from pdb import set_trace

est_out = []
true = []

for i in labels:
    true.append(labs[i[1]])
    est_out.append(est_o[i[1]])

ccorr = correlate(true, est_out, mode = 'same')/len(labs)
#set_trace()
grid(True)
title("Semantic Similarity Regression []")
grid(True)
p1 = Rectangle((0, 0), 1, 1, fc="r")
p2 = Rectangle((0, 0), 1, 1, fc="b")
p3 = Rectangle((0, 0), 1, 1, fc="g")
#p4 = Rectangle((0, 0), 1, 1, fc="k")
#p5 = Rectangle((0, 0), 1, 1, fc="c")
legend((p1, p2, p3), ["True ordered relationship", "Predicted ordered output", "Cross correlation"], loc=4)
xlabel('Samples')
ylabel('Semantic Similarity Score')
#yscale('log')
plot(sample, true, color = 'r', linewidth=2)
plot(sample, est_out, color = 'b', linewidth=2)
plot(sample, ccorr, color = 'g', linewidth=2)
#plot(sample, labs, color = 'k', linewidth=2)
#plot(sample, est_o, color = 'c', linewidth=2)
show()



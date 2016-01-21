#from matplotlib import *
from numpy import loadtxt
from scipy.signal import correlate
from matplotlib.pyplot import *
from ast import literal_eval
from argparse import ArgumentParser

def read_results(file_name):
    out = open(file_name, 'r').readlines()
    outs = []
    
    for res in out:
        if res.startswith('{'):
            outs.append(eval(res.strip()))
        else:
            continue
    return outs

parsed = ArgumentParser(description='Plots desired labels, predicted outputs and calculates their corss-correlation.')
parsed.add_argument('-g', type=str, dest = 'goldStandard_file', help='Specifies the goldStandard file.')
parsed.add_argument('-p', type=str, dest = 'predictions_file', help='Specifies the machine predictions file.')
parsed.add_argument('-l', action='store_true', dest = 'log_scale', help='Toggles log scale for plotting.')
args = parsed.parse_args()
label_file = args.goldStandard_file #"/home/iarroyof/data/sts_test_13/STS.output.FNWN.txt"
output_file = args.predictions_file # "/home/iarroyof/data/output_1_sub_ccbsp_topic.txt"

labs = loadtxt(label_file)#[:,0]
sample = range(0, len(labs))
est_outs = []       

for est in read_results(output_file):
    est_outs.append(est['estimated_output'])
    
labels = sorted(zip(labs, sample), key = lambda tup: tup[0])

ordd_est_outs = []
true = []
est_out = []
ccorrs = []
for i in labels:
    true.append(labs[i[1]]) 
    
for out in est_outs:
    for i in labels:
        est_out.append(out[i[1]])
    ccorrs.append(correlate(true, est_out, mode = 'same')/len(labs))
    ordd_est_outs.append(est_out)
    est_out = []   

i = 0
for est_o in ordd_est_outs:
    figure()
    grid(True)
    title("Semantic Similarity Regression []")
    grid(True)
    p1 = Rectangle((0, 0), 1, 1, fc="r")
    p2 = Rectangle((0, 0), 1, 1, fc="b")
    p3 = Rectangle((0, 0), 1, 1, fc="g")
    legend((p1, p2, p3), ["True ordered relationship", "Predicted ordered output", "Cross correlation"], loc=4)
    xlabel('Samples')
    ylabel('Semantic Similarity Score')
    if args.log_scale:
        yscale('log')
        
    plot(sample, true, color = 'r', linewidth=2)
    plot(sample, est_o, color = 'b', linewidth=2)
    plot(sample, ccorrs[i], color = 'g', linewidth=2)
    i += 1
    show()


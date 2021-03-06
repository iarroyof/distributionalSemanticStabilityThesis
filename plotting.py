#from matplotlib import *
from numpy import loadtxt
from scipy.signal import correlate
from matplotlib.pyplot import *
from matplotlib import pyplot as pp
from ast import literal_eval
from argparse import ArgumentParser

def read_results(file_name):
    out = open(file_name, 'r').readlines()
    outs = []
#    set_trace()    

    for res in out:
        if res.startswith('{'):
            outs.append(literal_eval(res.strip()))
        else:
            continue
    if not outs: 
        print "not found or unparsed results"
        exit()
    return outs

def plotter(goldStandard_file, predictions_file, log_scale=False, number_result=None, same_figure=True):
    label_file = goldStandard_file #"/home/iarroyof/data/sts_test_13/STS.output.FNWN.txt"
    output_file = predictions_file # "/home/iarroyof/data/output_1_sub_ccbsp_topic.txt"

    from pdb import set_trace
    labs = goldStandard_file
    sample = range(0, len(labs))
    est_outs = []       

    for est in output_file:
        est_outs.append(est['estimated_output'])

    if len(labs) != len(est_outs[0]):
        print "Compared predicitons and goldStandard are not of the same length"
        print "len gs: ", len(labs), " vs len outs: ",  len(est_outs[0])
        exit()
    
    labels = sorted(zip(labs, sample), key = lambda tup: tup[0])

    ordd_est_outs = []
    true = []
    est_out = []
    ccorrs = []
 
    true = zip(*labels)[0]
    for out in est_outs:
        for i in labels:
            est_out.append(out[i[1]])
        ccorrs.append(correlate(true, est_out, mode = 'same')/len(labs))
        ordd_est_outs.append(est_out)
        est_out = []   
#set_trace()
    i = 0

    if number_result:
        grid(True)
        title("Semantic Similarity Regression []")
        grid(True)
        p1 = Rectangle((0, 0), 1, 1, fc="r")
        p2 = Rectangle((0, 0), 1, 1, fc="b")
        p3 = Rectangle((0, 0), 1, 1, fc="g")
        legend((p1, p2, p3), ["Gd_std sorted relationship", "Predicted sorted output", "Cross correlation"], loc=4)
        xlabel('GoldStandad sorted samples')
        ylabel('Semantic Similarity Score')
        if log_scale:
            yscale('log')
        
        plot(sample, true, color = 'r', linewidth=2)
        plot(sample, ordd_est_outs[number_result], color = 'b', linewidth=2)
        plot(sample, ccorrs[number_result], color = 'g', linewidth=2)
        show()
    else:	
        for est_o in ordd_est_outs:
            figure()
            grid(True)
            title("Semantic Similarity Regression []")
            grid(True)
            p1 = Rectangle((0, 0), 1, 1, fc="r")
            p2 = Rectangle((0, 0), 1, 1, fc="b")
            p3 = Rectangle((0, 0), 1, 1, fc="g")
            legend((p1, p2, p3), ["Gd_std sorted relationship", "Predicted sorted output", "Cross correlation"], loc=4)
            xlabel('GoldStandad sorted samples')
            ylabel('Semantic Similarity Score')
            if log_scale:
                yscale('log')
            plot(sample, true, color = 'r', linewidth=2)
            plot(sample, est_o, color = 'b', linewidth=2)
            plot(sample, ccorrs[i], color = 'g', linewidth=2)
            i += 1
            if same_figure:
                show()
       

        if not same_figure:
            show()

parsed = ArgumentParser(description='Plots desired labels, predicted outputs and calculates their corss-correlation.')
parsed.add_argument('-g', type=str, dest = 'goldStandard_file', help='Specifies the goldStandard file.')
parsed.add_argument('-p', type=str, dest = 'predictions_file', help='Specifies the machine predictions file.')
parsed.add_argument('-l', action='store_true', dest = 'log_scale', help='Toggles log scale for plotting.')
parsed.add_argument('-r', type=int, dest = 'number_result', help='If you know wath of all input results only to show, give it.')
parsed.add_argument('-s', action='store_true', dest = 'same_figure', help='Toggles plotting all loaded results in the same figure or each result in a different figure.')
args = parsed.parse_args()
label_file = args.goldStandard_file #"/home/iarroyof/data/sts_test_13/STS.output.FNWN.txt"
output_file = args.predictions_file # "/home/iarroyof/data/output_1_sub_ccbsp_topic.txt"

from pdb import set_trace
labs = loadtxt(label_file)
sample = range(0, len(labs))
est_outs = []       

for est in read_results(output_file):
    est_outs.append(est['estimated_output'])

if len(labs) != len(est_outs[0]):
    print "Compared predicitons and goldStandard are not of the same length"
    print "len gs: ", len(labs), " vs len outs: ",  len(est_outs[0])
    exit()
    
labels = sorted(zip(labs, sample), key = lambda tup: tup[0])

ordd_est_outs = []
true = []
est_out = []
ccorrs = []
 
true = zip(*labels)[0]
for out in est_outs:
    for i in labels:
        est_out.append(out[i[1]])
    ccorrs.append(correlate(true, est_out, mode = 'same')/len(labs))
    ordd_est_outs.append(est_out)
    est_out = []   

i = 0

if args.number_result:
    grid(True)
    title("Semantic Similarity Regression []")
    grid(True)
    p1 = Rectangle((0, 0), 1, 1, fc="r")
    p2 = Rectangle((0, 0), 1, 1, fc="b")
    p3 = Rectangle((0, 0), 1, 1, fc="g")
    legend((p1, p2, p3), ["Gd_std sorted relationship", "Predicted sorted output", "Cross correlation"], loc=4)
    xlabel('GoldStandad sorted samples')
    ylabel('Semantic Similarity Score')
    if args.log_scale:
        yscale('log')
        
    plot(sample, true, color = 'r', linewidth=2)
    plot(sample, ordd_est_outs[args.number_result], color = 'b', linewidth=2)
    plot(sample, ccorrs[args.number_result], color = 'g', linewidth=2)
    show()
else:	
    for est_o in ordd_est_outs:
        figure()
        grid(True)
        title("Semantic Similarity Regression []")
        grid(True)
        p1 = Rectangle((0, 0), 1, 1, fc="r")
        p2 = Rectangle((0, 0), 1, 1, fc="b")
        p3 = Rectangle((0, 0), 1, 1, fc="g")
        legend((p1, p2, p3), ["Gd_std sorted relationship", "Predicted sorted output", "Cross correlation"], loc=4)
        xlabel('GoldStandad sorted samples')
        ylabel('Semantic Similarity Score')
        if args.log_scale:
            yscale('log')
        plot(sample, true, color = 'r', linewidth=2)
        plot(sample, est_o, color = 'b', linewidth=2)
        plot(sample, ccorrs[i], color = 'g', linewidth=2)
        i += 1
        if args.same_figure:
            show()
       

    if not args.same_figure:
        show()


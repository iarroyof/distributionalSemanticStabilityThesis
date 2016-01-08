# -*- coding: utf-8 -*-
import numpy as np
import sys
import db_word_space as d_ws
from scipy import signal

def T(x1, x2, name="conv"):
    """ Convolutional (or any other from the ones defined) operation between two CiS vectors 
    
    x1, x2 - vectors
    name - name of the operation (sub, conc, conv, corr), defaults to convolution
    """
    if name.startswith("sub"):
        return x1 - x2
    if name.startswith("conc"):
        return np.concatenate((x1, x2))
    if name.startswith("conv"):
        return signal.fftconvolve(x1, x2)
    if name.startswith("corr"):
        return np.correlate(x1, x2)

def scsis(dws, sentence):
    """ Single Context summation inside a sentence
    
    Sum of all the word vectors in the sentence
    
    Recieves:
    dws - the db_word_space object to get the word vectors
    sentence - the sentence as a list of words
    """
    result = dws.word_vector(sentence.pop(0))
    for word in sentence:
        result = result + dws.word_vector(word)
    return result

def ccbsp(dws, s1, s2, name="conv"):
    """ Context Convolution between Sentence Pairs
    
    Applies the T operation to the scsis vectors of the sentences
    
    Recieves:
    dws - the db_word_space object to get the word vectors
    s1, s2 - sentences to be combined, as lists of words
    name - name of the operation, defaults to convolution
    """
    x1 = scsis(dws, s1)
    x2 = scsis(dws, s2)
    return T(x1, x2, name)

def csosp(dws, s1, s2, name="conv"):
    """ Context Summation on Sentence Pairs
    
    First performs Single Context Convolution between Sentence Pairs (SCCbSP), i.e. T operation between the rows 
    of the sentence matrices, then sums the resulting combined rows.
    
    Recieves:
    dws - the db_word_space object to get the word vectors
    s1, s2 - sentences to be combined, as lists of words
    name - name of the operation, defaults to convolution
    """
    res = None
    for i in range(1, dws.dimension+1):
        print "%s/%s"%(i,dws.dimension)
        row1 = dws.context_vector(s1, i)
        row2 = dws.context_vector(s2, i)
        if res is not None:
            res = res + T(row1, row2, name)
        else:
            res = T(row1, row2, name)
    return res

def read_sentence_pairs(filename, n=False):
    """ Generator to read sentence pairs from "filename"
    
    Pairs must be separated by tab (\t). Yields a 3-tuple consisting of:
    (the pair number (the line in the file), the first sentence, the second sentence)
    The sentences are returned as list of words.
    If n is specified, it yields only n pairs.
    """
    with open(input_file) as fin:
        for row_i,line in enumerate(fin):
            if not n is False and row_i == n:
                break
            s1, s2 = line.strip().split("\t")
            s1 = s1.lower().split()
            s2 = s2.lower().split()
            yield row_i, s1, s2    

if __name__ == "__main__":
    """Command line tool to read sentence pairs and generate combined output vectors file
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="input file name (sentence pairs)", metavar="input_file", required=True)
    parser.add_argument("-d", help="name of the database (word space)", metavar="database", required=True)
    parser.add_argument("-o", help="output file name (optional, defaults to output.mtx)", default="output.mtx", metavar="output_file")
    parser.add_argument("-l", help="limit to certain number of pairs (optional, defaults to whole file)", default=False, metavar="limit")
    parser.add_argument("-m", help="method, ccbsp or csosp (optional, defaults to ccbsp)", default="ccbsp", metavar="method")
    parser.add_argument("-t", help="combiner operation, can be corr, conv, conc, sub", metavar="operation", required=True) 
    args = parser.parse_args()
    dws = d_ws.db_word_space(args.d)
    input_file = args.f
    output_file = args.o
    limit = int(args.l) or False
    operation = args.t
    method_dict = {'ccbsp': ccbsp, 'csosp': csosp}
    method = method_dict[args.m]
    row = []
    col = []
    data = []
    for row_i, s1, s2 in read_sentence_pairs(input_file, limit):
        v = method(dws, s1, s2, operation)
        for col_i in range(0,len(v)):
            if v[col_i]:
                row.append(row_i)
                col.append(col_i)
                data.append(v[col_i])
    from scipy.sparse import csr_matrix
    from scipy import io
    m = csr_matrix((data, (row, col)))
    print m.get_shape()
    io.mmwrite(output_file, m)    

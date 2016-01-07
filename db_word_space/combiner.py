# -*- coding: utf-8 -*-
import numpy as np
import sys
import db_word_space as d_ws

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
        return np.convolve(x1, x2)
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
    """Script to read sentence pairs and generate output vectors file
    
    Modify the variables input_file and output_file as needed, as well
    as the variable "limit" to limit the number of sentence pairs to read
    """
    dws = d_ws.db_word_space("wden_sparse")
    input_file = "train/STS.input.MSRpar.txt"
    output_file = "output.mtx"
    limit = 2 #False if the whole file is to be analyzed
    operation = "sub" #Change to use another operation (Available: corr, conv, conc, and sub)
    method = ccbsp #can be ccbsp or csosp
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

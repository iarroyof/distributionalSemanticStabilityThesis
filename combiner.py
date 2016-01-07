# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import mmread
import sys

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

def scsis(S):
    """ Single Context summation inside a sentence
    
    S - Q x I sentence matrix where Q is vocabulary size and I is sentence length
    """
    return np.sum(S, axis=1)

def ccbsp(s1, s2, name="conv"):
    """ Context Convolution between Sentence Pairs
    
    s1, s2 - sentence matrices to be convolved
    name - name of the operation, defaults to convolution
    """
    x1 = scsis(s1)
    x2 = scsis(s2)
    return T(x1, x2, name)

def sccbsp(s1, s2, name="conv"):
    """ Single Context Convolution between Sentence Pairs
    
    s1, s2 - sentence matrices to be convolved
    name - name of the operation, defaults to convolution
    """
    rows = zip(s1, s2)
    def _T((row1, row2)):
        return T(row1, row2, name)
    X_rows = map(_T, rows)
    return np.vstack(X_rows)
    
def csosp(s1, s2, name="conv"):
    """ Context Summation on Sentence Pairs
    
    s1, s2 - sentence matrices to be convolved
    name - name of the operation, defaults to convolution
    """
    X = sccbsp(s1, s2, name)
    return np.sum(X, axis=0)

def read_sentence_pairs(file_name):
    from re import compile
    #sentence_pairs = []
    #WORD_RE = compile(r"\w+")
    with open(file_name) as p:
        lines = p.readlines()
        for line in lines:
            str_pair = line.split('\t')
            yield ((clean_Ustring(str_pair[0]).split(),
                    clean_Ustring(str_pair[1]).split()))

    #return sentence_pairs

def clean_Ustring(string):
    from unicodedata import name, normalize
    gClean = ''
    for ch in u''.join(string.decode('utf-8')):
        try:
            if name(ch).startswith('LATIN') or name(ch) == 'SPACE':
                gClean = gClean + ch
            else: # Remove non-latin characters and change them by spaces
                gClean = gClean + ' '
        except ValueError: # In the case name of 'ch' does not exist in the unicode database.
            gClean = gClean + ' '

    return normalize('NFKC', gClean.lower()) # Return the unicode normalized document.


    return sentence_pairs
if __name__ == "__main__":
    from pdb import set_trace as st
    import hash_word_space as h_ws
    from numpy import vstack, empty
    from scipy.sparse import csr_matrix
    from scipy.io import mmwrite

    pairs_file = "/home/iarroyof/data/sts_trial/STS.input.txt"
    hws_file = "/home/iarroyof/data/wdEn_sparse.txt"

    hws = h_ws.hash_word_space(hws_file)
    #s1 = mmread(sys.argv[1])
    #s2 = mmread(sys.argv[2])
    #print ccbsp(s1, s2, "conv")
    #print csosp(s1, s2, "conv")
    sentence_wl_0=empty((0,hws.dimension),float)
    sentence_wl_1=empty((0,hws.dimension),float)
    for sentence in read_sentence_pairs(pairs_file):
        #st()
        for word in sentence[0]:
            sentence_wl_0 = vstack((sentence_wl_0, hws.word_vector(word)))
        for word in sentence[1]:
            sentence_wl_1 = vstack((sentence_wl_1, hws.word_vector(word)))

    mmwrite('xxx.mtx',csr_matrix(sentence_wl_0))
    mmwrite('xxx_1.mtx',csr_matrix(sentence_wl_1))
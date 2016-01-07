# -*- coding: utf-8 -*-

from ast import literal_eval
from numpy import array, zeros
import re
from sys import stderr
class hash_word_space(object):
    """ This object loads a hashed sparse matrix from a file which is called "file_name". Such a file should meet
    the format especified in 'pipeLine.pdf'. You can get a 'word_vector' at once by following the example below:
        
        import hash_word_space as hws # copy the module to your Python workPath before importing.
        file_name = '/the/path/of/wdSp_sparse.txt'
        
        h = hws(file_name)          # Instantiate the object by specifying the input file.
        wrd = u'ingeniería'         # We will look for an unicode word, for instance.  
        word_vector = h.get_word_vector(wrd)
        print word_vector[1528]     # 'word_vector' is a numpy.array object (a dense vector) and we can index its
                                    # elements
    
    """

    def __init__(self, file_name):
        self.sparseDic = {}
        with open(file_name, 'r') as f:
            for line in f:
                item = line.strip().split('\t')
                replaced = u''.join((self.unicodize(seg)
                                     for seg in re.split(r'(\\u[0-9a-f]{4})', item[0])))
                key = u''.join((c for c in replaced if c != '"'))
                try:
                    self.sparseDic[key] = list(literal_eval(item[1]))
                except IndexError:
                    stderr.write('Probable error format in input sparse data file.\n')
                    exit()

           
    def unicodize(self, segment):
        if re.match(r'\\u[0-9a-f]{4}', segment):
            return segment.decode('unicode-escape')
    
        return segment.decode('utf-8')

    
    def word_vector(self, word):
        """ Use this method for retrieving a numpy.array object (dense real row vector) associated to a
        desired 'word'. A vector of zeros is returned if the word isn't in the dictionary.
        """ 
        wordvector = zeros(self.dimension)
        try: 
            dic = self.sparseDic[word]            
        except KeyError:
            return wordvector
            #print 'There no exists such a desired KEY... You will be straightway expeled!'
            #exit()
        
        for item in dic:
            wordvector[item[0]] = item[1]
        
        return wordvector
        
        

    @property
    def sparseDic(self):
        """ The dictionary where the sparse matrix is loaded from file.
        """        
        return self.__sparseDic
    
    @property
    def dimension(self):
        """ The dimension of the word space.
        """
        self.__dimension = max((max(i) for i in self.sparseDic.values()))[0] + 1
        return self.__dimension

    @sparseDic.setter
    def sparseDic(self, value):
        """ You can modify the dictionary if an isolated case requires. It is better deleting the 
        'hash_word_space' object and instantiate it again, after the input file has been modificated assert
        required.
        """        
        assert isinstance(value, dict)
        self.__sparseDic = value


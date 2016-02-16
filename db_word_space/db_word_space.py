# -*- coding: utf-8 -*-

from numpy import array, zeros
import re
import psycopg2

class db_word_space(object):
    """This object gets word vectors from a database, which has to be created previousy using the
    db_store.py script
    
    Example of use:
    import db_word_space as dbs
    space = dbs.db_word_space("wden_sparse") #Create the object passing the name of the database
    space.word_vector("rainbow") #get the word vector
    """
        
    def __init__(self, database):
        self.conn = psycopg2.connect("dbname=%s user=semeval password=semeval"%database)
    
    def close_conn(self):
        """ Use this method when you're done to close the connection to the database """
        self.conn.close()
    
    def word_vector(self, word):
        """ Use this method for retrieving a numpy.array object (dense real row vector) associated to a
        desired 'word'. A vector of zeros is returned if the word isn't in the dictionary.
        """ 
        wordvector = zeros(self.dimension + 1)       
        cr = self.conn.cursor()
        cr.execute("select pos, val from word_list t1 join word_sparse t2 on t1.id=t2.word_id where t1.word=%s", (word,))
        
        rows = cr.fetchall()
        if not rows:
            return wordvector
        
        for row in rows:
            wordvector[row[0]] = row[1]
        
        return wordvector

    def context_vector(self, sentence, pos):
        """ Retrieves a vector of contexts at a given position given a sentence.
        
        Whereas the word_vector method returns all the positions for a given word, this one returns
        a specific position of a specific sequence of words, i.e. a row of the sentence matrix
        instead of a column word vector.
        """
        row = zeros(len(sentence))
        cr = self.conn.cursor()
        for i,word in enumerate(sentence):
            cr.execute("select val from word_list t1 join word_sparse t2 on t1.id=t2.word_id where t2.pos=%s and t1.word=%s", (pos,word))
            res = cr.fetchone()
            if res:
                row[i] = res[0]
        return row
        
    @property
    def dimension(self):
        """ The dimension of the word space.
        """
        cr = self.conn.cursor()
        cr.execute("select max(pos) from word_sparse")
        self.__dimension = cr.fetchone()[0]
        return self.__dimension


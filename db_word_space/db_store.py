# -*- coding: utf-8 -*-
#Script for storing the sparse data into a database
import psycopg2
import re
import argparse

def unicodize(segment):
    if re.match(r'\\u[0-9a-f]{4}', segment):
        return segment.decode('unicode-escape')
    return segment.decode('utf-8')

def create_tables(cr):
    cr.execute("create table word_list(id serial primary key, word character varying not null)")
    cr.execute("""create table word_sparse(
        id serial primary key, 
        word_id integer references word_list(id) not null,
        pos integer not null,
        val float not null)""")

def delete_tables(cr):
    cr.execute("drop table word_sparse")
    cr.execute("drop table word_list")

def store_words(cr, file_name):
    with open(file_name) as f:
        for line in f: 
            item = line.strip().split('\t')
            replaced = u"".join((unicodize(seg) for seg in re.split(r'(\\u[0-9a-f]{4})', item[0])))
            key = u''.join((c for c in replaced if c != '"'))
            
            cr.execute("insert into word_list(word) values(%s) returning id", (key,))
            word_id = cr.fetchone()[0]

            #Parse the list, literal_eval is avoided because of memory issues
            inside = False
            number = ""
            pos = 0
            val = 0
            for c in item[1]:
                if c == '[':
                    inside = True
                elif c.isdigit():
                    number += c
                elif c == ',':
                    if inside:
                        pos = int(number)
                        number = ""
                elif c == ']':
                    if inside:
                        val = int(number)
                        number = ""
                        cr.execute("insert into word_sparse(word_id, pos, val) values (%s, %s, %s)", (word_id, pos, val))
                    inside = False
    
if __name__ == "__main__":
    """
    Stores words in the database.
    
    The first time, run with the arguments -cs.
    If the database has to be recreated, run again with the d argument (-dcs)
    
    Use the -f argument to specify the input file (sparse data)
    Use the -n argument to specify the database name, which must be already created.
    
    It also asumes the owner of the database is a user named semeval with password semeval
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="file_name")
    parser.add_argument("-n", help="database name")
    parser.add_argument("-d", help="delete tables", action="store_true")
    parser.add_argument("-c", help="create tables", action="store_true")
    parser.add_argument("-s", help="store words", action="store_true")
    args = parser.parse_args()
    file_name = args.f
    conn = psycopg2.connect("dbname=%s user=semeval password=semeval"%args.n)
    cr = conn.cursor()
    if args.d:
        delete_tables(cr)
    if args.c:
        create_tables(cr)
    if args.s:
        store_words(cr, file_name)
    conn.commit()
    conn.close()

from re import search, M, I
import sys

def shatter_file_name():

    try:
        source = search(r"(?:vectors|pairs)_([A-Za-z]+[\-A-Za-z0-9]+)_?(T[0-9]{2,3}_C[1-9]_[0-9]{2}|d\d+t[-\d]*)?_([d2v|w2v|coocc\w*|doc\w*]*)_(H[0-9]{1,4})_?([sub|co[nvs{0,2}|rr|nc]+]?)?_(m[0-9]{1,3}[_w?[0-9]{0,3}]?)", args.x, M|I)
    # example filename: 'pairs_headlines13_T01.._d2v_H300_conc_m5.mtx'
        if args.c:     #           1        2*    3    4   5*  6  
            corpus = args.c
        else:
            corpus = source.group(1)
        if args.r:
            representation = args.r
        else:
            representation = source.group(3)
        if args.d:
            dimensions = args.d
        else:
            dimensions = source.group(4)[1:]
        if args.m:
            min_count = args.m
        else:
            min_count = source.group(6)[1:]
    except IndexError:
        sys.stderr.write(print "\nError in the filename. One or more indicators are missing. Notation: <vectors|pairs>_<source_corpus>_<model_representation>_<dimendions>_<operation>*_<minimum_count>.mtx\n")
        for i in range(6):
            try:
                sys.stderr.write("%s" % source.group(i))
            except IndexError:
                sys.stderr.write(":>> Unparsed: %s" % (i))
                pass
        exit()
    except AttributeError:
        sys.stderr.write("\nFatal Error in the filename. Notation: <vectors|pairs>_<source_corpus>_<model_representation>_<dimendions>_<operation>*_<mminimum_count>.mtx\n")
        for i in range(6):
            try:
                sys.stderr.write("%s" % source.group(i))
            except AttributeError:
                sys.stderr.write(":>> Unparsed: %s" % (i))
                pass            
        exit()
    

    return  {'corpus': corpus, 
                'representation': representation, 
                'dimensions': dimensions, 
                'min_count': min_count, 
                'source': source.group(2), 
                'operation': source.group(5)}

# Distributional Semantics Stability and Consistence
This repo contains codes and, selfcontained, some documentation about implementation of theoretical issues of my PhD thesis on Natural Language Processing (NLP). This work is entitled "Learning Kernels for Distributional Semantic Clustering" in which we present a novel (distributional) semantic embedding method, which aims at consistently performing semantic clustering at sentence level. Taking into account special aspects of Vector Space Models (VSMs), we propose to learn reproducing kernels in classification tasks. By this way, capturing spectral features from data is possible. These features make it theoretically plausible to model semantic similarity criteria in Hilbert spaces, i.e. the embedding spaces where stability and consistency of semantic similarity measures can be ensured (based on Tikhonov-Vapnik's theory). We could improve the semantic assessment over embeddings, which are criterion-derived representations from traditional semantic vectors. The learned kernel could be easily transferred to clustering methods, where the Multi-Class Imbalance Problem is considered (e.g. semantic clustering of definitions of terms).

See at my personal web site the initial publication for futher details: www.corpus.unam.mx:8069/member/1

# Usage

We have made some tests with toy datasets. At the moment we have parallelized the tool in a local machine with multiple CPUs. 

-- Dependencies --

Ubunbtu 14.04 - The Operating system we developed this tool

modshogun - The Shogun Machine Learning Toolbox

scipy - The scientific Python stack

-- Command line --

We need excecuting the following piped command in the Ubuntu shell (be sure all files are in the same system file path where you excecute the command):

`$ python gridGen.py -f gridParameterDic.txt -t 5 | ./mklParallel.sh | python mklReducer.py > o.txt`

The `gridGen.py` script writes, e.g., 5 paramemter search paths to the stdout separately. These paths are randomly generated from the  file `gridParameterDic.txt` which contains a dictionary of parameters. The set of paths is read by the `mklParallel.sh` bash script who yields multiple processing jobs (The OS manages the thing if there are more processes than machine cores). Until the results of these jobs are writen at all to stdout, the `mklReducer.py` script reads them, prints them and emites the one with the maximum performace. These results are subsequenly writen either to an output file, e.g. `o.txt`, or to stdout (if the last one is not specified).

After results are emited, we can use the resulting best path as parameter set for performing consistet inner products between pairs of input word/prhase/sentence (WPS) vectors. 

# Current idea

According to the approach given in SemEval2015-task 1, we are currently posing our problem as a two class one. The class 1 corresponds to similar sentences and the class 0 (-1) to dissimilar ones. A similarity criteria is learned from labels by the multikernel machine during training. Given that this machine classifies single vectors (but not pairs of them), we propose different sentence vector combination schemata, i.e. `sentence_pair_vector = combine_pair{combine_sa(word_vector_1, word_vector_2,...), combine_sb(word_vector_1, word_vector_2,...)}`, where `sentence_pair_vector` is associated to a label in `{0, 1}` and feed to the miltikernel machine. We estimate the `combine_pair{}` operation allows holding dissimilarity features between `combine_sa()` and `combine_sb()` vectors, so the multikernel machine will filter them.

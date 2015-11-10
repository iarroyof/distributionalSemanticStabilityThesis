# Distributional Semantics Stability and Consistence
This repo contains codes and, selfcontained, some documentation about implementation of theoretical issues of my PhD thesis on Natural Language Processing (NLP). This work is entitled "Learning Kernels for Distributional Semantic Clustering" in which we present a novel (distributional) semantic embedding method, which aims at consistently performing semantic clustering at sentence level. Taking into account special aspects of Vector Space Models (VSMs), we propose to learn reproducing kernels in classification tasks. By this way, capturing spectral features from data is possible. These features make it theoretically plausible to model semantic similarity criteria in Hilbert spaces, i.e. the embedding spaces where consistency of semantic similarity measures and stability of algorithms implementing them can be ensured. We could improve the semantic assessment over embeddings, which are criterion-derived distributional representations from traditional semantic vectors (e.g. windowed cooccurrence counting). The learned kernel could be easily transferred to clustering methods, where the Class Imbalance Problem is considered (e.g. semantic clustering of polysemic definitions of terms. Their multimple meanings are Zipf distributed: Passonneau, et.al, 2012).

See the initial publications at my personal web site for futher details: www.corpus.unam.mx:8069/member/1

# Usage

We have made some tests with toy datasets. At the moment we have parallelized the tool in a local machine with multiple CPUs. 

-- Dependencies --

Ubunbtu 14.04 - The Operating system we developed this tool

modshogun - The Shogun Machine Learning Toolbox

scipy - The scientific Python stack

-- Command line --

We need executing the following piped command in the Ubuntu shell (be sure all files are in the same system file path where you execute the command):

`$ python gridGen.py -f gridParameterDic.txt -t 5 | ./mklParallel.sh | python mklReducer.py [-options] > o.txt`

The `gridGen.py` script writes, e.g., 5 parameter search paths to the `stdout` separately. These paths are randomly generated from the  file `gridParameterDic.txt` which contains a python dictionary of parameters. The set of paths is read by the `mklParallel.sh` bash script who yields multiple training jobs (The OS manages the thing if there are more processes than machine cores). Until the results of these jobs are written at all to stdout, the `mklReducer.py` script reads them, prints them and emits the one with the maximum performance (command `python mklReducer.py -h` for seeing `[-options]`). These results are subsequently written either to an output file, e.g. `o.txt`, or to the stdout (if the last one is not specified by `>`).

After results are emitted, we can use the resulting best path as parameter set for performing inner products between pairs of input word/phrase/sentence (WPS) vectors. These pairwise products are easily conceived as consistent semantic similarity scores.

# Current idea

According to the approach given in SemEval2015-task 1, we are currently posing our problem as a two class one. The class 1 corresponds to similar sentence pairs and the class 0 (-1) to dissimilar ones (`{s_a, s_b}`). A similarity criterion is learned from labels by the multikernel machine during training. Given that this machine classifies single vectors (but not pairs of them), we propose different sentence vector combination schemata, i.e. `sentence_pair_vector = combine_pair{combine_sa(word_vector_1, word_vector_2,...), combine_sb(word_vector_1, word_vector_2,...)}`, where `sentence_pair_vector` is associated to a label in `{0, 1}` and fed to the multikernel machine. We estimate the `combine_pair{}` operation allows holding dissimilarity features between `combine_sa()` and `combine_sb()` vectors (`s_a` sentence and `s_b` sentence, respectively), so the multikernel machine will filter them. Constituent word vectors `word_vector_i` of each sentence are simply window co-occurrence counting.

We are also planning using morph and PMI vectors as alternative inputs to our combination schemata. A binary NBayes classifier as well as a logistic output neural network are considered as baseline algorithms for addressing the task. 

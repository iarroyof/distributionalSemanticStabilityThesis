# Distributional Semantics Stability and Consistence
This repo contains codes and, selfcontained, some documentation about implementation of theoretical issues of my PhD thesis on Natural Language Processing (NLP). This work is entitled "Learning Kernels for Distributional Semantic Clustering" in which we present a novel (distributional) semantic embedding method, which aims at consistently performing semantic clustering at sentence level. Taking into account special aspects of Vector Space Models (VSMs), we propose to learn reproducing kernels in classification tasks. By this way, capturing spectral features from data is possible. These features make it theoretically plausible to model semantic similarity criteria in Hilbert spaces, i.e. the embedding spaces where stability and consistency of semantic similarity measures can be ensured (based on Tikhonov-Vapnik's theory). We could improve the semantic assessment over embeddings, which are criterion-derived representations from traditional semantic vectors. The learned kernel could be easily transferred to clustering methods, where the Multi-Class Imbalance Problem is considered (e.g. semantic clustering of definitions of terms).

See at my personal web site the initial publication for futher details: http://describe.com.mx/~iarroyof/

# Usage

We have made some tests with toy datasets. At the moment we have paralelized the tool in a local machine. 

-- Dependencies --

Ububtu 14.04 - The Operating system we developed this tool
modshogun - The Shogun Machine Learning Toolbox
scipy - The scientific Python stack

-- Command line --

We need excecuting the following command in the Ubuntu shell:

$ python gridGen.py -f gridParameterDic.txt -t 5 | ./mklParallel.sh | python mklReducer.py > o.txt

The `gridGen.py` script writes, e.g., 5 paramemter search paths to the stdout separately. These paths are randomly generated from the  file `gridParameterDic.txt` which contains a ditionary of parameters. The set of paths is read by the `mklParallel.sh` bash script who yields multiple processing jobs. Until the results of these jobs are writen to stdout, the `mklReducer.py` script read them, print them and calculates the one with the maximum performace. These results are subsequenly writen either to an output file, e.g. `o.txt`, or to stdout.

After results are emited, we can use the resulting path for being our consistency inner product between pairs of input vectors.

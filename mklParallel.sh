#!/usr/bin/env bash
#Author: multiple fora :)

# The input file (if specified) must be terminated by a empty line (the EOF in bash)
if [ "$1" ] ; then exec < "$1" ; fi # verifying if there is a specified file in stdin

while IFS='' read -r path || [[ -n "$path" ]]; do # load in $path a line from the input paths file
    #(python /home/iarroyof/printLine.py -l "$path") #test file
    #(python mklCall.py -p "$path" ) & # Execute in parallel as many (subshell) mklCalls as paths in the input file
    (python mklCall.py -p "$path" >> $HOME/data/mkl_output_headlines_d2v_conc_u5_30_.txt) &
done

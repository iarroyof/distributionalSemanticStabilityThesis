#!/usr/bin/env bash
#Author: multiple fora :)
if [ "$1" ] ; then exec < "$1" ; fi

while IFS='' read -r path || [[ -n "$path" ]]; do
    #(python /home/iarroyof/printLine.py -l "$path") #test file
    (python mklCall.py -p "$path") &
done
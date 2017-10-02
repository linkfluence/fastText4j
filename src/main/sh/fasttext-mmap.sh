#!/bin/bash

DEPS="";
for file in lib/*
do
    if [[ -f $file ]]; then
        DEPS=$DEPS:$file
    fi
done

java -cp $DEPS fasttext.FastText $*
#!/usr/bin/bash

mkdir "../bin" # create landing directory for driver programs
mkdir "../lib" # create landing directory for library

# move the gen feats into bin
cd $MLADP/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/src/util/nedc_mladp_gen_feats/
make

# move the eval eval into bin
cd ../nedc_mladp_eval
make

# move the train into bin
cd ../nedc_mladp_train
make

# move run into bin
cd ../nedc_mladp_run
make

# ann tools into lib
cd ../../functs/nedc_mladp_ann_tools
make

# feats tools into lib
cd ../nedc_mladp_feats_tools
make

# geometry tools into lib
cd ../nedc_mladp_geometry_tools
make

# eval tools into lib
cd ../nedc_mladp_eval_tools
make

# fileio toosl into lib
cd ../nedc_mladp_fileio_tools
make

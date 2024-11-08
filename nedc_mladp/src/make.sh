#!/usr/bin/bash
mkdir "../bin" # create landing directory for driver programs
mkdir "../lib" # create landing directory for library

# move the gen feats into bin
cd $MLADP/nedc_mladp/src/util/nedc_mladp_gen_feats/
make

# move the eval eval into bin
cd ../nedc_mladp_gen_preds
make

# move the train into bin
cd ../nedc_mladp_train_model
make

# move run into bin
cd ../nedc_mladp_run
make

# move run into bin
cd ../nedc_mladp_gen_graphics
make

cd $MLADP/nedc_mladp/src/functs/nedc_mladp_feats_tools/
make

# geometry tools into lib
cd ../nedc_mladp_geometry_tools
make

# eval tools into lib
cd ../nedc_mladp_pred_tools
make

# fileio toosl into lib
cd ../nedc_mladp_fileio_tools
make

# fileio toosl into lib
cd ../nedc_mladp_label_enum
make

# models
cd ../nedc_mladp_models
make

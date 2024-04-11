#!/usr/bin/env python
#
# file: /data/isip/exp/tuh_dpath/exp_0280/nedc_mladp/src/util/nedc_train_model/train_model.py
#
# revision history:
#
# 
#
# This is a Python version of the C++ utility nedc_print_header.
#------------------------------------------------------------------------------

"""! @brief Clasifies the DCT values into different labeled regions"""
##
# @file classify_dct_values.py
#
# @brief Classify the DCT values into different labeled regions
#
# 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier as RNF
import os
import joblib

import nedc_mladp_fileio_tools as fileio_tools

#picone
import nedc_file_tools

def main():

    # set argument parsing
    #
    args_usage = "nedc_mladp_train_model.usage"
    args_help = "nedc_mladp_train_model.help"
    parameter_file = fileio_tools.parameters_only_args(args_usage,args_help)

    # parse parameters
    #
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"train_model")
    model_type=parsed_parameters['model_type']
    feature_data_list=parsed_parameters['data_list']
    output_model_directory=parsed_parameters['model_output_path']
    compression=int(parsed_parameters['compression'])
    
    # set the list of datapoints to all the files within that directory
    #
    train_list = fileio_tools.read_file_lists(feature_data_list)

    # parse the annotations
    #
    labels,mydata,unused1,unused2 = fileio_tools.read_feature_files(train_list)
    
    # Fit the model
    #
    model = None
    if model_type == "QDA":
        model = QDA()
    elif model_type == "RNF":
        model = RNF()
    else:
        print("No model supplied")
        return
    model.fit(mydata, labels)

    # change the directory to output the model
    #
    os.chdir(output_model_directory)

    # dump the model there
    #
    joblib.dump(model,'Trained_'+model_type+'.joblib',compress=compression)

    

if __name__ == "__main__":
    main()

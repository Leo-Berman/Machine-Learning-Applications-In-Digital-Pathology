#!/usr/bin/env python3
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
import sys

sys.path.append("../lib")
import nedc_fileio

#picone
import nedc_file_tools

def main():

    # set argument parsing
    #
    args_usage = "usagefiles/train_model_usage.txt"
    args_help = "helpfiles/train_model_help.txt"
    parameter_file = nedc_fileio.parameters_only_args(args_usage,args_help)

    # parse parameters
    #
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"train_model")
    model_type=parsed_parameters['model_type']
    input_data_directory=parsed_parameters['data_directory']
    output_model_directory=parsed_parameters['model_output_path']
    compression=int(parsed_parameters['compression'])
    
    # change directory to the appropriate train data file
    #
    os.chdir(input_data_directory)
    
    # set the list of datapoints to all the files within that directory
    #
    train_list = os.listdir()

    # parse the annotations
    #
    labels,mydata,unused1,unused2 = nedc_fileio.read_feature_files(train_list)
    
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
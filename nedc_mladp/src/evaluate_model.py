#!/usr/bin/env python3
"""! @brief Clasifies the DCT values into different labeled regions"""
##
# @file classify_dct_values.py
#
# @brief Classify the DCT values into different labeled regions
#
# 
import numpy as np
import os
import joblib
import csv
import sys

sys.path.append("../lib")
import nedc_fileio

#picone
import nedc_file_tools

def main():

    # set argument parsing
    #
    args_usage = "gen_feats_usage.txt"
    args_help = "gen_feats_help.txt"
    parameter_file = nedc_fileio.parameters_only_args(args_usage,args_help)

    # parse parameters
    #
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"eval_model")
    input_data_directory=parsed_parameters['data_directory']
    model_path=parsed_parameters['model']
    
    # load the model
    #
    model = joblib.load(model_path)        
    
    # change directory to the appropriate train data file
    #
    os.chdir(input_data_directory)
    
    # set the list of datapoints to all the files within that directory
    #
    train_list = os.listdir()

    # lists for holding the labels and data
    #
    mydata = []
    labels = []

    # iterate through the entire training list
    #
    for x in train_list:

        # rea
        with open (x) as file:
            reader = csv.reader(file)
            next(reader,None)
            for row in reader:
                row_list = list(row)
                labels.append(row_list.pop(0))
                mydata.append([float(x) for x in row_list])

    # reshape the arrays
    #
    labels = np.array(labels).ravel()
    mydata = np.array(mydata)
    
    
    print(model.score(mydata,labels))

    

if __name__ == "__main__":
    main()
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
    args_usage = "train_model_usage.txt"
    args_help = "train_model_help.txt"
    parameter_file = nedc_fileio.parameters_only_args(args_usage,args_help)

    # parse parameters
    #
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"train_model")
    model_type=parsed_parameters['model_type']
    input_data_directory=parsed_parameters['data_directory']
    output_model_directory=parsed_parameters['model_output_path']
    
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
    print(model.score(mydata,labels))

    # change the directory to output themodel
    #
    os.chdir(output_model_directory)

    # dump the model there
    #
    joblib.dump(model,'Trained_'+model_type+'.joblib')

    

if __name__ == "__main__":
    main()
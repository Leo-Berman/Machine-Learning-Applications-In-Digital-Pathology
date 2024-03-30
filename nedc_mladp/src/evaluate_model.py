#!/usr/bin/env python3
"""! @brief Clasifies the DCT values into different labeled regions"""
##
# @file classify_dct_values.py
#
# @brief Classify the DCT values into different labeled regions
#
# 
import os
import joblib
import sys

sys.path.append("../lib")
import nedc_fileio
import nedc_model_metrics

#picone
import nedc_file_tools

def main():

    # set argument parsing
    #
    args_usage = "usagefiles/evaluate_model_usage.txt"
    args_help = "helpfiles/evaluate_model_help.txt"
    parameter_file = nedc_fileio.parameters_only_args(args_usage,args_help)

    # parse parameters
    #
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"eval_model")
    input_data_directory=parsed_parameters['data_directory']
    model_path=parsed_parameters['model']
    generate_confusion_matrix=int(parsed_parameters['confusion_matrix'])
    confusion_matrix_path=parsed_parameters['output_graphics_path']
    generate_decisions=int(parsed_parameters['decisions'])

    # load the model
    #
    model = joblib.load(model_path)        
    
    # change directory to the appropriate train data file
    #
    os.chdir(input_data_directory)
    
    # set the list of datapoints to all the files within that directory
    #
    train_list = os.listdir()
    labels,mydata,frame_locations,framesizes = nedc_fileio.read_feature_files(train_list)
    
    # generate confusion matrix
    #
    if generate_confusion_matrix == 1:
        nedc_model_metrics.plot_confusion_matrix(model,labels,mydata,confusion_matrix_path)
    
    # to be implimented but this will draw the decision on an image
    #
    if generate_decisions == 1:
        pass

    # print the error rate and mean confidence %
    #
    print("Error rate = ",model.score(mydata,labels))
    print("Mean confidence %",nedc_model_metrics.mean_confidence(model,mydata))
    

if __name__ == "__main__":
    main()
#!/usr/bin/env python
#
# file: /data/isip/exp/tuh_dpath/exp_0280/nedc_mladp/src/util/nedc_mladp_evaluate_model/nedc_evaluate_model.py
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
import joblib

import nedc_fileio_tools as fileio_tools
import nedc_mladp_eval_tools as eval_tools

#picone
import nedc_file_tools

def main():

    # set argument parsing
    #
    args_usage = "nedc_mladp_eval.usage"
    args_help = "nedc_mladp_eval.help"
    parameter_file = fileio_tools.parameters_only_args(args_usage,args_help)

    # parse parameters
    #
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"eval_model")
    feature_data_list=parsed_parameters['data_list']
    model_path=parsed_parameters['model']
    generate_confusion_matrix=int(parsed_parameters['confusion_matrix'])
    confusion_matrix_path=parsed_parameters['output_graphics_path']
    
    # only a single image applies
    #
    generate_decisions=int(parsed_parameters['decisions'])
    decisions_path=parsed_parameters['output_decisions_path']
    
    # load the model
    #
    model = joblib.load(model_path)        
    
    feature_files_list = fileio_tools.read_file_lists(feature_data_list)

    labels,mydata,frame_locations,framesizes = fileio_tools.read_feature_files(feature_files_list)
    
    # generate confusion matrix
    #
    if generate_confusion_matrix == 1:
        eval_tools.plot_confusion_matrix(model,labels,mydata,confusion_matrix_path)

    # generates a list of guess and their top level coordinates only applies to single image
    # 
    if generate_decisions == 1:
        eval_tools.plot_decisions(model,mydata,decisions_path,frame_locations,framesizes)

    # print the error rate and mean confidence %
    #
    print("Error rate = ",model.score(mydata,labels))
    print("Mean confidence %",eval_tools.mean_confidence(model,mydata))
    

if __name__ == "__main__":
    main()
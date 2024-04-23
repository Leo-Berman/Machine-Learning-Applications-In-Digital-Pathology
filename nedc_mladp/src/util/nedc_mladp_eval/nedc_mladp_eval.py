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

import nedc_mladp_fileio_tools as fileio_tools
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

    run_params = nedc_file_tools.load_parameters(parameter_file,"gen_feats")
    if run_params['run']==1:
        feature_data_list=run_params['output_list']
        model_path = nedc_file_tools.load_parameters(parameter_file,"train_model")['model_output_path']
    
    # only a single image applies
    #
    generate_decisions=int(parsed_parameters['decisions'])
    decisions_path=parsed_parameters['output_decisions_path']
    generate_histogram = int(parsed_parameters['generate_histogram'])                          
    even_data = int(parsed_parameters['even_data'])
    histogram_output=parsed_parameters['hist_out']
    # load the model
    #
    model = joblib.load(model_path)        
    
    feature_files_list = fileio_tools.read_file_lists(feature_data_list)

    labels,mydata,frame_locations,framesizes = fileio_tools.read_feature_files(feature_files_list)

    print("before = ",len(mydata),len(labels))
    
    # even the data out
    #
    if even_data == 1:
        mydata,labels = eval_tools.even_data(mydata,labels)

    print("after = ",len(mydata),len(labels))
    # generate confusion matrix
    #
    if generate_confusion_matrix == 1:
        eval_tools.plot_confusion_matrix(model,labels,mydata,confusion_matrix_path)

    # generates a list of guess and their top level coordinates only applies to single image
    # 
    if generate_decisions == 1:
        eval_tools.plot_decisions(model,mydata,decisions_path,frame_locations,framesizes)

    if generate_histogram == 1:
        eval_tools.plot_histogram(labels,histogram_output)

    # print the error rate and mean confidence %
    #
    print("Accuracy rate = ",model.score(mydata,labels))
    print("Mean confidence %",eval_tools.mean_confidence(model,mydata))
    

if __name__ == "__main__":
    main()

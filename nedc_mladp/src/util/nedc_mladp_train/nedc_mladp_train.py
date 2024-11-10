#!/usr/bin/env python
#

# import python libraries
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier as RNF
from sklearn.svm import SVC as SVM
import os
import joblib
import numpy


# import project specific libraries
import nedc_mladp_fileio_tools as fileio_tools
import nedc_mladp_feats_tools as feats_tools

# import NEDC libraries
import nedc_file_tools

import sys
sys.path.append('/home/tul16619/SD1/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/src/functs/nedc_mladp_model_tools')
import nedc_mladp_model_tools as model_tools
from nedc_mladp_train_model import modelCNN

def main():

    # set argument parsing
    #
    args_usage = "nedc_mladp_train_model.usage"
    args_help = "nedc_mladp_train_model.help"
    parameter_file = fileio_tools.parseArguments(args_usage,args_help)

    # parse parameters
    #
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"train_model")
    model_type=parsed_parameters['model_type']
    model_path=parsed_parameters['model_path']
    train_data_list=parsed_parameters['data_list_train']
    eval_data_list=parsed_parameters['data_list_eval']
    output_model_directory=parsed_parameters['model_output_path']
    batch_size = int(parsed_parameters['batch_size'])
    num_epochs = int(parsed_parameters['num_epochs'])
    # if not (output_model_directory.endswith('/')):
    #     output_model_directory=output_model_directory + "/"
    
    # If run parameter is set high then get the feature data list from gen feats
    # run_params = nedc_file_tools.load_parameters(parameter_file,"gen_feats")
    # if run_params['run']==1:
    #     feature_data_list=run_params['output_list']

    # Fit the model
    #
    model = None
    if model_type == "RNF":
        model = RNF()
    elif model_type == "CNN":
        model = modelCNN(
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_cls=9)

        # dataset 1: train
        train_feats, train_labels, train_num_cls = model.input_data(train_data_list)

        # dataset 2: evaluation
        eval_feats, eval_labels, eval_num_cls = model.input_data(eval_data_list)

        model.prep_model(
            feats_train=train_feats,
            labels_train=train_labels,
            feats_dev=eval_feats,
            labels_dev=eval_labels
            )
        model.build_model(
            model_path=model_path,
            train_num_cls=train_num_cls,
            dev_num_cls=eval_num_cls
            )
        model.simple_train_model()

    else:
        print("No model supplied")
        return
    # model.fit(mydata, labels)

    # # change the directory to output the model
    # #
    # os.chdir(output_model_directory)

if __name__ == "__main__":
    main()

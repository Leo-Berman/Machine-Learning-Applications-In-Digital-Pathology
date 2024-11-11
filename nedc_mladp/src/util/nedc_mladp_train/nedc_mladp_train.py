#!/usr/bin/env python
#
# import python libraries
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier as RNF
from sklearn.svm import SVC as SVM
import os
import joblib
import numpy
import pandas
import torch
import sys

# import project specific libraries
sys.path.append('Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/lib/')
from nedc_mladp_models import convolutional_neural_network as CNN
import nedc_mladp_fileio_tools

# import NEDC libraries
import nedc_file_tools

def train_model(feature_files:dict=None):

    # Set argument parsing + help and usage
    #
    args_usage = "nedc_mladp_train_model.usage"
    args_help = "nedc_mladp_train_model.help"
    parameter_file = nedc_mladp_fileio_tools.parseArguments(args_usage,args_help)

    # Parse parameters
    #
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"train_model")
    model_type=parsed_parameters['model_type']
    model_path=parsed_parameters['model_path']

    # Features list
    train_data_list=parsed_parameters['data_list_train']
    eval_data_list=parsed_parameters['data_list_eval']

    # CNN attributes
    #
    if model_type == "CNN":
        number_of_classes = int(parsed_parameters["number_of_classes"])
        batch_size = int(parsed_parameters['batch_size'])
        number_of_epochs = int(parsed_parameters['num_epochs'])
        step_size = int(parsed_parameters['step_size'])
        momentum = float(parsed_parameters["momentum"])
        learning_rate = float(parsed_parameters["learning_rate"])
        gamma = float(parsed_parameters['gamma'])

    # Output
    #
    output_directory = parsed_parameters['output_directory']
    if not (output_directory.endswith("/")):
        output_directory += "/"         
    os.makedirs(output_directory,exist_ok=True)
 
    # Train model
    #
    model = None
    if model_type == "QDA":
        model = QDA()
    elif model_type == "RNF":
        model = RNF()
    elif model_type == "SVM":
        model = SVM()
    elif model_type == "CNN":
        model = CNN(
            num_epochs=number_of_epochs,
            batch_size=batch_size,
            num_cls=number_of_classes,
            lr=learning_rate,
            step_size=step_size,
            momentum=momentum,
            gamma=gamma)

        # dataset 1: train
        train_feats, train_labels, train_num_cls = model.input_data(train_data_list)

        # dataset 2: evaluation
        eval_feats, eval_labels, eval_num_cls = model.input_data(eval_data_list)

        model.prep_model(
            feats_train=train_feats,
            labels_train=train_labels,
            feats_dev=eval_feats,
            labels_dev=eval_labels,
            train_num_cls=train_num_cls,
            dev_num_cls=eval_num_cls
            )
        model.build_model(
            model_path=model_path,
            )
        model.simple_train_model()
        
    else:
        print("No model supplied")
        return

    # model.fit(train_data, labels)


    if model_type == "CNN":
        torch.save(model, output_directory+model_type+'.pth')
    else:
        compression=int(parsed_parameters['compression'])
        joblib.dump(model,output_directory+model_type+'.joblib',compress=compression)

    return model

if __name__ == "__main__":
    train_model()

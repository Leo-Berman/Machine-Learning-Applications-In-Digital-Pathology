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

    # Features list + other information
    train_data_list=parsed_parameters['data_list_train']
    eval_data_list=parsed_parameters['data_list_eval']
    number_of_classes = int(parsed_parameters["number_of_classes"])
    PCA_components = int(parsed_parameters["PCA_components"])

    # CNN attributes
    #
    if model_type == "CNN":
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

    output_plot_directory = parsed_parameters['output_plot_directory']
    output_plot_name = parsed_parameters['output_plot_name']
 
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
            gamma=gamma,
            input_size=PCA_components
            )

        # dataset 1: train
        train_feats, train_labels, train_num_cls, train_images_count = model.load_data(train_data_list)
        train_dataloader = model.dataloader(train_feats, train_labels, shuffle_flag=True)
        train_weights = model.compute_weights(train_labels, train=True)

        # dataset 2: evaluation
        eval_feats, eval_labels, eval_num_cls, eval_images_count = model.load_data(eval_data_list)
        eval_dataloader = model.dataloader(eval_feats, eval_labels, shuffle_flag=False)
        eval_weights = model.compute_weights(eval_labels, train=False)

        model.load_info(
            train_num_cls=train_num_cls,
            train_images_count=train_images_count,
            train_feats=train_feats,
            eval_num_cls=eval_num_cls,
            eval_images_count=eval_images_count,
            eval_feats=eval_feats
            )

        model.build_model(
            model_path=model_path,
            train_weights=train_weights,
            eval_weights=eval_weights
            )
        model.train_model(
            train_dataloader=train_dataloader,
            train_weights=train_weights,
            eval_dataloader=eval_dataloader,
            eval_weights=eval_weights,
            validate=True
            )
        model.plot(output_plot_directory, output_plot_name)
        
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

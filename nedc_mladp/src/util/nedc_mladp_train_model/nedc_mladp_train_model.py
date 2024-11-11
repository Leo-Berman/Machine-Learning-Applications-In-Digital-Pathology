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

# import project specific libraries
import nedc_mladp_fileio_tools as fileio_tools
from nedc_mladp_models import convolutional_neural_network as CNN


# import NEDC libraries
import nedc_file_tools

def train_model(feature_files:dict=None):

    # set argument parsing
    #
    args_usage = "nedc_mladp_train_model.usage"
    args_help = "nedc_mladp_train_model.help"
    parameter_file = fileio_tools.parseArguments(args_usage,args_help)

    # parse parameters
    #
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"train_model")
    model_type=parsed_parameters['model_type']

    if model_type == "CNN":
        PCA_components = int(parsed_parameters["PCA_components"])
        number_of_classes = int(parsed_parameters["number_of_classes"])
        layer_01_output_channels = int(parsed_parameters["layer_01_output_channels"])
        layer_01_kernel_size = int(parsed_parameters["layer_01_kernel_size"])
        layer_02_output_channels = int(parsed_parameters["layer_02_output_channels"])
        layer_02_kernel_size = int(parsed_parameters["layer_02_kernel_size"])
        layer_03_output_channels = int(parsed_parameters["layer_03_output_channels"])
        layer_03_kernel_size = int(parsed_parameters["layer_03_kernel_size"])
        dropout_coefficient = float(parsed_parameters["dropout_coefficient"])
        number_of_epochs = int(parsed_parameters["number_of_epochs"])
        learning_rate = float(parsed_parameters["learning_rate"])
        
    output_directory = parsed_parameters['output_directory']
    if not (output_directory.endswith("/")):
        output_directory += "/"
    feature_files = fileio_tools.readLines(parsed_parameters['feature_files_list'])
    labels = []
    train_data = None
    
    for file in feature_files:
        lines = [line.split(',') for line in fileio_tools.readLines(file) if ':' not in line]
        dataframe = pandas.DataFrame(lines[1:],columns=lines[0])
        labels.extend(dataframe['Label'].to_list())
        dataframe = dataframe.drop(['Label','TopLeftRow','TopLeftColumn'],axis=1)
        if train_data is None:
            train_data=dataframe.to_numpy().astype(numpy.float32)
        else:
            train_data = numpy.vstack([train_data,dataframe.to_numpy().astype(numpy.float32)])
                
    os.makedirs(output_directory,exist_ok=True)

            
    # Fit the model
    #
    model = None
    if model_type == "QDA":
        model = QDA()
    elif model_type == "RNF":
        model = RNF()
    elif model_type == "SVM":
        model = SVM()
    elif model_type == "CNN":
        model = CNN(PCA_components, number_of_classes,
                    layer_01_output_channels = layer_01_output_channels,
                    layer_01_kernel_size = layer_01_kernel_size,
                    layer_02_output_channels = layer_02_output_channels,
                    layer_02_kernel_size = layer_02_kernel_size,
                    layer_03_output_channels = layer_03_output_channels,
                    layer_03_kernel_size = layer_03_kernel_size,
                    dropout_coefficient = dropout_coefficient,
                    number_of_epochs = number_of_epochs,
                    learning_rate = learning_rate,
                    model_output_path = output_directory)
    else:
        print("No model supplied")
        return

    model.fit(train_data, labels)


    if model_type == "CNN":
        torch.save(model, output_directory+model_type+'.pth')
    else:
        compression=int(parsed_parameters['compression'])
        joblib.dump(model,output_directory+model_type+'.joblib',compress=compression)

    return model

if __name__ == "__main__":
    train_model()

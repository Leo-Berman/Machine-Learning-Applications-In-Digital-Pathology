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
import ray



# import project specific libraries
import nedc_mladp_fileio_tools as fileio_tools
from nedc_mladp_models import convolutional_neural_network as CNN


# import NEDC libraries
import nedc_file_tools

@ray.remote(num_cpus=32)
def parse(input_file, file_number):
    lines = [line.split(',') for line in fileio_tools.readLines(input_file) if ':' not in line]
    dataframe = pandas.DataFrame(lines[1:],columns=lines[0])
    labels = dataframe['Label'].to_list()
    dataframe = dataframe.drop(['Label','TopLeftRow','TopLeftColumn'],axis=1)
    print(f"File number {file_number} process")
    return labels,dataframe.to_numpy().astype(numpy.float32)

            
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

    tmp_data = ray.get(parse.remote(feature_files.pop(0),0))

    train_data = [tmp_data[1]]
    labels = tmp_data[0]


    batch_size = 400
    for i in range(0,len(feature_files),batch_size):
        start_index = i
        end_index = i + batch_size

        if end_index > len(feature_files):
            end_index = len(feature_files) - 1
        
        tmp_data = ray.get([parse.remote(feature_file, i+file_number+1) for file_number, feature_file in enumerate(feature_files[start_index:end_index])])

        for data in tmp_data:
            train_data.extend(data[1])
            labels.extend(data[0])
        del tmp_data
        print(f"File {i+end_index-start_index} of {len(feature_files)} read",flush=True)
    ray.shutdown()
    print("Past read")
    train_data = numpy.vstack(train_data)
    print("Past vstack")
    
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

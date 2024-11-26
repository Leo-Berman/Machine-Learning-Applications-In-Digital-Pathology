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
import nedc_mladp_models as models

# import NEDC libraries
import nedc_file_tools

@ray.remote(num_cpus=32)
def parse(input_file, file_number, PCA_components):
    lines = [line.split(',') for line in fileio_tools.readLines(input_file) if ':' not in line]
    dataframe = pandas.DataFrame(lines[1:],columns=lines[0])
    labels = dataframe['Label'].to_list()
    dataframe = dataframe.drop(['Label','TopLeftRow','TopLeftColumn'],axis=1)
    columns = dataframe.shape[1]
    print(f"File number {file_number} process", flush=True)
    return labels,dataframe.to_numpy()[:,columns-PCA_components:columns].astype(numpy.float32)

            
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

    if model_type != "CNN_2D" and model_type != "CNN_2D_claudia":
        PCA_components = int(parsed_parameters["PCA_components"])
        number_of_cpus = float(parsed_parameters["number_of_cpus"])
        memory_per_cpu = float(parsed_parameters["memory_per_cpu"])
        feature_files = fileio_tools.readLines(parsed_parameters['feature_files_list'])
    
    if model_type == "CNN_1D":
        number_of_classes = int(parsed_parameters["number_of_classes"])
        layer_01_convolution_output_channels = int(parsed_parameters["layer_01_convolution_output_channels"])
        layer_01_convolution_kernel_size = int(parsed_parameters["layer_01_convolution_kernel_size"])
        layer_01_max_pooling_kernel_size = int(parsed_parameters["layer_01_max_pooling_kernel_size"])
        layer_02_convolution_output_channels = int(parsed_parameters["layer_02_convolution_output_channels"])
        layer_02_convolution_kernel_size = int(parsed_parameters["layer_02_convolution_kernel_size"])
        layer_02_max_pooling_kernel_size = int(parsed_parameters["layer_02_max_pooling_kernel_size"])
        layer_03_convolution_output_channels = int(parsed_parameters["layer_03_convolution_output_channels"])
        layer_03_convolution_kernel_size = int(parsed_parameters["layer_03_convolution_kernel_size"])
        layer_03_max_pooling_kernel_size = int(parsed_parameters["layer_03_max_pooling_kernel_size"])
        dropout_coefficient = float(parsed_parameters["dropout_coefficient"])
        number_of_epochs = int(parsed_parameters["number_of_epochs"])
        learning_rate = float(parsed_parameters["learning_rate"])
        batch_size = int(parsed_parameters["batch_size"])

    elif model_type == "CNN_2D":
        number_of_classes = int(parsed_parameters["number_of_classes"])
        layer_01_convolution_output_channels = int(parsed_parameters["layer_01_convolution_output_channels"])
        layer_01_convolution_kernel_size = int(parsed_parameters["layer_01_convolution_kernel_size"])
        layer_01_max_pooling_kernel_size = int(parsed_parameters["layer_01_max_pooling_kernel_size"])
        layer_01_max_pooling_stride = int(parsed_parameters["layer_01_max_pooling_stride"])
        layer_02_convolution_output_channels = int(parsed_parameters["layer_02_convolution_output_channels"])
        layer_02_convolution_kernel_size = int(parsed_parameters["layer_02_convolution_kernel_size"])
        layer_02_max_pooling_kernel_size = int(parsed_parameters["layer_02_max_pooling_kernel_size"])
        layer_02_max_pooling_stride = int(parsed_parameters["layer_02_max_pooling_stride"])
        layer_03_convolution_output_channels = int(parsed_parameters["layer_03_convolution_output_channels"])
        layer_03_convolution_kernel_size = int(parsed_parameters["layer_03_convolution_kernel_size"])
        layer_03_max_pooling_kernel_size = int(parsed_parameters["layer_03_max_pooling_kernel_size"])
        layer_03_max_pooling_stride = int(parsed_parameters["layer_03_max_pooling_stride"])
        dropout_coefficient = float(parsed_parameters["dropout_coefficient"])
        number_of_epochs = int(parsed_parameters["number_of_epochs"])
        learning_rate = float(parsed_parameters["learning_rate"])
        batch_size = int(parsed_parameters["batch_size"])
        frame_size = ( int(parsed_parameters["frame_width"]),int(parsed_parameters["frame_height"]) )
        window_size = ( int(parsed_parameters["window_width"]),int(parsed_parameters["window_height"]) )
        image_files = parsed_parameters["image_files"]
        annotation_files = parsed_parameters["annotation_files"]
        overlap_threshold = float(parsed_parameters["overlap_threshold"])
        cpus_per_batch = float(parsed_parameters["cpus_per_batch"])
        memory_per_batch = float(parsed_parameters["memory_per_batch"]) * 1024 * 1024 * 1024
        image_cache_size = int(parsed_parameters["image_cache_size"])

    elif model_type == "CNN_2D_claudia":
        number_of_classes = int(parsed_parameters["number_of_classes"])
        layer_01_convolution_output_channels = int(parsed_parameters["layer_01_convolution_output_channels"])
        layer_01_convolution_kernel_size = int(parsed_parameters["layer_01_convolution_kernel_size"])
        layer_01_max_pooling_kernel_size = int(parsed_parameters["layer_01_max_pooling_kernel_size"])
        layer_01_max_pooling_stride = int(parsed_parameters["layer_01_max_pooling_stride"])
        layer_02_convolution_output_channels = int(parsed_parameters["layer_02_convolution_output_channels"])
        layer_02_convolution_kernel_size = int(parsed_parameters["layer_02_convolution_kernel_size"])
        layer_02_max_pooling_kernel_size = int(parsed_parameters["layer_02_max_pooling_kernel_size"])
        layer_02_max_pooling_stride = int(parsed_parameters["layer_02_max_pooling_stride"])
        dropout_coefficient = float(parsed_parameters["dropout_coefficient"])
        number_of_epochs = int(parsed_parameters["number_of_epochs"])
        learning_rate = float(parsed_parameters["learning_rate"])
        batch_size = int(parsed_parameters["batch_size"])
        frame_size = ( int(parsed_parameters["frame_width"]),int(parsed_parameters["frame_height"]) )
        window_size = ( int(parsed_parameters["window_width"]),int(parsed_parameters["window_height"]) )
        image_files = parsed_parameters["image_files"]
        annotation_files = parsed_parameters["annotation_files"]
        overlap_threshold = float(parsed_parameters["overlap_threshold"])
        cpus_per_batch = float(parsed_parameters["cpus_per_batch"])
        memory_per_batch = float(parsed_parameters["memory_per_batch"]) * 1024 * 1024 * 1024
        image_cache_size = int(parsed_parameters["image_cache_size"])
        object_store_memory = float(parsed_parameters["object_store_memory"])
        load_model=int(parsed_parameters["load_model"])
        if load_model == 1:
            load_model_path = parsed_parameters["load_model_path"]
    output_directory = parsed_parameters['output_directory']
    if not (output_directory.endswith("/")):
        output_directory += "/"

    print(f"Model type = {model_type}")
        
    if model_type != "CNN_1D" and model_type != "CNN_2D" and model_type != "CNN_2D_claudia":
        print('Parsing data into memory', flush=True)
        ray.init(ignore_reinit_error=True)
        train_data = []
        labels = []
        tmp_data = ray.get([parse.options(num_cpus=number_of_cpus, num_gpus=0, memory = memory_per_cpu * 1024 * 1024 * 1024).remote(feature_file, file_number+1, PCA_components) for file_number, feature_file in enumerate(feature_files)])
        ray.shutdown()
        for data in tmp_data:
            train_data.extend(data[1])
            labels.extend(data[0])

        train_data = numpy.vstack(train_data)

        print('Data parsed',flush=True)
    os.makedirs(output_directory,exist_ok=True)

            
    # Fit the model
    #
    model = None
    if model_type == "QDA":
        model = QDA()
    elif model_type == "RNF":
        model = RNF(n_jobs=-1, max_depth=10, min_samples_leaf = 3, n_estimators=300)
    elif model_type == "SVM":
        model = SVM()
    elif model_type == "CNN_1D":
        model = models.CNN(PCA_components, number_of_classes,

                           layer_01_convolution_output_channels = layer_01_convolution_output_channels,
                           layer_01_convolution_kernel_size = layer_01_convolution_kernel_size,
                           layer_01_max_pooling_kernel_size = layer_01_max_pooling_kernel_size,
                           
                           layer_02_convolution_output_channels = layer_02_convolution_output_channels,
                           layer_02_convolution_kernel_size = layer_02_convolution_kernel_size,
                           layer_02_max_pooling_kernel_size = layer_02_max_pooling_kernel_size,
                           
                           layer_03_convolution_output_channels = layer_03_convolution_output_channels,
                           layer_03_convolution_kernel_size = layer_03_convolution_kernel_size,
                           layer_03_max_pooling_kernel_size = layer_03_max_pooling_kernel_size,
                           
                           dropout_coefficient = dropout_coefficient,
                           number_of_epochs = number_of_epochs,
                           learning_rate = learning_rate,
                           model_output_path = output_directory)
        model = model.to(model.getDevice())

    elif model_type == "CNN_2D":
        model = models.CNN_2D(number_of_classes,
                              layer_01_convolution_output_channels,
                              layer_01_convolution_kernel_size,
                              layer_01_max_pooling_kernel_size,
                              layer_01_max_pooling_stride,
                              layer_02_convolution_output_channels,
                              layer_02_convolution_kernel_size,
                              layer_02_max_pooling_kernel_size,
                              layer_02_max_pooling_stride,
                              layer_03_convolution_output_channels,
                              layer_03_convolution_kernel_size,
                              layer_03_max_pooling_kernel_size,
                              layer_03_max_pooling_stride,
                              dropout_coefficient,
                              window_size)
    elif model_type == "CNN_2D_claudia":
        if load_model == 0:
            model = models.CNN_2D_claudia(number_of_classes,
                                          layer_01_convolution_output_channels,
                                          layer_01_convolution_kernel_size,
                                          layer_01_max_pooling_kernel_size,
                                          layer_01_max_pooling_stride,
                                          layer_02_convolution_output_channels,
                                          layer_02_convolution_kernel_size,
                                          layer_02_max_pooling_kernel_size,
                                          layer_02_max_pooling_stride,
                                          dropout_coefficient,
                                          window_size)
        else:
            model = models.CNN_2D_claudia(number_of_classes,
                                          layer_01_convolution_output_channels,
                                          layer_01_convolution_kernel_size,
                                          layer_01_max_pooling_kernel_size,
                                          layer_01_max_pooling_stride,
                                          layer_02_convolution_output_channels,
                                          layer_02_convolution_kernel_size,
                                          layer_02_max_pooling_kernel_size,
                                          layer_02_max_pooling_stride,
                                          dropout_coefficient,
                                          window_size,
                                          load_model_path=load_model_path)
    else:
        print("No model supplied")
        return

    
    if model_type == "CNN_2D" or model_type == "CNN_2D_claudia":
        model.fit(image_files, annotation_files,
                  number_of_epochs, learning_rate, batch_size,
                  overlap_threshold, frame_size, output_directory,
                  cpus_per_batch=cpus_per_batch,
                  memory_per_batch=memory_per_batch,
                  image_cache_size = image_cache_size,
                  object_store_memory = object_store_memory)
        torch.save(model, output_directory+model_type+'.pth')

    elif model_type == "CNN_1D":
        model.fit(feature_files, batch_size = batch_size)
        torch.save(model, output_directory+model_type+'.pth')

    else:
        model.fit(train_data, labels)
        compression=int(parsed_parameters['compression'])
        joblib.dump(model,output_directory+model_type+'.joblib',compress=compression)
    
    return model

if __name__ == "__main__":
    train_model()

#!/usr/bin/env python

# import python libraries
import joblib
import os
import pandas
import numpy
import torch

import ray

# import project specific libraries
import nedc_mladp_fileio_tools as fileio_tools
import nedc_mladp_pred_tools as pred_tools
from nedc_mladp_models import *

# import NEDC libraries
import nedc_file_tools
import nedc_dpath_ann_tools

@ray.remote
def RGB_predictFile(image_file:str, annotation_file:str, model,
                    regions_output_directory:str, frame_size:tuple,
                    overlap_threshold:float, window_size:tuple):
    print(image_file, annotation_file)

    try:
        dataset = ImageDataset([image_file], [annotation_file], frame_size,
                               window_size, overlap_threshold, cpus_per_batch = .25,
                               memory_per_batch = .5, prediction_bool=True)

        coordinates = dataset.returnCoordinates()
        
        dataloader = DataLoader(dataset, batch_size = len(dataset), collate_fn=frameCollate)
        
        # Get the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # If there is more than 1 GPU, parallelize
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            
        # Send the model to the appropriate device
        model = model.to(device)
        model.eval()
        # iterate through the batches
        for data, labels in dataloader:
        
            # send the data and labls to the proper device
            device_data = data.to(device)
            device_labels = labels.to(device)
        
            # get the class predictions
            outputs = model(device_data)
            
            # track for epoch accuracy
            confidences, predicted_classes = torch.max(outputs,1)
            total = predicted_classes.size(0)
            correct = sum([(predicted == actual) for predicted,actual in zip(predicted_classes,device_labels)])
            labels_correct=correct
            total_labels=total
            #probabilities = outputs.softmax(dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            confidences = torch.max(confidences, dim=1).values

        predicted_classes = predicted_classes.cpu().tolist()
        confidences = confidences.cpu().tolist()
        print(confidences, predicted_classes)
        annotation_writer = nedc_dpath_ann_tools.AnnDpath()
        annotation_writer.load(annotation_file)
        header = annotation_writer.get_header()
        output_filepath = regions_output_directory+header['bname']+"_REGIONDECISIONS.csv"
        print("Predicted Classes = ",predicted_classes)
        print("Coordinates = ",coordinates)
        print("Confidence = ",confidences)
        print("Frame size = ",frame_size)
        prediction_graph = pred_tools.regionPredictions(predicted_classes,
                                                        coordinates,
                                                        confidences,
                                                        frame_size)
        print(prediction_graph)
        annotation_writer.set_graph(prediction_graph)
        annotation_writer.set_type("csv")
        annotation_writer.write(output_filepath)
        return {"predictions":output_filepath+'\n', "original":annotation_file}
    except Exception as e:
        print(e)
        return None
@ray.remote(num_cpus=1, num_gpus=1)
def predict_file(input_feature_file:str, annotation_file:str, model, regions_output_directory:str, PCA_components:int, index):

    try:        
        lines = [line.split(',') for line in fileio_tools.readLines(input_feature_file)]
        header_info = {}
        i = 0
        while ':' in lines[i][0]:
            key,value = lines.pop(0)[0].split(':')
            header_info[key]=value

        dataframe = pandas.DataFrame(lines[1:],columns=lines[0])
        labels = dataframe['Label'].to_list()
        top_left_coordinates = list(zip(dataframe['TopLeftColumn'].to_list(),dataframe['TopLeftRow'].to_list()))
        annotation_reader = nedc_dpath_ann_tools.AnnDpath()
        annotation_reader.load(annotation_file)
        header = annotation_reader.get_header()
        dataframe = dataframe.drop(['Label','TopLeftColumn','TopLeftRow'], axis=1)
        columns = dataframe.shape[1]
        PCs = dataframe.to_numpy()[:,columns-PCA_components:columns].astype(numpy.float32)
        
        feature_file = { 'Frame Decisions':labels,
                         'Top Left Coordinates':top_left_coordinates,
                         'Header':header,
                         'PCs':PCs,
                         'Frame Size':(int(header_info['frame_height']),int(header_info['frame_width'])),
                        }
        feature_file['Frame Confidences'] = [max(predictions) for predictions in model.predict_proba(numpy.array(feature_file['PCs']).astype(numpy.float32))]
        feature_file['Frame Decisions'] = model.predict(numpy.array(feature_file['PCs']).astype(numpy.float32))
        prediction_graph = pred_tools.regionPredictions(feature_file['Frame Decisions'],
                                                        feature_file['Top Left Coordinates'],
                                                        feature_file['Frame Confidences'],
                                                        feature_file['Frame Size'])
        
        annotation_writer = nedc_dpath_ann_tools.AnnDpath()
        annotation_writer.set_type("csv")
        annotation_writer.set_header(feature_file['Header'])
        annotation_writer.set_graph(prediction_graph)
        output_filepath = regions_output_directory+feature_file['Header']['bname']+"_REGIONDECISIONS.csv"
        annotation_writer.write(output_filepath)
        print(f"File {index} done")
        return {"predictions":output_filepath+'\n', "original":annotation_file}
    except Exception as e:
        print(f"Error {e}")
        return None
def gen_preds(feature_files:dict=None, model=None):

    # set argument parsing
    #
    args_usage = "nedc_mladp_gen_preds.usage"
    args_help = "nedc_mladp_gen_preds.help"
    parameter_file = fileio_tools.parseArguments(args_usage,args_help)
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"gen_preds")
    number_of_cpus = float(parsed_parameters['number_of_cpus'])
    number_of_gpus = float(parsed_parameters['number_of_gpus'])
    write_region_decisions = int(parsed_parameters['write_region_decisions'])
    write_frame_decisions = int(parsed_parameters['write_frame_decisions'])
    memory_per_cpu = float(parsed_parameters['memory_per_cpu'])
    object_memory = float(parsed_parameters['object_memory'])

    model_type = parsed_parameters['model_type']
    output_directory = parsed_parameters['output_directory']
    if not (output_directory.endswith("/")):
        output_directory += "/"
    model_file = parsed_parameters['model_file']
    original_annotation_files_list = parsed_parameters['original_annotation_files_list']

    if model_type == "CNN_2D":
        image_files_list = fileio_tools.readLines(parsed_parameters['image_files_list'])
        frame_size = (int(parsed_parameters['frame_width']), int(parsed_parameters['frame_height']))
        window_size = (int(parsed_parameters['window_width']), int(parsed_parameters['window_height']))
        overlap_threshold = float(parsed_parameters['overlap_threshold'])

    else:
        feature_files_list = parsed_parameters['feature_files_list']
        PCA_components = int(parsed_parameters['PCA_components'])
        feature_files = []
        feature_files_list = fileio_tools.readLines(feature_files_list)
    
    original_files_list = fileio_tools.readLines(original_annotation_files_list)

    regions_output_directory = output_directory + 'regions/'
    frames_output_directory = output_directory + 'frames/'
    os.makedirs(regions_output_directory,exist_ok=True)
    os.makedirs(frames_output_directory,exist_ok=True)

    if model is None:
        if "CNN" in model_type:
            model = torch.load(model_file)
        else:
            model = joblib.load(model_file)
        print("Model loaded")

             
    region_decision_files = []
    original_annotations = []
    ray.init(object_store_memory=object_memory * 1024 * 1024 * 1024, ignore_reinit_error=True)
    ray_model = ray.put(model.module)
    ray_output_directory = ray.put(regions_output_directory)

    if model_type == "CNN_2D":
        print(image_files_list)
        prediction_lists = ray.get([RGB_predictFile.options(num_cpus=number_of_cpus,
                                                            num_gpus=number_of_gpus,
                                                            memory=memory_per_cpu * 1024 * 1024 * 1024).remote(image_file, annotation_file, ray_model, ray_output_directory, frame_size, overlap_threshold, window_size) for image_file, annotation_file in zip(image_files_list, original_files_list)])
    else:
        prediction_lists = ray.get([predict_file.options(num_cpus=number_of_cpus, num_gpus=number_of_gpus, memory=memory_per_cpu*1024*1024*1024).remote(feature_file, original_file, ray_model, ray_output_directory, PCA_components, i) for i,(feature_file,original_file) in enumerate(zip(feature_files_list,original_files_list))])

    ray.shutdown()
    for pair in prediction_lists:
        if pair is not None:
            region_decision_files.append(pair["predictions"])
            original_annotations.append(pair["original"])

    with open(output_directory+'regions/region_decisions_files.list','w') as f:
        f.writelines(region_decision_files)
    with open(output_directory+'regions/original_annotation_files.list','w') as f:
        f.writelines([file +'\n' for file in original_annotations])

if __name__ == "__main__":
    gen_preds()

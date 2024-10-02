#!/usr/bin/env python3

import nedc_mladp_fileio_tools as local_fileio

import nedc_file_tools

import sys
sys.path.append('./bin')

from generate_features import generate_features
from train_model import train_model
from generate_predictions import generate_predictions

def main():

    # parse command line arguments
    #
    arguments_usage = "main.usage"
    arguments_help = "main.help"
    parameter_file = local_fileio.parseArguments(arguments_usage,
                                                 arguments_help)

    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,'main')

    window_height = int(parsed_parameters['window_height'])
    window_width = int(parsed_parameters['window_width'])
    frame_height = int(parsed_parameters['frame_height'])
    frame_width = int(parsed_parameters['frame_width'])

    window_dimensions = (window_width,window_height)
    frame_dimensions = (frame_width, frame_height)
    
    images_list = local_fileio.readFileLists(parsed_parameters['images_list'])
    annotations_list = local_fileio.readFileLists(parsed_parameters['annotations_list'])

    master_dictionary = {}

    
    for image,annotation in zip(images_list,annotations_list):

        # albert
        generateFeatures(master_dictionary,
                          frame_dimensions,
                          window_dimensions,
                          image,
                          annotation)
        # yuan
        trainModel(master_dictionary)

        # leo
        generatePredictions(master_dictionary)
                          
    
    
if __name__ == "__main__":
    main()

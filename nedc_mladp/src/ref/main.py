#!/usr/bin/env python3

import nedc_mladp_fileio_tools as local_fileio

def main():

    # parse command line arguments
    #
    arguments_usage = "main.usage"
    arguments_help = "main.help"
    parameter_file = local_fileio.parseArguments(arguments_usage,
                                                 arguments_help)

    
    window_size = int(parsed_parameters['window_size'])
    frame_size = int(parsed_parameters['frame_size'])
    images_list = local_fileio.readFileLists(parsed_parameters['images_list'])
    annotations_list = local_fileio.readFileLists(parsed_parameters['annotations_list'])

    
    images_per_increment = int(parsed_parameters['images_per_increment'])
    number_of_epochs = int(parsed_parameters['number_of_epochs'])

    master_dictionary = {}
    epochs_ran = 0
    images_processed = 0
    for image_index in images_per_increment:
        header, ids, labels, coordinates = local_fileio.parseAnnotations(annotations_list[image_index])
        print("Header = ",header)
        print("Ids = ",ids)
        print("Labels ", labels)
        print("Coordinates = ",coordinates)
        

    

    
    
    
if __name__ == "__main__":
    main()

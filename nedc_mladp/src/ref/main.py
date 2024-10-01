#!/usr/bin/env python3

import nedc_mladp_fileio_tools as local_fileio
import nedc_mladp_ann_tools as local_ann

import nedc_file_tools
import nedc_dpath_ann_tools

def main():

    # parse command line arguments
    #
    arguments_usage = "main.usage"
    arguments_help = "main.help"
    parameter_file = local_fileio.parseArguments(arguments_usage,
                                                 arguments_help)

    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,'main')
    window_size = int(parsed_parameters['window_size'])
    frame_size = int(parsed_parameters['frame_size'])
    images_list = local_fileio.readFileLists(parsed_parameters['images_list'])
    annotations_list = local_fileio.readFileLists(parsed_parameters['annotations_list'])
    images_per_increment = int(parsed_parameters['images_per_increment'])
    number_of_epochs = int(parsed_parameters['number_of_epochs'])

    master_dictionary = {}
    epochs_ran = 0
    images_processed = 0
    for image_index in range(images_per_increment):


        annotation_tool = nedc_dpath_ann_tools.AnnDpath()
        annotation_tool.load(annotations_list[image_index])
        
        master_dictionary[images_processed] = {
            'header':annotation_tool.get_header(),
            'frame_size':frame_size,
            'window_size':window_size,
        }
        


        local_ann.generateFeatures(annotation_tool.get_graph(),
                                   master_dictionary[images_processed]['header'],
                                   frame_size, window_size)

        
        images_processed += 1
    
    
if __name__ == "__main__":
    main()

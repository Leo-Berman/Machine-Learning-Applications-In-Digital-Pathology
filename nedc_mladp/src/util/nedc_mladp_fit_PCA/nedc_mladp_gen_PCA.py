#!/usr/bin/env python

# import python libraries
#
import os
import pandas
import numpy

#import sklearn.decomposition
import dask_ml.decomposition
import dask.array

# import project-specific libraries
#
import nedc_mladp_fileio_tools as fileio_tools
import nedc_mladp_feats_tools as feats_tools

# import NEDC libraries
#
import nedc_file_tools
import joblib


def gen_PCA():

    # Set command line arguments and get name of parameter file
    args_usage = "nedc_PCA.usage"
    args_help = "nedc_PCA.help"
    parameter_file = fileio_tools.parseArguments(args_usage,args_help)

    # parse the parameter file
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"gen_PCA")
    window_region_overlap_threshold = float(parsed_parameters['window_region_overlap_threshold'])
    PCA_components = int(parsed_parameters['PCA_components'])
    PCA_compression = int(parsed_parameters['PCA_compression'])
    window_size = (int(parsed_parameters['window_width']),int(parsed_parameters['window_height']))
    frame_size = (int(parsed_parameters['frame_width']),int(parsed_parameters['frame_height']))
    image_files_list = fileio_tools.readLines(parsed_parameters['image_files_list'])
    annotation_files_list = fileio_tools.readLines(parsed_parameters['annotation_files_list'])
    output_directory = parsed_parameters['output_directory']
    if not (output_directory.endswith("/")):
        output_directory += "/"
    
    # create output directory
    os.makedirs(output_directory, exist_ok=True)

    # initialize PCA
    PCA = dask_ml.decomposition.IncrementalPCA(n_components=PCA_components, copy=False)

    # Create variables to hold information
    DCTs_for_PCA = [] # List to hold DCTs for the PCA to train on
    total_windows_PCA_trained_on = 0 # Total number of windows PCA has processed

    # set maximum size of array of floats that can be stored in memory
    number_of_gigabytes = 10
    PCA_fit_size = number_of_gigabytes * 1000000000 // (32 * window_size[0]*window_size[1])
    if PCA_components > PCA_fit_size:
        PCA_fit_size = PCA_components

    PCA_buffer = []
    
    # iterate through image files and annotation files and create a dictionary
    # to hold information for each set of files
    #
    for i,image_file,annotation_file in zip(range(len(image_files_list)),
                                            image_files_list,
                                            annotation_files_list):

        # update the user
        print(f"Processing file {annotation_file} {i+1} of {len(image_files_list)}")
        
        try:

            # parse annotations
            header, ids, labels, coordinates = fileio_tools.parseAnnotations(annotation_file)

            # get labeled regions
            labeled_regions = feats_tools.labeledRegions(coordinates)

            # return top left coordinates of frames that have center coordinates in labels
            frame_top_left_coordinates,frame_labels = feats_tools.classifyFrames(labels,
                                                                                 int(header['height']),
                                                                                 int(header['width']),
                                                                                 window_size,
                                                                                 frame_size,
                                                                                 labeled_regions,
                                                                                 window_region_overlap_threshold)

            # get list of rgba values
            window_RGBs = feats_tools.windowRGBValues(image_file,
                                                      frame_top_left_coordinates,
                                                      window_size)

            # perform dct on rgba values and typecase to 32bit floats
            window_DCTs = feats_tools.windowDCT(window_RGBs)

            PCA_buffer.extend(window_DCTs)
            
            current_buffer_length = len(PCA_buffer)
            print(f"Buffer Capacity = {current_buffer_length}/{PCA_fit_size}")
            if current_buffer_length >= PCA_fit_size:    
                try:
                    PCA_buffer = dask.array.from_array(PCA_buffer)
                    PCA.partial_fit(PCA_buffer)
                    print(f"Trained successfully on {current_buffer_length} windows. {total_windows_PCA_trained_on} Total")
                    total_windows_PCA_trained_on+=current_buffer_length
                    PCA_buffer = []
                except Exception as e:
                    print(f"Incremental PCA training Failed due to: \n{e}\n")

                        
        except Exception as e:
            print(f"{header['bname']} File Failed due to: \n{e}\n", flush=True)

    if len(PCA_buffer) >= PCA_components:
                try:
                    PCA_buffer = dask.array.from_array(DCTs_for_PCA)
                    PCA.partial_fit(PCA_buffer)
                    total_windows_PCA_trained_on+=len(PCA_buffer)
                    PCA_buffer = []
                    print(f"Trained successfully on {current_buffer_length} windows. {total_windows_PCA_trained_on} Total")
                except Exception as e:
                    print(f"Incremental PCA training Failed due to: \n{e}\n", flush=True)
                
    joblib.dump(PCA,output_directory+"PCA.joblib",compress=PCA_compression)
    print(f"PCA {output_directory+'PCA.joblib'} successfully written", flush=True)
        
if __name__ == "__main__":
    gen_PCA()

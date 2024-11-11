#!/usr/bin/env python

# import python libraries
#
import os
import pandas
import numpy
import ray

ray.init()

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
from functools import partial

@ray.remote(num_cpus=4)
def my_parallel(i, image_file, annotation_file, frame_size, window_size, PCA, window_region_overlap_threshold, features_header, original_annotation_files, feature_files, output_directory):
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
        
        # perform dct on rgba values and cast to 32 bit floats
        window_DCTs = feats_tools.windowDCT(window_RGBs)
        
        # Execute PCA transform
        PCs = PCA.transform(window_DCTs)
        
        labels_dataframe = pandas.DataFrame({'Label':frame_labels})
        
        coordinates_dataframe = pandas.DataFrame(frame_top_left_coordinates,
                                                 columns=['TopLeftColumn','TopLeftRow'])
        
        features_dataframe = pandas.DataFrame(PCs,columns=features_header)
        
        dataframe=labels_dataframe.join([coordinates_dataframe,features_dataframe])
        
        file_path = output_directory + header['bname'] + "_FEATS.csv"
        
        with open(file_path,'w') as f:
            for key,value in header.items():
                f.write(f'{key}:{value}\n')
            f.write(f'frame_height:{frame_size[0]}\n')
            f.write(f'frame_width:{frame_size[1]}\n')                    
            f.write(f'window_height:{window_size[0]}\n')
            f.write(f'window_width:{window_size[1]}\n')
            dataframe.to_csv(f, index=False, header = True)    
        print(f"File {i} processed")
        original_annotation_files.append(annotation_file)
        feature_files.append(file_path)
        
    except Exception as e:
        print(f"{header['bname']} Write Failed Due To\n{e}\n")

def gen_feats():

    # Set command line arguments and get name of parameter file
    args_usage = "nedc_mladp_gen_feats.usage"
    args_help = "nedc_mladp_gen_feats.help"
    parameter_file = fileio_tools.parseArguments(args_usage,args_help)

    # parse the parameter file
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"gen_feats")
    window_region_overlap_threshold = float(parsed_parameters['window_region_overlap_threshold'])
    window_size = (int(parsed_parameters['window_width']),int(parsed_parameters['window_height']))
    frame_size = (int(parsed_parameters['frame_width']),int(parsed_parameters['frame_height']))
    PCA_path = parsed_parameters['PCA_path']
    image_files_list = fileio_tools.readLines(parsed_parameters['image_files_list'])
    annotation_files_list = fileio_tools.readLines(parsed_parameters['annotation_files_list'])
    output_directory = parsed_parameters['output_directory']
    if not (output_directory.endswith("/")):
        output_directory += "/"

    PCA = joblib.load(PCA_path)
    print(f"\nPCA {PCA_path} Successfully Loaded")

    # create feature files schema
    features_header = []
    for i in range(PCA.n_components):
        features_header.append(f"PCA_From_DCT_Feature{i}")


    original_annotation_files = []
    feature_files = []
        
    '''# iterate through files
    for i,image_file,annotation_file in zip(range(len(image_files_list)),
                                            image_files_list,
                                            annotation_files_list):
    '''


    ray_PCA = ray.put(PCA)
    partial_function = partial(i, image_file, annotation_file, frame_size,
                               window_size, PCA=ray_PCA, 
    

    batch_size = 88//4
    
    for i in range(0,len(image_files_list),batch_size):

        start_index = i
        end_index = i + batch_size

        if end_index > len(image_files_list):
            end_index = len(image_files_list) - 1
        
        to_get = [my_parallel.remote(i, image_file, annotation_file, frame_size, window_size,
                                     ray_PCA, window_region_overlap_threshold, features_header,
                                     original_annotation_files, feature_files, output_directory) for i,image_file,annotation_file in zip(range(len(image_files_list[start_index:end_index])), image_files_list, annotation_files_list[start_index:end_index])]
            
        ray.get(to_get)
            
    with open(output_directory +"original_annotations.list",'w') as f:
        f.writelines(line + '\n' for line in original_annotation_files)
    print(f"Original list of annotation files {output_directory + 'original_annotations.list'} written\n")

    with open(output_directory +"feature_files.list",'w') as f:
        f.writelines(line + '\n' for line in feature_files)
    print(f"Generated list of feature files {output_directory+'feature_files.list'} written\n")
        
if __name__ == "__main__":
    gen_feats()

#!/usr/bin/env python

# import python libraries
#
import os
import pandas
import numpy
import sklearn.decomposition

# import project-specific libraries
#
import nedc_mladp_fileio_tools as fileio_tools
import nedc_mladp_feats_tools as feats_tools

# import NEDC libraries
#
import nedc_file_tools
import joblib

def gen_feats():

    # Set command line arguments and get name of parameter file
    args_usage = "nedc_mladp_gen_feats.usage"
    args_help = "nedc_mladp_gen_feats.help"
    parameter_file = fileio_tools.parseArguments(args_usage,args_help)

    # parse the parameter file
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"gen_deveval")

    # Parse parameters for generating features
    window_region_overlap_threshold = float(parsed_parameters['window_region_overlap_threshold'])
    window_width = int(parsed_parameters['window_width'])
    window_height = int(parsed_parameters['window_height'])
    window_size = (window_width,window_height)
    frame_width =  int(parsed_parameters['frame_width'])
    frame_height =  int(parsed_parameters['frame_height'])
    frame_size = (frame_width,frame_height)
    output_directory = parsed_parameters['output_directory']
    if not (output_directory.endswith("/")):
        output_directory += "/"
    
    # read list of files into memory
    image_files_list = fileio_tools.readLines(parsed_parameters['image_files_list'])
    annotation_files_list = fileio_tools.readLines(parsed_parameters['annotation_files_list'])

    
    # update user
    print(f"\nParameter file {parameter_file} Parsed Successfully\n")

    # create output directory
    os.makedirs(output_directory,exist_ok=True)

    # update user
    print(f"Output directory {output_directory} successfully created\n")
    
    PCA_path = parsed_parameters['PCA_path']
    PCA = joblib.load(PCA_path)
    print(f"PCA {PCA_path}Successfully Loaded")

    # create feature files schema
    features_header = []
    for i in range(PCA.n_components):
        features_header.append(f"PCA_From_DCT_Feature{i}")

    
    # Create variables to hold information
    finished_files = [] # Dictionaries holding information for next step
    original_files_written = [] # List of original files used
    feature_files_written = [] # List of features files written
    DCTs_for_PCA = [] # List to hold DCTs for the PCA to train on
    total_windows_PCA_trained_on = 0 # Total number of windows PCA has processed

    
    
    # iterate through image files and annotation files and create a dictionary
    # to hold information for each set of files
    #
    for i,image_file,annotation_file in zip(range(len(image_files_list)),
                                            image_files_list,
                                            annotation_files_list):

        PCs = None
        
        try:
        
            # parse annotations
            #
            header, ids, labels, coordinates = fileio_tools.parseAnnotations(annotation_file)

            print(f"File {i+1} of {len(image_files_list)} Processing\n")
            
            # get height and width of image (in pixels) from the header
            #
            height = int(header['height'])
            width = int(header['width'])

            # get labeled regions
            #
            labeled_regions = feats_tools.labeledRegions(coordinates)

        except Exception as e:
            print(f"{annotation_file} Failed to parse image annotations due to {e}\n")



        print(f"File {i+1} of {len(image_files_list)} Getting Principal Components\n")
            
            
        # return top left coordinates of frames that have center coordinates in labels
        #
        frame_top_left_coordinates,frame_labels = feats_tools.classifyFrames(labels,height, width,
                                                                             window_size, frame_size,
                                                                             labeled_regions,
                                                                             window_region_overlap_threshold,
                                                                             1)
            
        try:
            print("Length = ",len(frame_top_left_coordinates))
            for i in range(0,len(frame_top_left_coordinates),1000):
                # get list of rgba values
                #
                window_RGBs = feats_tools.windowRGBValues(image_file,
                                                          frame_top_left_coordinates[i:i+1000],
                                                          window_size)
                
                # perform dct on rgba values
                #
                tmp_PCs = PCA.transform(feats_tools.windowDCT(window_RGBs))
                if PCs is None:
                    PCs = tmp_PCs
                else:
                    PCs = numpy.vstack([PCs,tmp_PCs])
                #print(PCs)
                
            print(f"{header['bname']} DCT + PCA Succeeded")
            feature_files_written.append(output_directory + header['bname'] + "_FEATS.csv")
            original_files_written.append(annotation_file)
        except Exception as e:
            print(f"{header['bname']} PCA Failed due to {e}")
            
        labels_dataframe = pandas.DataFrame({'Label':labels})
            
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
                print(f"File {i+1} of {len(finished_files)} Write Succeeded")

    with open(output_directory +"original_annotations.list",'w') as f:
        f.writelines(line + '\n' for line in original_files_written)
        print(f"Original list of annotation files {output_directory + 'original_annotations.list'} written")


    with open(output_directory +"feature_files.list",'w') as f:
        f.writelines(line + '\n' for line in feature_files_written)
    print(f"Generated list of feature files {output_directory+'feature_files.list'} written")
        
if __name__ == "__main__":
    gen_feats()

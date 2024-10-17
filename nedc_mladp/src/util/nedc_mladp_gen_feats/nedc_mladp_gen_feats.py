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
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"gen_feats")

    # check to see if features are to be written
    write_features=int(parsed_parameters['write_features'])

    # Parse parameters for generating features
    window_region_overlap_threshold = float(parsed_parameters['window_region_overlap_threshold'])
    window_width = int(parsed_parameters['window_width'])
    window_height = int(parsed_parameters['window_height'])
    window_size = (window_width,window_height)
    frame_width =  int(parsed_parameters['frame_width'])
    frame_height =  int(parsed_parameters['frame_height'])
    frame_size = (frame_width,frame_height)
    
    # Check if the program is being run from the top
    run_parameters = nedc_file_tools.load_parameters(parameter_file,"run_pipeline")
    if int(run_parameters['run']) == 1:
        output_directory = run_parameters['output_directory']
        if not (output_directory.endswith("/")):
            output_directory += "/"
        output_directory += "features/"
        existing_PCA = 0
    else:
        existing_PCA = int(parsed_parameters['existing_PCA'])
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
    
    # If PCA exists load it, otherwise initialize it with given parameters
    if existing_PCA == 0:
        PCA_components = int(parsed_parameters['PCA_components'])
        PCA_compression = int(parsed_parameters['PCA_compression'])
        PCA = sklearn.decomposition.IncrementalPCA(n_components=PCA_components)
        print(f"PCA Successfully Initialized\n")
    else:
        PCA_path = parsed_parameters['PCA_path']
        PCA = joblib.load(PCA_path)
        print(f"PCA {PCA_path}Successfully Loaded")

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
        try:
        
            # parse annotations
            #
            header, ids, labels, coordinates = fileio_tools.parseAnnotations(annotation_file)

            print(f"File {i+1} of {len(image_files_list)} Processing DCT")
            
            # get height and width of image (in pixels) from the header
            #
            height = int(header['height'])
            width = int(header['width'])

            # get labeled regions
            #
            labeled_regions = feats_tools.labeledRegions(coordinates)
            
            # return top left coordinates of frames that have center coordinates in labels
            #
            frame_top_left_coordinates,frame_labels = feats_tools.classifyFrames(labels,height, width,
                                                                                 window_size, frame_size,
                                                                                 labeled_regions,
                                                                                 window_region_overlap_threshold)


            
            # get list of rgba values
            #
            window_RGBs = feats_tools.windowRGBValues(image_file,
                                                      frame_top_left_coordinates,
                                                      window_size)
            
            
            # perform dct on rgba values
            #
            window_DCTs = feats_tools.windowDCT(window_RGBs)
            
            append_dictionary = {
                "Header":header,
                "DCTs":window_DCTs,
                "Labels":frame_labels,
                "Top Left Coordinates":frame_top_left_coordinates,
                "Image File":image_file,
                "Annotation File":annotation_file,
                "Frame Size":(frame_size[0],frame_size[1])
            }
            
            finished_files.append(append_dictionary)

            feature_files_written.append(output_directory + header['bname'] + "_FEATS.csv")
            original_files_written.append(annotation_file)

            
            print(f"{header['bname']} DCT Succeeded")

            if existing_PCA == 0:
                DCTs_for_PCA.extend(window_DCTs)
                
                if len(DCTs_for_PCA) >= PCA_components:
                    try:
                        PCA.partial_fit(DCTs_for_PCA)
                        total_windows_PCA_trained_on+=len(DCTs_for_PCA)
                        print(f"PCA trained on {total_windows_PCA_trained_on}\n")
                        DCTs_for_PCA = []
                    except Exception as e:
                        print(f"Incremental PCA training Failed due to: \n{e}\n")

            

            
        except Exception as e:
            print(f"{header['bname']} DCT Failed due to: \n{e}\n")

    if existing_PCA == 0:
        joblib.dump(PCA,output_directory+"PCA.joblib",compress=PCA_compression)
        print(f"PCA {output_directory+'PCA.joblib'} successfully written")

    # create feature files schema
    features_header = []
    for i in range(PCA.n_components):
        features_header.append(f"PCA_From_DCT_Feature{i}")
        
    for i,finished_file in enumerate(finished_files):
        try:

            print(f"File {i+1} of {len(finished_files)} Processing PCA")

            finished_file['PCs'] = PCA.transform(finished_file['DCTs'])
            
            del finished_file['DCTs']
            
            if write_features == 1:

                labels_dataframe = pandas.DataFrame({'Label':finished_file['Labels']})

                coordinates_dataframe = pandas.DataFrame(finished_file['Top Left Coordinates'],
                                                         columns=['TopLeftColumn','TopLeftRow'])

                features_dataframe = pandas.DataFrame(finished_file['PCs'],columns=features_header)

                dataframe=labels_dataframe.join([coordinates_dataframe,features_dataframe])

                file_path = output_directory + finished_file['Header']['bname'] + "_FEATS.csv"
                
                with open(file_path,'w') as f:
                    for key,value in finished_file['Header'].items():
                        f.write(f'{key}:{value}\n')
                    f.write(f'frame_height:{frame_size[0]}\n')
                    f.write(f'frame_width:{frame_size[1]}\n')                    
                    f.write(f'window_height:{window_size[0]}\n')
                    f.write(f'window_width:{window_size[1]}\n')
                    
                    
                    dataframe.to_csv(f, index=False, header = True)

            
                print(f"File {i+1} of {len(finished_files)} Write Succeeded")

            print(f"{finished_file['Header']['bname']} PCA Succeeded\n")
            
        except Exception as e:
            print(f"{finished_file['Header']['bname']} PCA & Write Failed Due To\n{e}\n")
            
    with open(output_directory +"original_annotations.list",'w') as f:
        f.writelines(line + '\n' for line in original_files_written)
        print(f"Original list of annotation files {output_directory + 'original_annotations.list'} written")

    if write_features == 1:
        with open(output_directory +"feature_files.list",'w') as f:
            f.writelines(line + '\n' for line in feature_files_written)
        print(f"Generated list of feature files {output_directory+'feature_files.list'} written")

    return finished_files
        
if __name__ == "__main__":
    gen_feats()

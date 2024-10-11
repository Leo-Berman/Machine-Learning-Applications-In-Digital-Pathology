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
import nedc_mladp_ann_tools as ann_tools
import nedc_mladp_feats_tools as feats_tools

# import NEDC libraries
#
import nedc_file_tools


def main():

    # set argument parsing
    #
    args_usage = "nedc_mladp_gen_feats.usage"
    args_help = "nedc_mladp_gen_feats.help"
    parameter_file = fileio_tools.parseArguments(args_usage,args_help)

    # parse parameters
    #
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"gen_feats")
    PCA_components = int(parsed_parameters['PCA_components'])
    frame_region_overlap_threshold = float(parsed_parameters['frame_region_overlap_threshold'])
    window_x_size = int(parsed_parameters['window_x_size'])
    window_y_size = int(parsed_parameters['window_y_size'])
    window_size = (window_x_size,window_y_size)
    frame_x_size =  int(parsed_parameters['frame_x_size'])
    frame_y_size =  int(parsed_parameters['frame_y_size'])
    frame_size = (frame_x_size,frame_y_size)
    output_directory = parsed_parameters['output_directory']
    if not (output_directory.endswith("/")):
        output_directory = output_directory + "/"
    
    # read list of files
    #
    image_files_list = fileio_tools.readLines(parsed_parameters['image_files_list'])
    annotation_files_list = fileio_tools.readLines(parsed_parameters['annotation_files_list'])

    print(f"Parameter file {parameter_file} Parsed Successfully\n")


    os.makedirs(output_directory,exist_ok=True)
    
    finished_files = []
    

    PCA = sklearn.decomposition.IncrementalPCA(n_components=PCA_components)

    original_files_written = []
    feature_files_written = []
    
    # iterate through and create a feature vector file for each file
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
                                                                                 frame_region_overlap_threshold)
            
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
                "Annotation File":annotation_file
            }
            
            finished_files.append(append_dictionary)
            
            PCA.partial_fit(window_DCTs)
            
            print(f"{header['bname']} DCT Succeeded\n")
            
        except Exception:
            print(f"{header['bname']} DCT Failed\n")

    
    features_header = []
    for i in range(PCA_components):
        features_header.append(f"Feature{i}")
        
    for i,finished_file in enumerate(finished_files):
        try:

            print(f"File {i+1} of {len(finished_files)} Processing PCA & Write")

            
            finished_file['PCs'] = PCA.transform(finished_file['DCTs'])
            
            del finished_file['DCTs']
            
            labels_dataframe = pandas.DataFrame({'Label':finished_file['Labels']})
            coordinates_dataframe = pandas.DataFrame(finished_file['Top Left Coordinates'],
                                                     columns=['TopLeftX','TopLeftY'])
            features_dataframe = pandas.DataFrame(finished_file['PCs'],columns=features_header)
            dataframe=labels_dataframe.join([coordinates_dataframe,features_dataframe])
            
            
            file_path = output_directory + finished_file['Header']['bname'] + "_FEATS.csv"
            
            with open(file_path,'w') as f:
                for key,value in finished_file['Header'].items():
                    f.write(f'{key}:{value}\n')
                f.write(f'frame_size:frame_size\n')
                f.write(f'window_size:window_size\n')
                dataframe.to_csv(f, index=False, header = True)

            original_files_written.append(finished_file['Annotation File'])
            feature_files_written.append(file_path)
            print(f"{finished_file['Header']['bname']} PCA & Write Succeeded\n")
            
        except Exception:
            print(f"{finished_file['Header']['bname']} PCA & Write Failed\n")
            
    with open(output_directory +"original_annotations.list",'w') as f:
        f.writelines(line + '\n' for line in original_files_written)

    with open(output_directory +"feature_files.list",'w') as f:
        f.writelines(line + '\n' for line in feature_files_written)

    return finished_files
        
if __name__ == "__main__":
    main()

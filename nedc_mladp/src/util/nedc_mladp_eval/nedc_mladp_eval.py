#!/usr/bin/env python
#

# import python libraries
import joblib
import os

# import project specific libraries
import nedc_mladp_fileio_tools as fileio_tools
import nedc_mladp_eval_tools as eval_tools
import nedc_mladp_feats_tools as feats_tools
import nedc_mladp_ann_tools as ann_tools

# import NEDC libraries
import nedc_file_tools
import nedc_dpath_ann_tools

def main():

    # set argument parsing
    #
    args_usage = "nedc_mladp_eval.usage"
    args_help = "nedc_mladp_eval.help"
    parameter_file = fileio_tools.parseArguments(args_usage,args_help)

    # parse parameters
    #
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"eval_model")

    # list of feature files
    #
    features_data_list = parsed_parameters['features_data_list']

    # path to joblib model
    #
    model_path=parsed_parameters['model_path']

    # boolean of whether or not to write frames
    #
    write_frame_decisions=int(parsed_parameters['write_frame_decisions'])

    # path to where to write frame decisions
    #
    output_frame_decisions_directory=parsed_parameters['output_frame_decisions_directory']
    if not (output_frame_decisions_directory.endswith('/')):
        output_frame_decisions_path += '/'
        
    # boolean of whether or not to write regions
    #
    write_region_decisions=int(parsed_parameters['write_region_decisions'])

    # path to where to write region decisions
    #
    output_region_decisions_directory=parsed_parameters['output_region_decisions_directory']
    if not (output_region_decisions_directory.endswith('/')):
        output_region_decisions_directory += '/'
        
    # if run is set, update a couple paths to reflect
    #
    run_parameters = nedc_file_tools.load_parameters(parameter_file,"gen_feats")
    if run_parameters['run'] == 1:
        feature_data_list=run_params['output_list']
        model_path = nedc_file_tools.load_parameters(parameter_file,"train_model")['model_output_path']

    # load the model
    #
    model = joblib.load(model_path)        

    # read the features files list
    #
    features_files_list = fileio_tools.read_file_lists(features_data_list)

    # read all the files data and the header
    #
    files_data,headers = fileio_tools.read_feature_files(features_files_list, get_header = True)

    # iterate through each file
    #
    for data,current_file,header in zip(files_data,features_files_list,headers):

        # extract the data, labels, frame locations, and sizes
        #
        features = data[:, 4::]
        labels = data[:,0].tolist()
        frame_locations = data[:,1:3].tolist()
        frame_sizes=data[:,3]
            
        # generates a list of guess and their top level coordinates only applies to single image
        #
        file_frame_decisions_path = output_frame_decisions_directory+current_file.split('/')[-1][:-11]+"FRAME_DECISIONS.csv"

        # get the frame decisions
        #
        frame_decisions = eval_tools.generate_frame_decisions(model,features,file_frame_decisions_path,frame_locations,frame_sizes,header)

        # get the sparse matrixes
        #
        sparse_matrixes = ann_tools.coords_to_dict(frame_decisions)

        # generate a tuple of framesizes
        #
        framesize_fib = (int(frame_sizes[0]),int(frame_sizes[0]))

        # generate a heatmap of labels
        #
        heatmap = ann_tools.heatmap(sparse_matrixes,framesize_fib)
        
        # generate the regions
        #
        regions = eval_tools.generateRegionDecisions(heatmap,framesize_fib[0])

        # generate the header
        #
        ann_dpath_header = eval_tools.generateAnnotationsHeader(header)

        if write_region_decisions == 1:
            file_region_decisions_path = output_region_decisions_directory+current_file.split('/')[-1][:-11]+"REGION_DECISIONS.csv"
            annotation_writer = nedc_dpath_ann_tools.AnnDpath()
            annotation_writer.set_type("csv")
            annotation_writer.set_header(ann_dpath_header)
            annotation_writer.set_graph(regions)
            annotation_writer.write(file_region_decisions_path)
        
if __name__ == "__main__":
    main()

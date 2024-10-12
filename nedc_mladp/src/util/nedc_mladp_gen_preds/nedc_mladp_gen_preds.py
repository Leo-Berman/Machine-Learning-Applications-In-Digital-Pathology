#!/usr/bin/env python

# import python libraries
import joblib
import os
import pandas
import numpy

# import project specific libraries
import nedc_mladp_fileio_tools as fileio_tools
import nedc_mladp_pred_tools as pred_tools

# import NEDC libraries
import nedc_file_tools
import nedc_dpath_ann_tools

def gen_preds(feature_files:dict=None, model=None):

    # set argument parsing
    #
    args_usage = "nedc_mladp_gen_preds.usage"
    args_help = "nedc_mladp_gen_preds.help"
    parameter_file = fileio_tools.parseArguments(args_usage,args_help)

    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"gen_preds")

    write_region_decisions = int(parsed_parameters['write_region_decisions'])
    write_frame_decisions = int(parsed_parameters['write_frame_decisions'])

    run_parameters = nedc_file_tools.load_parameters(parameter_file,"run_pipeline")

    if int(run_parameters['run']) == 1:
        output_directory = run_parameters['output_directory']
        if not (output_directory.endswith("/")):
            output_directory += "/"
        output_directory += "predictions/"            
        original_files_list =  [file['Annotation File'] for file  in feature_files]
    else:
        output_directory = parsed_parameters['output_directory']
        if not (output_directory.endswith("/")):
            output_directory += "/"
        feature_files_list = parsed_parameters['feature_files_list']
        model_file = parsed_parameters['model_file']
        original_annotation_files_list = parsed_parameters['original_annotation_files_list']
        feature_files = []
        feature_files_list = fileio_tools.readLines(feature_files_list)
        original_files_list = fileio_tools.readLines(original_annotation_files_list)
            
        for feature_file,original_file in zip(feature_files_list,original_files_list):
            lines = [line.split(',') for line in fileio_tools.readLines(feature_file)]
            header_info = {}
            i = 0
            while ':' in lines[i][0]:
                key,value = lines.pop(0)[0].split(':')
                header_info[key]=value
            dataframe = pandas.DataFrame(lines[1:],columns=lines[0])
            labels = dataframe['Label'].to_list()
            top_left_coordinates = list(zip(dataframe['TopLeftRow'].to_list(),dataframe['TopLeftColumn'].to_list()))
            annotation_reader = nedc_dpath_ann_tools.AnnDpath()
            annotation_reader.load(original_file)
            header = annotation_reader.get_header()
            dataframe = dataframe.drop(['Label','TopLeftRow','TopLeftColumn'], axis=1)
            PCs = dataframe.to_numpy()
            
            append_dictionary = { 'Frame Decisions':labels,
                                  'Top Left Coordinates':top_left_coordinates,
                                  'Header':header,
                                  'PCs':PCs,
                                  'Frame Size':(int(header_info['frame_height']),int(header_info['frame_width'])),
                                 }
            feature_files.append(append_dictionary)
        
    regions_output_directory = output_directory + 'regions/'
    frames_output_directory = output_directory + 'frames/'
    os.makedirs(regions_output_directory,exist_ok=True)
    os.makedirs(frames_output_directory,exist_ok=True)

    if model is None:
        model = joblib.load(model_file)

             
    region_decision_files = []
    frame_decision_files = []
    for i,feature_file in enumerate(feature_files):
        
        feature_file['Frame Confidences'] = [max(predictions) for predictions in model.predict_proba(numpy.array(feature_file['PCs']))]
        feature_file['Frame Decisions'] = model.predict(numpy.array(feature_file['PCs']))
        prediction_graph = pred_tools.regionPredictions(feature_file['Frame Decisions'],
                                                        feature_file['Top Left Coordinates'],
                                                        feature_file['Frame Confidences'],
                                                        feature_file['Frame Size'])
        if write_region_decisions == 1:
            annotation_writer = nedc_dpath_ann_tools.AnnDpath()
            annotation_writer.set_type("csv")
            #annotation_writer.set_type("xml")
            annotation_writer.set_header(feature_file['Header'])
            annotation_writer.set_graph(prediction_graph)
            output_filepath = regions_output_directory+feature_file['Header']['bname']+"_REGIONDECISIONS.csv"
            annotation_writer.write(output_filepath)
            region_decision_files.append(output_filepath+'\n')

        print(f"{i+1} of {len(feature_files)} Decisions Generated") 

    if write_region_decisions == 1:
        with open(output_directory+'regions/region_decisions_files.list','w') as f:
            f.writelines(region_decision_files)
        with open(output_directory+'regions/original_annotation_files.list','w') as f:
            f.writelines([file +'\n' for file in original_files_list])
            
if __name__ == "__main__":
    gen_preds()

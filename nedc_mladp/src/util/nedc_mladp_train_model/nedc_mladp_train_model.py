#!/usr/bin/env python
#
# import python libraries
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier as RNF
from sklearn.svm import SVC as SVM
import os
import joblib
import numpy
import pandas

# import project specific libraries
import nedc_mladp_fileio_tools as fileio_tools

# import NEDC libraries
import nedc_file_tools

def train_model(feature_files:dict=None):

    # set argument parsing
    #
    args_usage = "nedc_mladp_train_model.usage"
    args_help = "nedc_mladp_train_model.help"
    parameter_file = fileio_tools.parseArguments(args_usage,args_help)

    # parse parameters
    #
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"train_model")
    model_type=parsed_parameters['model_type']
    write_model=int(parsed_parameters['write_model'])
    
    run_parameters = nedc_file_tools.load_parameters(parameter_file,"run_pipeline")
    if int(run_parameters['run']) == 1:
        output_directory = run_parameters['output_directory']
        if not (output_directory.endswith("/")):
            output_directory += "/"
        output_directory += "model/"
        labels = []
        train_data = None
        for file in feature_files:
            if train_data is None:
                train_data = numpy.array(file['PCs'])
            else:
                train_data = numpy.vstack([train_data,numpy.array(file['PCs'])])

            labels.extend(file['Labels'])
    else:
        output_directory = parsed_parameters['output_directory']
        if not (output_directory.endswith("/")):
            output_directory += "/"
        feature_files_list = parsed_parameters['feature_files_list']
        feature_files = fileio_tools.readLines(feature_files_list)
        labels = []
        train_data = None

        for file in feature_files:
            lines = [line.split(',') for line in fileio_tools.readLines(file) if ':' not in line]
            dataframe = pandas.DataFrame(lines[1:],columns=lines[0])
            labels.extend(dataframe['Label'].to_list())
            dataframe = dataframe.drop(['Label','TopLeftRow','TopLeftColumn'],axis=1)
            if train_data is None:
                train_data=dataframe.to_numpy()
            else:
                train_data = numpy.vstack([train_data,dataframe.to_numpy()])
                
    os.makedirs(output_directory,exist_ok=True)

            
    # Fit the model
    #
    model = None
    if model_type == "QDA":
        model = QDA()
    elif model_type == "RNF":
        model = RNF()
    elif model_type == "SVM":
        model = SVM()
    else:
        print("No model supplied")
        return
    model.fit(train_data, labels)

    if write_model == 1:
        compression=int(parsed_parameters['compression'])
        joblib.dump(model,output_directory+model_type+'.joblib',compress=compression)

    return model

if __name__ == "__main__":
    train_model()

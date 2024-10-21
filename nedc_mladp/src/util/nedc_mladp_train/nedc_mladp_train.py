#!/usr/bin/env python
#

# import python libraries
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier as RNF
from sklearn.svm import SVC as SVM
import os
import joblib
import numpy


# import project specific libraries
import nedc_mladp_fileio_tools as fileio_tools
import nedc_mladp_feats_tools as feats_tools
import nedc_model_cnn as cnn

# import NEDC libraries
import nedc_file_tools

import sys
sys.path.append('/home/tul16619/SD1/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/src/functs/nedc_mladp_model_tools')
import nedc_mladp_model_tools as model_tools

def main():

    # set argument parsing
    #
    args_usage = "nedc_mladp_train_model.usage"
    args_help = "nedc_mladp_train_model.help"
    parameter_file = fileio_tools.parameters_only_args(args_usage,args_help)

    # parse parameters
    #
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"train_model")
    model_type=parsed_parameters['model_type']
    feature_data_list=parsed_parameters['data_list']
    output_model_directory=parsed_parameters['model_output_path']
    if not (output_model_directory.endswith('/')):
        output_model_directory=output_model_directory + "/"
    input_type=int(parsed_parameters['input_direction'])
    # compression=int(parsed_parameters['compression'])
    # even_data = int(parsed_parameters['even_data'])
    
    # If run parameter is set high then get the feature data list from gen feats
    run_params = nedc_file_tools.load_parameters(parameter_file,"gen_feats")
    if run_params['run']==1:
        feature_data_list=run_params['output_list']
    
    # parse the feature files and combine the data
    #
    train_list = fileio_tools.read_file_lists(feature_data_list)
    totaldata = model_tools.parsePCA(train_list) # for PCA features

    # split the data
    #
    labels = totaldata[:,0]
    data = totaldata[:,1:]
    

    
    # If even data is set than normalize the number of labels
    # if even_data == 1:
    #     mydata,labels = feats_tools.even_data(mydata,labels)

    # Fit the model
    #
    model = None
    if model_type == "RNF":
        model = RNF()
    elif model_type == "CNN":
        cnn.trainModel(data,labels)
    else:
        print("No model supplied")
        return
    # model.fit(mydata, labels)

    # # change the directory to output the model
    # #
    # os.chdir(output_model_directory)

    # # dump the model there
    # #
    # joblib.dump(model,'Trained_'+model_type+'.joblib',compress=compression)

    

if __name__ == "__main__":
    main()

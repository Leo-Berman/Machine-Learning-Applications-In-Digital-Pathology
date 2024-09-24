#!/usr/bin/env python
#

# import python libraries
import joblib

# import project specific libraries
import nedc_mladp_fileio_tools as fileio_tools
import nedc_mladp_eval_tools as eval_tools
import nedc_mladp_feats_tools as feats_tools
import nedc_mladp_ann_tools as ann_tools

# import NEDC libraries
import nedc_file_tools

def main():

    # set argument parsing
    #
    args_usage = "nedc_mladp_eval.usage"
    args_help = "nedc_mladp_eval.help"
    parameter_file = fileio_tools.parameters_only_args(args_usage,args_help)

    # parse parameters
    #
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"eval_model")
    feature_data_list=parsed_parameters['data_list']
    model_path=parsed_parameters['model']
    run_params = nedc_file_tools.load_parameters(parameter_file,"gen_feats")
    if run_params['run']==1:
        feature_data_list=run_params['output_list']
        model_path = nedc_file_tools.load_parameters(parameter_file,"train_model")['model_output_path']    
    generate_decisions=int(parsed_parameters['decisions'])
    decisions_path=parsed_parameters['output_decisions_path']
    if not (decisions_path.endswith('/')):
        decisions_path = decisions_path + '/'
    generate_histogram = int(parsed_parameters['label_histogram'])                          
    histogram_output=parsed_parameters['output_histogram_path']
    even_data = int(parsed_parameters['even_data'])
    confusion_matrix_path=parsed_parameters['output_confmat_path']
    generate_confusion_matrix=int(parsed_parameters['confusion_matrix'])

    # load the model
    #
    model = joblib.load(model_path)        

    # read the features files list
    #
    feature_files_list = fileio_tools.read_file_lists(feature_data_list)

    # read all the files data and the header
    #
    filesdata,headers = fileio_tools.read_feature_files(feature_files_list, get_header = True)

    # iterate through each file
    #
    for data,currfile,header in zip(filesdata,feature_files_list,headers):

        # extract the data, labels, frame locations, and sizes
        #
        mydata = data[:, 4::]
        labels = data[:,0].tolist()
        frame_locations = data[:,1:3].tolist()
        framesizes=data[:,3]
        
        # even the data out
        #
        if even_data == 1:
            mydata,labels = feats_tools.even_data(mydata,labels)
            
        # generate confusion matrix
        #
        #if generate_confusion_matrix == 1:
        #    eval_tools.plot_confusion_matrix(model,labels,mydata,confusion_matrix_path)

        # generates a list of guess and their top level coordinates only applies to single image
        #
        file_decision_path = decisions_path+currfile.split('/')[-1][:-11]+"DECISIONS.csv"
        print(file_decision_path)

        # get the frame decisions
        #
        frame_decisions = eval_tools.generate_frame_decisions(model,mydata,file_decision_path,frame_locations,framesizes,header)

        # get the sparse matrixes
        #
        sparse_matrixes = ann_tools.coords_to_dict(frame_decisions)

        # generate a tuple of framesizes
        #
        framesize_fib = (int(framesizes[0]),int(framesizes[0]))

        # generate a heatmap of labels
        #
        heatmap = ann_tools.heatmap(sparse_matrixes,framesize_fib)

        regions = eval_tools.generate_region_decisions(heatmap,framesize_fib[0])
        
        print(regions)
        
        #if generate_histogram == 1:
        #    eval_tools.plot_histogram(labels,histogram_output)

    # print the error rate and mean confidence %
    #
    # print("Accuracy rate = ",model.score(mydata,labels))
    # print("Mean confidence %",eval_tools.mean_confidence(model,mydata))
    

if __name__ == "__main__":
    main()

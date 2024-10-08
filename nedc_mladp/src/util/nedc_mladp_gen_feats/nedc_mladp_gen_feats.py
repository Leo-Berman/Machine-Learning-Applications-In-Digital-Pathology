#!/usr/bin/env python

# import python libraries
#
import os
import polars

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
    parameter_file = fileio_tools.parameters_only_args(args_usage,args_help)

    # parse parameters
    #
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"gen_feats")
    windowsize = int(parsed_parameters['windowsize'])
    framesize =  int(parsed_parameters['framesize'])
    output_path = parsed_parameters['output_dir']
    if not (output_path.endswith("/")):
        output_path = output_path + "/"
    output_txt_file = parsed_parameters['output_list']

    # read list of files
    #
    svs_list = fileio_tools.read_file_lists(parsed_parameters['imagefile_list'])
    csv_list = fileio_tools.read_file_lists(parsed_parameters['labelfile_list'])



    
    # iterate through and create a feature vector file for each file
    #
    list_of_files=[]
    for svs,csv in zip(svs_list,csv_list):

        # parse annotations
        #
        header, ids, labels, coordinates = fileio_tools.parse_annotations(csv)
        
        # get height and width of image (in pixels) from the header
        #
        height = int(header['height'])
        width = int(header['width'])

        # get labeled regions
        #
        labeled_regions = ann_tools.labeled_regions(coordinates)

        # return top left coordinates of frames that have center coordinates in labels
        #
        labeled_frames,frame_labels = ann_tools.labeled_frames(labels,height,width,windowsize,framesize,labeled_regions)

        # return top left coordinates of frames that overlap 50% or more with labels
        #
        # labeled_frames,frame_labels = nedc_regionid.classify_frame(svs,framesize,labels,labeled_regions)
        
        # get list of rgba values
        #
        frame_rgbas = ann_tools.frame_rgba_values(svs,frame_labels,labeled_frames,windowsize)

        # perform dct on rgba values
        #
        frame_dcts = feats_tools.rgba_to_dct(frame_rgbas,labeled_frames,framesize,windowsize)
        if len(frame_dcts) > 0:

            
            # set column index names
            #
            my_schema = []
            for i in range(len(frame_dcts[0])):
                if i == 0:
                    my_schema.append('label')
                elif i == 1:
                    my_schema.append('top_left_corner_x_coord')
                elif i == 2:
                    my_schema.append('top_left_corner_y_coord')
                elif i == 3:
                    my_schema.append('framesize')
                else:
                    my_schema.append(str(i))
                    
            # print dct frames to csv
            #
            df = polars.DataFrame(frame_dcts,schema=my_schema,orient="row")
            ifile,iextension = os.path.splitext(os.path.basename(os.path.normpath(svs)))
            write_path = output_path+ifile+"_RGBADCT.csv"

            if os.path.exists(write_path):
                os.remove(write_path)
            
            with open(write_path, 'a') as file:
                file.write('# version = mladp_v1.0.0\n'+
                      '# MicronsPerPixel = '+header['MicronsPerPixel']+'\n'+
                      '# bname = '+header['bname']+'\n'+
                      '# width = '+header['width']+', height = '+header['height']+'\n'+
                      '# tissue = '+", ".join(header['tissue'])+'\n'+
                      '# \n% ')
            
                df.write_csv(file)
                print(csv, "Sucess")
            list_of_files.append(write_path)
        else:
            print(csv, "Failed")

    if os.path.exists(output_txt_file):
        os.remove(output_txt_file)

            
    f = open(output_txt_file,"a")
    for x in list_of_files:
        f.write(x+'\n')
    f.close()
if __name__ == "__main__":
    main()

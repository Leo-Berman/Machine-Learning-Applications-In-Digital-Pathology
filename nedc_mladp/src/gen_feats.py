# import python libraries
#
import sys
import pandas
import os
import polars

# our libraries
#
sys.path.append("../lib")
import nedc_fileio
import nedc_regionid
import nedc_svsfeatures

# picones libraries
#
sys.path.append("/data/isip/tools/linux_x64/nfc/class/python/nedc_sys_tools")
import nedc_file_tools

# read lists of files in
#
def read_file_lists(file_name):
    df = pandas.read_csv(file_name)
    return df.columns.to_list()

def main():

    # set argument parsing
    #
    args_usage = "usagefiles/gen_feats_usage.txt"
    args_help = "helpfiles/gen_feats_help.txt"
    parameter_file = nedc_fileio.parameters_only_args(args_usage,args_help)

    # parse parameters
    #
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"gen_feats")
    windowsize = int(parsed_parameters['windowsize'])
    framesize =  int(parsed_parameters['framesize'])
    output_path = parsed_parameters['output_file']

    # read list of files
    #
    svs_list = read_file_lists(parsed_parameters['imagefile_list'])
    csv_list = read_file_lists(parsed_parameters['labelfile_list'])

    # iterate through and create a feature vector file for each file
    #
    for svs,csv in zip(svs_list,csv_list):

        # parse annotations
        #
        header, ids, labels, coordinates = nedc_fileio.parse_annotations(csv)
        
        # get height and width of image (in pixels) from the header
        #
        height = int(header['height'])
        width = int(header['width'])

        # get labeled regions
        #
        labeled_regions = nedc_regionid.labeled_regions(coordinates)

        # return top left coordinates of frames that have center coordinates in labels
        #
        # labeled_frames,frame_labels = nedc_regionid.labeled_frames(labels,height,width,windowsize,framesize,labeled_regions)

        # return top left coordinates of frames that overlap 50% or more with labels
        #
        labeled_frames,frame_labels = nedc_regionid.classify_frame(svs,framesize,labels,labeled_regions)
        
        # get list of rgba values
        #
        frame_rgbas = nedc_regionid.frame_rgba_values(svs,frame_labels,labeled_frames,windowsize)

        # perform dct on rgba values
        #
        frame_dcts = nedc_svsfeatures.rgba_to_dct(frame_rgbas,labeled_frames,framesize)

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
        df = polars.DataFrame(frame_dcts,schema=my_schema)
        ifile,iextension = os.path.splitext(os.path.basename(os.path.normpath(svs)))
        write_path = output_path+ifile+"_RGBADCT.csv"
        df.write_csv(write_path)


if __name__ == "__main__":
    main()
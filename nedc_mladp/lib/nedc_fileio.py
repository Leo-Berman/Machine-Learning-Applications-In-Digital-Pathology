import sys

sys.path.append('/data/isip/tools/linux_x64/nfc/class/python/nedc_ann_dpath_tools')
import nedc_ann_dpath_tools as nadt

sys.path.append("/data/isip/tools/linux_x64/nfc/class/python/nedc_sys_tools")
import nedc_cmdl_parser

import pandas

import csv

import numpy

def read_feature_files(feature_file_list:list):
    # lists for holding the labels and data
    #
    mydata = []
    labels = []

    # iterate through the entire training list
    #
    for x in feature_file_list:

        # rea
        with open (x) as file:
            reader = csv.reader(file)
            next(reader,None)
            for row in reader:
                row_list = list(row)
                labels.append(row_list.pop(0))
                mydata.append([float(x) for x in row_list])

    # reshape the arrays
    #
    labels = numpy.array(labels).ravel()
    mydata = numpy.array(mydata)

    return labels,mydata

def parameters_only_args(usage,help):
    argparser = nedc_cmdl_parser.Cmdl(usage,help)
    argparser.add_argument('-p', type = str)
    parsed_args = argparser.parse_args()
    parameter_file = parsed_args.p
    return parameter_file

def read_list_of_files(csv_file:str):
    return(pandas.read_csv(csv_file).to_list())

def parse_annotations(file):
    
    ''' 
    USE:
    file = file path of xml or csv with this format:
        index,region_id,tissue,label,coord_index,row,column,depth,confidence
    
    with use:
        headers,region_ids,label_names,coordinates = parse_annotations(file)

    headers will be a dictionary:
        header: {
            'MicronsPerPixel' : microns_value (str),
            'height' : height_value (str),
            'width' : width_value (str)
        }
    
    region_id will be a list:
        [label number of 1st boundary, label number of 2nd boundary, ..., label number of nth boundary]

    label_name will be a list:
        [label name of 1st boundary, label name of 2nd boundary, ..., label number of nth boundary]

    coordinates will be a list of list of lists:

        [
            [[boundary1x1,boundary1y1], [boundary1x2,boundary1y2], ..., [boundary1xn,boundary1yn]],
            [[boundary2x1,boundary2y1], [boundary2x2,boundary2y2], ..., [boundary2xn,boundary2yn]],
            ...
            [[boundarynx1,boundaryny1], [boundarynx2,boundaryny2], ..., [boundarynxn,boundarynyn]]
            
        ]
    '''

    # read the data this uses the nedc_ann_dpath_tools library which 
    # reads the data with coords in row,column format
    #
    header,data = nadt.read(file)
    
    # create lists to contain each labelled boundary
    # region ids: (numeric form of labels),
    # text: label names for graphic purposes
    # coords: traditional x,y coordinates for each boundary
    #
    region_ids = []
    labels = []
    coords = []
    
    # iterate through the data
    #
    for i in data:

        # convert region id into an int
        #
        region_ids.append(int(data[i]['region_id']))
        
        # append label name
        #
        labels.append(data[i]['text'])

        # append list of lists row,column,depth coordinates
        #
        coords.append(data[i]['coordinates'])

    # iterate through the lists of list of coordinates
    #
    for i in range(len(coords)):

        # iterate through the list of coordinates
        #
        for j in range(len(coords[i])):

            # convert from row,column,depth into traditional (x,y) coordinates
            #
            coords[i][j][0] = int(header['height'])-coords[i][j][0]
            coords[i][j].pop()
            coords[i][j].reverse()

    # return the header, numeric region ids, label names, and coordinates in x,y format
    #
    return header,region_ids,labels,coords


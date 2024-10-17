# Import python libraries

import csv
import numpy
import os

# picone's libraries
#
import nedc_dpath_ann_tools
import nedc_cmdl_parser

# rerad lists of files in
#
def readLines(file_name):
    with open(file_name, 'r') as f:
        Lines = [line.strip() for line in f.readlines()]
    return Lines

def readDecisions(file_name):
    # Using readlines()
    file1 = open(file_name, 'r')
    Lines = file1.readlines()
    ret = [x.strip().split(',') for x in Lines]
    return ret

# set cmdl to only process a parameter file
#
def parseArguments(usage,help):
    parent_path = os.environ.get('MLADP') + '/nedc_mladp/src/util/' + help[:-5] + '/'    
    argument_parser = nedc_cmdl_parser.Cmdl(parent_path + usage,parent_path + help)
    #argument_parser.print_usage()
    argument_parser.add_argument('-p', type = str)
    parsed_args = argument_parser.parse_args()
    parameter_file = parsed_args.p
    return parameter_file

def parseAnnotations(file):
    
    
    annotation_tools = nedc_dpath_ann_tools.AnnDpath()
    annotation_tools.load(file)
    header = annotation_tools.get_header()
    
    data = annotation_tools.get_graph()

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

        # switch from x y to row column
        #
        coords.append(data[i]['coordinates'])
        #coords.append(data[i]['coordinates'])


        
    # return the header, numeric region ids, label names, and coordinates in x,y format
    #
    return header,region_ids,labels,coords


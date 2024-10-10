# Import python libraries
#
import csv
import numpy
import os

# picone's libraries
#
import nedc_dpath_ann_tools as nadt
import nedc_cmdl_parser

# read lists of files in
#
def readFileLists(file_name):
    # Using readlines()
    file1 = open(file_name, 'r')
    Lines = file1.readlines()
    ret = [x.strip() for x in Lines]
    return ret

def read_decisions(file_name):
    # Using readlines()
    file1 = open(file_name, 'r')
    Lines = file1.readlines()
    ret = [x.strip().split(',') for x in Lines]
    return ret

def read_feature_files(feature_file_list:list,get_header=False):
    
    # lists for holding the labels, data, top left corner of frames, and framesizes
    #
    myfiles = []
    
    # iterate through the entire training list and read the data into the appropriate lists
    #
    headers = []
    header = ''
    for x in feature_file_list:
        with open (x) as file:
            reader = csv.reader(file)
            labels = []
            xcoords = []
            ycoords = []
            framesizes = []
            features = []
            for row in reader:
                if list(row)[0].startswith('#'):
                    header += ','.join(list(row)) + '\n'
                elif list(row)[0].startswith('%'):
                    headers.append(header)
                    header = ''
                else:
                    row_list = list(row)
                    labels.append(row_list.pop(0)) # label
                    
                    xcoords.append(row_list.pop(0)) # x coord
                    ycoords.append(row_list.pop(0)) # y coord
                    framesizes.append(row_list.pop(0)) # framesize
                    features.append([float(x) for x in row_list]) # append the data
                    
            numpy_array = numpy.transpose(numpy.array([labels,xcoords,ycoords,framesizes]))
            numpy_feature_array = numpy.array(features)
            numpy_array = numpy.concatenate([numpy_array,numpy_feature_array],axis =1)
            myfiles.append(numpy_array)

    if(get_header == False):
        # return the appropriate data
        #
        return myfiles
    return myfiles,headers

# set cmdl to only process a parameter file
#
def parseArguments(usage,help):
    parent_path = os.environ.get('MLADP') + '/nedc_mladp/src/util/' + help[:-5] + '/'

    
    argument_parser = nedc_cmdl_parser.Cmdl(parent_path + usage,parent_path + help)
    argument_parser.print_usage()
    argument_parser.add_argument('-p', type = str)
    parsed_args = argument_parser.parse_args()
    parameter_file = parsed_args.p
    return parameter_file

def parseAnnotations(file):
    
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


    #initialize csv class
    annotation_tools = nadt.Xml()
    annotation_tools.load(file)
    header = annotation_tools.get_header()
    
    data = annotation_tools.get_graph()
    #header,data = nadt.load(file)
    
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
            tmp1 = coords[i][j][0]
            
            coords[i][j][0] = int(header['height']) - coords[i][j][1]
            coords[i][j][1] = tmp1

            coords[i][j].pop()
            coords[i][j].reverse()

    # return the header, numeric region ids, label names, and coordinates in x,y format
    #
    return header,region_ids,labels,coords


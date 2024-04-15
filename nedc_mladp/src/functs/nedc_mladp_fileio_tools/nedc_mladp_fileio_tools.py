# Import python libraries
#
import csv
import numpy

# picone's libraries
#
import nedc_ann_dpath_tools as nadt
import nedc_cmdl_parser

# read lists of files in
#
def read_file_lists(file_name):
    """
        Objective:
            Reads textfile of files and returns a list of all files.

        :param file_name: Name of the textfile.
        :type file_name: string
        
        :return: List of files extracted from textfile.
        :rtype: list of strings
    """
    # Using readlines()
    file1 = open(file_name, 'r')
    Lines = file1.readlines()
    ret = [x.strip() for x in Lines]
    return ret

def read_decisions(file_name):
    """
        Objective:
            Reads decisions of the file.

        :param file_name: Name of the textfile.
        :type file_name: string
        
        :return: -
        :rtype: -
    """

    # Using readlines()
    file1 = open(file_name, 'r')
    Lines = file1.readlines()
    ret = [x.strip().split(',') for x in Lines]
    return ret

def read_feature_files(feature_file_list:list):
    """
        Objective:
            Reads all features of each file.

        :param file_name: Name of the textfile.
        :type file_name: list
        
        :return: -
        :rtype: -
    """

    # lists for holding the labels, data, top left corner of frames, and framesizes
    #
    mydata = []
    labels = []
    frame_locations = []
    framesizes = []

    # iterate through the entire training list and read the data into the appropriate lists
    #
    for x in feature_file_list:
        with open (x) as file:
            reader = csv.reader(file)
            next(reader,None)
            for row in reader:
                row_list = list(row)
                labels.append(row_list.pop(0))
                xcoord=row_list.pop(0)
                ycoord=row_list.pop(0)
                frame_locations.append((xcoord,ycoord))
                framesizes.append(row_list.pop(0))
                mydata.append([float(x) for x in row_list])

    # reshape the arrays
    #
    labels = numpy.array(labels).ravel()
    mydata = numpy.array(mydata)

    # return the appropriate data
    #
    return labels,mydata,frame_locations,framesizes

# set cmdl to only process a parameter file
#
def parameters_only_args(usage,help):
    """
        Objective:
            Reads all features of each file.

        :param usage: File for usage reference.
        :type usage: string

        :param usage: File for help reference.
        :type usage: string
        
        :return: File of parameters.
        :rtype: string
    """
    argparser = nedc_cmdl_parser.Cmdl(usage,help)
    argparser.add_argument('-p', type = str)
    parsed_args = argparser.parse_args()
    parameter_file = parsed_args.p
    return parameter_file

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


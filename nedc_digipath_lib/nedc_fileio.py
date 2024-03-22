import numpy
import sys
sys.path.append('/data/isip/tools/linux_x64/nfc/class/python/nedc_ann_dpath_tools')
import nedc_ann_dpath_tools as nadt

import sys
import csv
import numpy as np
import scipy
import pandas
sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")
# Import phuykongs library
import nedc_image_tools as phg
import os

def parse_parameters(parameter_file:str):
    # define default argument values
    DEF_FRAMESIZE = int(-1)
    DEF_WINDOWSIZE = int(-1)
    # DEF_LVL = int(0)
    # DEF_XOFF = float(0)
    # DEF_YOFF = float(0)

    # search for the specific parameter through the parameter file 
    def parameter_search(parameter, parameter_file):
        # read the parameter file
        with open(parameter_file, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            # look for parameter
            for row in csv_reader:
                if row[0].startswith(parameter):
                    # if there is no value associated with the parameter, set to default value
                    if row[1] is None:
                        return default_value(parameter)
                    # if there is a value associated with the parameter, return the value
                    else:
                        return row[1]
                
    def default_value(parameter):
        if parameter == "framesize":
            value = DEF_FRAMESIZE

        elif parameter == "windowsize":
            value = DEF_WINDOWSIZE
        
        # elif parameter == "level":
        #     value = DEF_LVL
        
        # elif parameter == "xoff":
        #     value = DEF_XOFF
        
        # elif parameter == "yoff":
        #     value = DEF_YOFF

    def parse_param(parameter_file):
        param1 = "framesize"
        framesize = parameter_search(param1, parameter_file)

        param2 = "windowsize"
        windowsize = parameter_search(param2, parameter_file)

        # param3 = "level"
        # level = parameter_search(param3, parameter_file)

        # param4 = "xoff"
        # xoff = parameter_search(param4, parameter_file)

        # param5 = "yoff"
        # yoff = parameter_search(param5, parameter_file)

        param6 = "imagefile_list"
        imagefile_list = parameter_search(param6, parameter_file)

        param7 = "labelfile_list"
        labelfile_list = parameter_search(param7, parameter_file)

        return (imagefile_list, labelfile_list, int(windowsize), int(framesize))


def read_list_of_files(csv_file:str):
    return(pandas.read_csv(csv_file).to_list())


# parse annotations
#
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

def svs_windows_to_RGBA(image_file:str,labels:list,coords:list = [(0,0)], window_frame:list = [50,50],name:str=""):
    # open the imagefile
    NIL = phg.Nil()
    NIL.open(image_file)
    xdim,ydim = NIL.get_dimension()
    
    window = NIL.read_data_multithread(coords,npixx = window_frame[0],npixy = window_frame[1],color_mode="RGBA")
    window_list = []

    # save all the images as JPEGS
    for i in range(len(window)):
        workwindow = [labels[i]]
        for j in window[i]:
            for k in j:
                workwindow.extend(k.tolist())
        window_list.append(workwindow)


    return window_list

def RGBA_to_dct(framelist,imagename):
    
    list_of_rows = []
    for framevalues in framelist:        
        red = []
        green = []
        blue = []
        alpha = []

        # Append corresponding list values in separate RGBA lists
        #

        for i in range(1,len(framevalues)-1,4):
            red.append(framevalues[i])
            green.append(framevalues[i+1])
            blue.append(framevalues[i+2])
            alpha.append(framevalues[i+3])

        # concatenate the dcts of each vector
        # probably need to index each DCT for the most
        # signifcant terms
        #

        vector = []
        vector.extend(scipy.fftpack.dct(red)[0:10])
        vector.extend(scipy.fftpack.dct(green)[0:10])
        vector.extend(scipy.fftpack.dct(blue)[0:10])
        vector.extend(scipy.fftpack.dct(alpha)[0:10])

        # Convert vector to numpy array
        vector_numpy = np.array(vector).tolist()

        vector_numpy.insert(0,framevalues[0])

        list_of_rows.append(vector_numpy)

    ifile,iextension = os.path.splitext(imagename)
    
    file = open(ifile+"_RGBAVALUES.csv",'w')
    writer = csv.writer(file)
    writer.writerow(list_of_rows)
    pass

def parse_annotations(file):
    
    ''' USE:
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
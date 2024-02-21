from PIL import Image
import nedc_image_tools as phg
import sys
import csv
import re

sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")
labelfile = "/data/isip/data/tuh_dpath_breast/deidentified/v3.0.0/svs/train/00466205_aaaaaage/s000_2015_03_01/breast/00466205_aaaaaage_s000_0hne_0000_a001_lvl002_t000.csv"
#labelfile = "/home/tul16619/SD1/Machine-Learning-Applications-In-Digital-Pathology/nedc_pyprint_image/random_test.csv"
imgfile =0

def parse_labels(file):
    '''
    Get the label information and use that to show where borders are
    return label info this could be a zipped list of shapes and their 
    coordinates tbd up 2 u
    '''

    # initialize empty lists
    image_size = []     # holds the width (idx=0) and height (idx=1) in pixels
    index = []          # empty list for all index values
    region_id = []      # empty list for all region id values
    tissue = []         # empty list for all tissue type
    label = []          # empty list for all labels of regions
    coord_index = []    # empty list for all coodinate index values
    row = []            # empty list for all row values
    column = []         # empty list for all column values
    depth = []          # empty list for all depth values
    confidence = []     # empty list for all confidence values
    headers = [index, region_id, tissue, label, coord_index, row, column, depth, confidence]

    # open the CSV file in read mode
    with open(labelfile, 'r', newline='') as lfile:
        csvreader = csv.reader(lfile)

        # extract headers and data
        for row_index,r in enumerate(csvreader):
            # extract image size data
            if row_index == 3:
                for cell in r:
                    pixel = re.findall('\d+', str(cell))
                    image_size.append(int(pixel[0]))
            # extract data and store into corresonding lists
            elif row_index >= 7:
                for cell_index,cell in enumerate(r):
                    value = str(cell)
                    headers[cell_index].append(value)
    lfile.close()

    # convert each string-list to the correct type except for tissue and label
    index = list(map(int,index))
    region_id = list(map(int,region_id))
    coord_index = list(map(int,coord_index))
    row = list(map(int,row))
    column = list(map(int,column))
    depth = list(map(int,depth))
    confidence = list(map(float,confidence))

    width = image_size[0]
    height  = image_size[1]

    return headers, width, height

def classify_center(imgfile,labelfile,framesize = -1):

    labels = parse_labels(labelfile)
    NIL = phg.Nil()
    NIL.open(imgfile)
    
    # Get dimensions
    xdim,ydim =NIL.get_dimension()
    
    #print("Dimensions = ",xdim,ydim)
    

    if framesize == -1:
        # frame = [xdim,ydim]
        pass
    else:
        frame = [framesize,framesize]


    # Get all the coordinates for each windows
    # coordinates = [(x, y) for x in range(0, xdim, frame[0]) for y in range(0, ydim, frame[1])]
    
    # Read all the windows for each coordinate WINDOW_FRAME x WINDOW_FRAME
    # windows = NIL.read_data_multithread(coordinates, window_frame[0], window_frame[1])
    
    # save all the images as JPEGS
    '''
    for x in coordinates:
        if (x + frame[0]/2,x + frame[1]/2) in label:
            print("frame (upper left corner) is (label type))
    

    if processed:
        return True
    else:
        return False
        '''
    
classify_center(imgfile, labelfile)

import nedc_image_tools as phg
import sys
import annotations
import pointwithin
import matplotlib.pyplot as plt
from svstojpg import svs_to_jpg as stj
from shapely.geometry import Point

sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")
labelfile = "/data/isip/data/tuh_dpath_breast/deidentified/v2.0.0/svs/train/00707578/s000_2015_04_01/breast/00707578_s000_0hne_0000_a004_lvl000_t000.csv"
imagefile = "/data/isip/data/tuh_dpath_breast/deidentified/v2.0.0/svs/train/00707578/s000_2015_04_01/breast/00707578_s000_0hne_0000_a004_lvl000_t000.svs"



def get_center_frame(height, width, framesize):
    '''get the center coodinates of the top-left-most frame'''

    # print("height:", height, ", width:", width)
    center_x = 0 + framesize/2
    center_y = height - framesize/2
    center = Point(center_x, center_y)
    # print("center coodinate of top-left frame:", center)

    return center

def classification(labels, height, width, framesize, regions):
    '''
    objective: classify whether the center fo the first frame (top-left-most) is within a labeled region
        if within labeled region -> set coordinate to 'labeled' status
        if not within labeled region -> set coordinate to 'unlabeled' status
        status is set by storing these coordinates into
    '''

    def reposition(coord):
        # if the next center coordinate is out of bounds of the image (towards the right)
        if (coord.x + framesize) > width:
            # if the next center coordinate is out of bounds of the image (towards the bottom)
            # iteration complete
            if (coord.y - framesize) < 0:
                center = False
                
            # else if the center coordinate is only out of bounds on the right,
            # slide frame back to the left and bottom by one framesize
            else:
                center = Point(framesize/2, coord.y-framesize)
                # print(2)
        # else if not out of bounds, only move frame to the right
        else:
            center = Point((coord.x+framesize), coord.y)
            # print(3)
            # print(center)
        return center
    
    def within_region(point):
        '''
        objective: check whether the coordinate is within any of the regions.
        '''

        coord = point
        
        while(True):
            # print()
            # print("original coodrinate:", coord)
            
            for r in range(num_regions):
                if regions[r].contains(coord) is True:
                    labeled.append([Point((coord.x-framesize/2), (coord.y+framesize/2)), labels[r]])
                    break
                else:
                    unlabeled.append(coord)
            coord = reposition(coord)
            if reposition(coord) is False:
                break

    labeled = []
    unlabeled = []

    # get number of unique region IDS
    num_regions = len(regions)
    # print(num_regions)

    center = get_center_frame(height,width,framesize)
    within_region(center)

    # total number of frames
    total_frames = (height/framesize) * (width/framesize)

    print(labeled)
    
    print("there are {} frames total".format(total_frames))
    print("there are {} that are within a labeled region".format(len(labeled)))
    print("there are {} that are not within a labeled region". format(len(set(unlabeled))))



def classify_center(imgfile,labelfile,framesize = -1):
    
    # output is
    # IDS = list of ints
    # LABELS = list of strings
    # COORDS = list of list of ints
    # HEADER of file
    HEADER,IDS,LABELS,COORDS = annotations.parse_annotations(labelfile)
    NIL = phg.Nil()
    NIL.open(imgfile)

    # get height and width of image (in pixels) from the header
    height = int(HEADER['height'])
    width = int(HEADER['width'])
    
    # Get dimensions
    xdim,ydim =NIL.get_dimension()
    
    if framesize == -1:
        # frame = [xdim,ydim]
        pass
    else:
        shapes = []

        # generates polygon of regions within the image
        for i in range(len(COORDS)):
            shapes.append(pointwithin.generate_polygon(COORDS[i]))

        # classify the frames based on if it is within any region (shape)
        classification(LABELS, height, width, framesize, shapes)

    

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
classify_center(imagefile, labelfile, 1000)
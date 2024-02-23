import nedc_image_tools as phg
import sys
import annotations
import pointwithin
import matplotlib.pyplot as plt
from svstojpg import svs_to_jpg as stj
import creategraphic
import shapely

sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")
labelfile = "/data/isip/data/tuh_dpath_breast/deidentified/v2.0.0/svs/train/00707578/s000_2015_04_01/breast/00707578_s000_0hne_0000_a004_lvl000_t000.csv"
imagefile = "/data/isip/data/tuh_dpath_breast/deidentified/v2.0.0/svs/train/00707578/s000_2015_04_01/breast/00707578_s000_0hne_0000_a004_lvl000_t000.svs"



def get_center_frame(height, width, framesize):
    '''get the center coodinates of the top-left-most frame'''

    center_x = height + framesize/2
    center_y = width - framesize/2
    center = [center_x, center_y]

    return center

def classification(IDS, height, width, framesize, shapes):
    '''
    objective: classify whether the center fo the first frame (top-left-most) is within a labeled region
        if within labeled region -> set coordinate to 'labeled' status
        if not within labeled region -> set coordinate to 'unlabeled' status
        status is set by storing these coordinates into
    '''

    labeled = []
    unlabeled = []

    # get number of unique region IDS
    num_regions = len(set(IDS))

    # create empty list for coordinates of each region
    region = [[] for _ in range(len(num_regions))]
    
    # get the coordinate at the center of the top-left-most frame
    center = get_center_frame(height,width)

    # initialize region id to the first ID in the file
    region_id = IDS[0]

    # separate all the regions
    for i in range(len(shapes)):
        # if this is the same region, add coordinate to the list
        if region_id == IDS[i]:
            region[region_id].append(shapes[i])
        # else if this is a different region, add it to the next list
        else:
            region_id = IDS[i]
            region[region_id].append(shapes[i])

    # first center coordinate
    within_region(center)

    # the rest of the center coordinates
    for k in range(len(shapes) - 1):
        within_region(reposition())

    def reposition():
        # if the next center coordinate is out of bounds of the image (towards the right)
        if (center[0] + framesize) > width:
            # if the next center coordinate is out of bounds of the image (towards the bottom)
            # iteration complete
            if (center[1] - framesize) < 0:
                pass
            # else if the center coordinate is only out of bounds on the right,
            # slide frame back to the left and bottom by one framesize
            else:
                center = [0,center[1]-framesize]
        # else if not out of bounds, only move frame to the right
        else:
            center = [center[0]+framesize,center[1]]
        return tuple(center)
    
    def within_region(coord):
        '''
        objective: check whether the coordinate is within any of the regions.
            1. split the list of coordinates by the region.
            2. check if the coordinate falls within region.
        '''

        for r in range(len(num_regions)):
            polygon = shapely.Polygon(region[r])
            for i in range(len(shapes)):
                if polygon.contains(coord) is True:
                    labeled.append()
                else:
                    unlabeled.append()
    
    print("labeled coordinates:", labeled)
    print("unlabeled coordinates:", unlabeled)



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
    
    #print("Dimensions = ",xdim,ydim)
    

    if framesize == -1:
        # frame = [xdim,ydim]
        pass
    else:
        frame = [framesize,framesize]

    
    classification(IDS, height, width, framesize, creategraphic.main())

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
    
classify_center(imagefile, labelfile)

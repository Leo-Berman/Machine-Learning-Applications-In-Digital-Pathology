import nedc_image_tools as phg
import sys
import annotations

sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")
labelfile = "/data/isip/data/tuh_dpath_breast/deidentified/v3.0.0/svs/train/00466205_aaaaaage/s000_2015_03_01/breast/00466205_aaaaaage_s000_0hne_0000_a001_lvl002_t000.csv"
#labelfile = "/home/tul16619/SD1/Machine-Learning-Applications-In-Digital-Pathology/nedc_pyprint_image/random_test.csv"
imgfile = "/data/isip/data/tuh_dpath_breast/deidentified/v3.0.0/svs/train/00466205_aaaaaage/s000_2015_03_01/breast/00466205_aaaaaage_s000_0hne_0000_a001_lvl002_t000.svs"

def classify_center(imgfile,labelfile,framesize = -1):
    
    # output is
    # IDS = list of ints
    # LABELS = list of strings
    # COORDS = list of list of ints
    IDS,LABELS,COORDS = annotations.parse_annotations(labelfile)
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

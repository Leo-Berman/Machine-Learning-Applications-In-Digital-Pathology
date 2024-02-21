from PIL import Image
import nedc_image_tools as phg
import sys
sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")

def parse_labels(file):
    '''
    Get the label information and use that to show where borders are
    return label info this could be a zipped list of shapes and their 
    coordinates tbd up 2 u
    '''

def classify_center(imgfile,labelfile,framesize = -1):

    labels = parse_labels(labelfile)
    NIL = phg.Nil()
    NIL.open(imgfile)
    
    # Get dimensions
    xdim,ydim =NIL.get_dimension()
    
    #print("Dimensions = ",xdim,ydim)
    

    if framesize == -1:
        frame = [xdim,ydim]
    else:
        frame = [framesize,framesize]


    # Get all the coordinates for each windows
    coordinates = [(x, y) for x in range(0, xdim, frame[0]) for y in range(0, ydim, frame[1])]
    
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
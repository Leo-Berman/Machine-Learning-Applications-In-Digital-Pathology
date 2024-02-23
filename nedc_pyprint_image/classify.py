import nedc_image_tools as phg
import sys
import annotations
import pointwithin
import matplotlib.pyplot as plt
from svstojpg import svs_to_jpg as stj
import creategraphic

sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")
labelfile = "/data/isip/data/tuh_dpath_breast/deidentified/v3.0.0/svs/train/00466205_aaaaaage/s000_2015_03_01/breast/00466205_aaaaaage_s000_0hne_0000_a001_lvl002_t000.csv"
#labelfile = "/home/tul16619/SD1/Machine-Learning-Applications-In-Digital-Pathology/nedc_pyprint_image/random_test.csv"
imgfile = "/data/isip/data/tuh_dpath_breast/deidentified/v3.0.0/svs/train/00466205_aaaaaage/s000_2015_03_01/breast/00466205_aaaaaage_s000_0hne_0000_a001_lvl002_t000.svs"

#imagefile = "/data/isip/data/tuh_dpath_breast/deidentified/v2.0.0/svs/train/00707578/s000_2015_04_01/breast/00707578_s000_0hne_0000_a004_lvl000_t000.svs"


def get_center_frame(height, width, framesize, coordinates):
    '''get the center coodinates of the frame'''

    y = coordinates[0]  # coordinates from the ROW header
    x = coordinates[1]  # cooridnates from the COLUMN header

    # start from the top-left of the image
    center_x = height + framesize/2
    center_y = width - framesize/2
    center = [center_x, center_y]

def reposition(height, width, framesize):
    '''
    slide the frame (starting from the top-left) to the right until it reaches the end of the image.
    slide the image down and start from the left and repeat.
    '''
    y = height - framesize
    x = width + framesize

    # if the x-coordinate (on the right-hand side of frame) is out of bounds of the image (greater than the width),
    # reset it back to left-hand side and slide frame down (by framesize)
    if (x+framesize > width):
        # if the y-coordinate is out of bounds of the image (less than zero), finish iteration
        # otherwise, reposition
        if (y-framesize < 0) is False:
            x = 0
            y = y - framesize
    # if x-coordinate is not out of bounds, slide it over to the right (by framesize)
    else:
        x = x + framesize

def draw_square(height,width,framesize):

    x_left = 0
    x_right = x_left + framesize
    y_top = height
    y_bottom = height - framesize
    frame = [(x_left,y_top), (x_right,y_top), (x_left,y_bottom), (x_right,y_bottom)]
    print(pointwithin.generate_polygon(frame))

    plt.xlim(0,width)
    plt.ylim(0,height)
    stj(imgfile,"./DATA/graphic")
    im = plt.imread("./DATA/graphic.jpg")
    plt.imshow(im,extent=[0,width,0,height])
    plt.savefig("./DATA/drawframe.png")

            

def classify_center(imgfile,labelfile,framesize = -1):
    
    # output is
    # IDS = list of ints
    # LABELS = list of strings
    # COORDS = list of list of ints
    # HEADER of file
    HEADER,IDS,LABELS,COORDS = annotations.parse_annotations(labelfile)
    NIL = phg.Nil()
    NIL.open(imgfile)

    # get height and width from the header
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

    draw_square(height,width,framesize)


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

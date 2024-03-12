import nedc_image_tools as phg
import sys
import annotations
import pointwithin
from shapely.geometry import Point
from printframevalues import window_to_rgb
from rgbatodct import rgba_to_dct

sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")
labelfile = "/data/isip/data/tuh_dpath_breast/deidentified/v2.0.0/svs/train/00707578/s000_2015_04_01/breast/00707578_s000_0hne_0000_a004_lvl000_t000.csv"
imagefile = "/data/isip/data/tuh_dpath_breast/deidentified/v2.0.0/svs/train/00707578/s000_2015_04_01/breast/00707578_s000_0hne_0000_a004_lvl000_t000.svs"



def get_center_frame(height, width, framesize):
    '''
    objective:
        get the center coodinate of the top-left-most frame
    return:
        center coordinate of top-left-most frame in shapely-point format.
    '''

    center_x = 0 + framesize/2
    center_y = height - framesize/2
    center = Point(center_x, center_y)

    return center

def classification(labels, height, width, windowsize, framesize, regions):
    '''
    objective: classify whether the center of each frame is within a labeled or unlabeled region.
        - if within labeled region -> set coordinate to 'labeled' status.
        - if not within labeled region -> set coordinate to 'unlabeled' status.
        - status is set by storing these coordinates into.
    
    return:
        - list of label-classified windows as x and y coordinates with corresponding label.
        - format: list of lists -> [[x-coord, y-coord, label], [...],...].
            - x-coordinate and y-coordinate is the coordinate of the top-left corner of the window (not frame).
            - 'label' is the label that the center-coordinate of the frame falls within, such as NORM, BKG, SUSP, etc.

    functions:
        within_region:
            the first center-coordinate (the center of the top-left frame) gets passed in the function.
            the function checks if the coordinate is in any of the labeled regions.
            if it is, the top-left coordinate of the corresponding WINDOW and the label gets appended to the 'labeled' list.
            otherwise, the coordinate is appended to the 'unlabeled' list.
            the current center-coordinate gets passed to the 'reposition' function.
        repostion:
            a coordinate gets passed in the function.
            the coordinate is repositioned accordingly:
                - if the next coordinate is out-of-bounds ONLY from the right side of the image,
                    move to the next row of frames and start from the left again.
                - if the next coordinate is out-of-bounds from the right side AND bottom side of the image,
                    all frames were iterated through.
                - otherwise, move the coordinate to the right (by framesize).
                  ex) if frame size is 100, move the coordinate to the right by 100.
    '''

    def reposition(coord):
        '''
        objective:
            - repostion the coordinate according to the conditions met.
        '''

        # if the next center coordinate is out of bounds of the image (towards the right).
        if (coord.x + framesize) > width:
            # if the next center coordinate is out of bounds of the image (towards the bottom).
            # iteration complete. return false.
            if (coord.y - framesize) < 0:
                center = False
            # else if the center coordinate is only out of bounds on the right,
            # slide frame back to the left and bottom by one framesize.
            else:
                center = Point(framesize/2, coord.y-framesize)
        # else if not out of bounds, only move frame to the right.
        else:
            center = Point((coord.x+framesize), coord.y)
        return center
    
    def within_region(coord):
        '''
        objective:
            - check whether the coordinate is within any of the regions.
            - coordinates organized into 'labeled' and 'unlabeled' lists,
                - along with the coordinates' corresponding label

        format:
            - 'labeled' (list) = [[coordinate, label], ...]
        '''
        
        while(True):
            # check if the coordinate is within region[r].
            for r in range(num_regions):
                # if it within one of the regions, add to the labeled list with its corresponding label and move to the next coordinate.
                if regions[r].contains(coord) is True:
                    labeled.append([((coord.x-windowsize/2), (coord.y+windowsize/2)), labels[r]])
                    break
                # if not in any regions, classify as unlabeled.
                else:
                    unlabeled.append(coord)
            # reposition to the next frame.
            coord = reposition(coord)
            # if the new coordinate is false, terminate the loop.
            if reposition(coord) is False:
                break

    # initialize 'labled' and 'unlabeled' list
    labeled = []
    unlabeled = []

    # get total number of unique region IDS
    num_regions = len(regions)

    # get the center coordinate of the top-left frame.
    center = get_center_frame(height,width,framesize)

    # start classification of all center coordinates.
    within_region(center)

    # total number of frames
    total_frames = (height/framesize) * (width/framesize)

    # TEST PRINTS
    # print(labeled)
    # print("there are {} frames total".format(total_frames))
    # print("there are {} that are within a labeled region".format(len(labeled)))
    # print("there are {} that are not within a labeled region". format(len(set(unlabeled))))

    return labeled



def classify_center(imgfile,labelfile,windowsize,framesize = -1):
    '''
        MAIN FUNCTON
        objective:
            get the headers, region ids, labels, and coordinates from annotations

        return:
            list of list coordinates and labels, framesize
    '''

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
    
    # if framesize is not given, don't do anything for now.
    if framesize == -1:
        print("default frame size: {} x {}".format(height,width))
    else:
        # generate polygon of regions within the image
        shapes = []
        for i in range(len(COORDS)):
            shapes.append(pointwithin.generate_polygon(COORDS[i]))

        # classify the frames based on if it is within any region (shape)
        labeled_list = classification(LABELS, height, width, windowsize, framesize, shapes)
        # print(labeled_list)

        outlabels = []
        outcoords = []
        # pass the image file, all the top-left labeled coordinates, and framesize to the window_to_rgb file
        for x in range(len(labeled_list)):
            outlabels.append(labeled_list[x][1])
            outcoords.append((int(labeled_list[x][0][0]),int(height - labeled_list[x][0][1]+framesize)))
            # window_to_rgb(imagefile,label = labeled_list[x][1],coord = labeled_list[x][0],window_frame = [framesize,framesize],name = "file")
        # print(outcoords)
        window_list = window_to_rgb(imagefile,labels = outlabels,coords = outcoords,window_frame = [framesize,framesize],name = "file")
        for x in window_list:
            rgba_to_dct(x)

        # return(labeled_list, framesize)

    '''  
    if processed:
        return True
    else:
        return False
    '''
classify_center(imagefile, labelfile, 5000, 1000)

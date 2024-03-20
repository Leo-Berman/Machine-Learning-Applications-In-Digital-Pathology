"""
The maketraining.py module will classify each frame as labeled or unlabeled whether it falls within a labeled or unlabeled region, respectively.
"""

import nedc_image_tools as phg
import sys
import shapely

import nedc_pyprint_image_lib

sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")

def classify_center(imagefile,labelfile,windowsize,framesize = -1):
    """
        Based on the parameters inputted, this function will do pre-measures before calling the 'classification' function.
        1. Calls a function that parses the annotations of the given label file, and returns the headers, labels, and coordinates (x,y).
        2. Creates a list of shapes based on the coordinates.
        It extracts the height and width of the image from the header.

        After the pre-measures, this function will call the 'classification' function and receive a list of lists containing a label and coordinate of the top-left corner of each window.
        The 'classification' function will output a list of list of a coordinate and label.

        The 'labeled' coordinates get fed to the 'svstorgb' module to get the RGBA values.
        The RGBA are put into a DCT

        :param imagefile: Image file of the tissue in the form of a svs file.
        :type imagefile: string (ends with .svs)

        :param labelfile: Label file of tissue in the form of a CSV file. Consists of headers and values of the headers, such as region indices, x- and y- coordinates, and labels.
        :type param2: string (ends with .csv)

        :param windowsize: Length and width of the window. Must be 1.5x bigger than framesize.
        :type windowsize: int

        :param framesize: Length and width of the frame that is used to iterate over the whole image.
        :type framesize: int
    """

    # IDS = list of ints
    # LABELS = list of strings
    # COORDS = list of list of ints
    # HEADER of file
    HEADER,LABELS,COORDS = nedc_pyprint_image_lib.nedc_labels_lib.parseannotations.parse_annotations(labelfile)
    NIL = phg.Nil()
    NIL.open(imagefile)

    # get height and width of image (in pixels) from the header
    height = int(HEADER['height'])
    width = int(HEADER['width'])
    
    # if framesize is not given, don't do anything for now.
    if framesize == -1:
        print("Default frame size: {} x {}".format(height,width))
    else:
        print("Processing files and classifiying each {} x {} window...".format(windowsize,windowsize))
        # generate polygon of regions within the image
        shapes = []
        for i in range(len(COORDS)):
            shapes.append(nedc_pyprint_image_lib.nedc_labels_lib.geometry.generate_polygon(COORDS[i]))

        # classify the frames based on if it is within any region (shape)
        labeled_list = classification(LABELS, height, width, windowsize, framesize, shapes)

        outlabels = []
        outcoords = []
        # pass the image file, all the top-left labeled coordinates, and framesize to the window_to_rgb file
        for x in range(len(labeled_list)):
            outlabels.append(labeled_list[x][1])
            outcoords.append((int(labeled_list[x][0][0]),int(height - labeled_list[x][0][1]+framesize)))

        print("Classification completed. Sending data to window_to_rgb module...")

        window_list = nedc_pyprint_image_lib.nedc_labels_lib.svstorgb.window_to_rgb(imagefile,labels = outlabels,coords = outcoords,window_frame = [framesize,framesize],name = "file")
        for x in window_list:
            nedc_pyprint_image_lib.nedc_labels_lib.svstorgb.rgba_to_dct(x)

def classification(labels, height, width, windowsize, framesize, regions):
    """
        This function's objective is to classify whether the center of each frame is within a labeled or unlabeled region.
        1) Calls the 'get_top_left' function to get the center coordinate of the top-left-most frame.
        2) Calls the 'within_region' function to check if the coordinate is in a region.
            - The coordinate will go through a loop between 'within_region' and 'repostion' as the coordinate gets repositioned until all the frames are iterated through.

        :param labels: List of labels of each coordinate pair.
        :type labels: list of strings

        :param height: Height of image.
        :type height: int

        :param width: Width of image.
        :type width: int

        :param windowsize: Length and width of the window. Must be 1.5x bigger than framesize.
        :type windowsize: int

        :param framesize: Length or width of the frame
        :type framesize: int

        :param regions: List (of each region (or shape)) of lists (of coordinates of the specific region)
        :type regions: list of list of tuples

        :return: Returns a list of lists of a 'labeled' coordinate and its label.
        :rtype: list of lists: [[coordinate(tuple), label(string)], [coordinate(tuple), label(string)], ...] 
    """

    def get_top_left(height, framesize):
        """
            This subfunction gets the center coodinate of the top-left-most frame.

            :param height: Height of the image.
            :type height: int

            :param framesize: Length or width of the frame
            :type framesize: int

            :return: Returns the center coordinate of the top-left-most.
            :rtype: tuple of floats: (x,y)
        """

        center_x = 0 + framesize/2
        center_y = height - framesize/2
        center = shapely.geometry.Point(center_x, center_y)

        return center

    def reposition(coord):
        """
            This function repositions the given coordinate accordingly:
                - if the next coordinate is out-of-bounds ONLY from the right side of the image,
                    move to the next row of frames and start from the left again.
                - if the next coordinate is out-of-bounds from the right side AND bottom side of the image,
                    all frames were iterated through.
                - otherwise, move the coordinate to the right (by framesize).
                  ex) if frame size is 100, move the coordinate to the right by 100.

            :param coord: x- and y- coordinate of the center of the current frame.
            :type coord: tuple of floats: (x,y)

            :return: Returns the repostioned coordinate.
            :rtype: tuple of floats: (x,y)
        """

        # if the next center coordinate is out of bounds of the image (towards the right).
        if (coord.x + framesize) > width:
            # if the next center coordinate is out of bounds of the image (towards the bottom).
            # iteration complete. return false.
            if (coord.y - framesize) < 0:
                center = False
            # else if the center coordinate is only out of bounds on the right,
            # slide frame back to the left and bottom by one framesize.
            else:
                center = shapely.geometry.Point(framesize/2, coord.y-framesize)
        # else if not out of bounds, only move frame to the right.
        else:
            center = shapely.geometry.Point((coord.x+framesize), coord.y)
        return center
    
    def within_region(coord):
        """
            This function checks whether the coordinate is within any of the regions.
            The top-left coordinate of each WINDOW gets organized into:
                'labeled' list if the center coordinate is inside a region along with the coordinates' corresponding label as pair.
                'unlabeled' list if the center coordinate is not inside a region.

            :param coord: x- and y- coordinate of the center of the current frame.
            :type coord: tuple of floats: (x,y)
        """
        
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
    center = get_top_left(height,framesize)

    # start classification of all center coordinates.
    within_region(center)

    # total number of frames
    total_frames = (height/framesize) * (width/framesize)

    return labeled





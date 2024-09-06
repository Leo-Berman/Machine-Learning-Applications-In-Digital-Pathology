"""
The maketraining.py module will classify each frame as labeled or unlabeled whether it falls within a labeled or unlabeled region, respectively.
"""
import nedc_image_tools as phg
import shapely
import nedc_mladp_geometry_tools as geometry_tools


def labeled_regions(coordinates:list):
    """
        Takes a list of list of coordinates and generates a list of correlating shapes
    """

    # generate polygon of regions within the image
    #
    ret_shapes = []
    for i in range(len(coordinates)):
        ret_shapes.append(geometry_tools.generate_polygon(coordinates[i]))

    return ret_shapes

def labeled_frames(labels:list,height:int,width:int,windowsize:int,framesize:int,shapes:list):
    
    # classify the frames based on if it is within any region (shape)
    #
    labeled_list = classify_center(labels, height, width, windowsize, framesize, shapes)

    # create a list for return labels and their top left coordinate
    #
    ret_labels = []
    ret_coords = []

    # iterate through labeled list
    #
    for x in range(len(labeled_list)):

        # create list of labels
        #
        ret_labels.append(labeled_list[x][1])

        # create list of tuples of coordinates compensating for column row format vs traditional
        # x,y format
        #
        ret_coords.append((int(labeled_list[x][0][0]),int(height - labeled_list[x][0][1]+framesize)))

    # return the top left coordinates of each frame and their corresponding labels
    #
    return ret_coords,ret_labels

def frame_rgba_values(image_file:str,labels:list,coords:list, windowsize:int):
    
    # open the imagefile
    # 
    NIL = phg.Nil()
    NIL.open(image_file)

    # read all of the windows into memory
    #
    window = NIL.read_data_multithread(coords,npixx = windowsize,npixy = windowsize,color_mode="RGBA")
    
    # create a list of rgba values for the frame
    #
    window_list = []
    for i in range(len(window)):
        workwindow = [labels[i]]
        for j in window[i]:
            for k in j:
                workwindow.extend(k.tolist())
        window_list.append(workwindow)

    # return list of lists of rgba values
    #
    return window_list



def classify_center(labels, height, width, windowsize, framesize, regions):
    """
        objective:
            This function's objective is to classify whether the center of each frame is within a labeled or unlabeled region.\n
            1) Calls the 'get_top_left' function to get the center coordinate of the top-left-most frame.\n
            2) Calls the 'within_region' function to check if the coordinate is in a region.\n
            3) The coordinate will go through a loop between 'within_region' and 'repostion' as the coordinate gets repositioned until all the frames are iterated through.\n

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

# consider frames labelled if more than 50% are in a labelled area
# this may not be working
#
def classify_frame(imagefile, framesize, labels, shapes:list):
    
    # create lists to hod labelled frames and their coordinates
    #
    labeled = []
    labeled_frame_coords = []

    # create the frames
    #
    frame_coords = geometry_tools.getframestart(imagefile,framesize)
    frames = geometry_tools.createboxshapes(frame_coords,framesize)

    # iterate through the frames and if there is more than 50% overlap
    # append the top left coordinate and label to the appropriate list
    #
    for x in frames:
        for i,y in enumerate(shapes):
            overlap = shapely.intersection(x,y)
            if overlap.area/x.area > .5:
                border = geometry_tools.get_border(x)
                labeled_frame_coords.append((int(border[0][0]),int(border[0][1])))
                labeled.append(labels[i])

    # return labeled frame's top left coordinate and a corresponding label list
    #
    return labeled_frame_coords,labeled

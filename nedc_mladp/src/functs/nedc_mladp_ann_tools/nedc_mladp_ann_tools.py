"""
The maketraining.py module will classify each frame as labeled or unlabeled whether it falls within a labeled or unlabeled region, respectively.
"""
import nedc_image_tools as phg
import shapely
import nedc_mladp_geometry_tools as geometry_tools
import numpy as np

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
    
    '''window_list = []
    for i in range(len(window)):
        workwindow = [labels[i]]
        for j in window[i]:
            for k in j.tolist():
                workwindow.extend(k)
        window_list.append(workwindow)'''
        
        
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
def coords_to_bits(coords:list[tuple], frame_dx:int, frame_dy:int) -> np.array:
    '''Converts coordinates to a bit matrix. 
    
    :param coords: Coordinates for one label.
        Example: coords = [(200,200),(0,400),(400,400),(600,200)]
    :param frame_dx: Frame width.
    :param frame_dy: Frame height.
    :return: Bit matrix, each bit representing a frame. 
    :return type: np.array[int]
    '''
    # Convert coords to numpy array for slicing.
    #
    coords = np.array(coords)
    '''e.g., np.array(coords)
    array([ [200, 200],
            [  0, 400],
            [400, 400],
            [200, 600]  ])'''
    # Find the highest x and y dimensions.
    #
    max_x = coords[:,0].max()
    max_y = coords[:,1].max()
    '''e.g., max_x = 400, max_y = 600'''
    # Divide by frame_dx and frame_dy to get matrix size.
    #
    rows = int(max_y/frame_dy) + 1
    cols = int(max_x/frame_dx) + 1
    '''e.g., rows = 4, cols = 3'''
    
    # Create a matrix.
    #
    m = np.zeros((rows,cols), dtype=int)
    '''e.g., m = array([[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]])'''
    # Convert coords to matrix indices.
    #   Coords should always be on frame boundaries or problems will result.
    #
    coords[:,0] = coords[:,0] / frame_dx
    coords[:,1] = coords[:,1] / frame_dy
    '''e.g., coords = array([   [1, 1],
                                [0, 2],
                                [2, 2],
                                [1, 3]  ])'''
    # Populate matrix.
    #
    for [col,row] in coords:
        m[row,col] = 1
    '''e.g., m = array([[0, 0, 0],
                        [0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])'''
    
    return m
def _in_bounds(point:tuple, bottom_right_bnd:tuple, top_left_bnd:tuple = (0,0)) -> bool:
    '''Determines if a point lies on the matrix (bounds-inclusive).
    
    :param point: Index of the point to check, of type tuple[int].
        Example: (1,1)
    :param bottom_right_bnd: Bottom-right boundary of matrix.
        Example: (40,40) for a 41x41 matrix.
    :param top_left_bnd: Top-left boundary, typically (0,0).
    
    :return: Boolean, true if the point lies on the matrix, false if not.
    '''
    # Compare columns.
    #
    if top_left_bnd[0] <= point[0] <= bottom_right_bnd[0]:
        # Compare rows.
        #
        if top_left_bnd[1] <= point[1] <= bottom_right_bnd[1]:
            return True
    return False
def _flood_fill(matrix:np.ndarray, start_point:tuple) -> None:
    '''4-way-fills a bit matrix with oness starting at [0,0] (top-left), 
        stops at matrix boundaries or regions with ones.
        
    :param matrix: Matrix to fill in (passed as a reference, so there is no return value).
    :param start_point: Where to start filling in.'''
    
    # Get the bottom-rightmost point index. 
    #
    bottom_right_bnd = np.subtract(matrix.shape,(1,1))
    
    # Start the flood-fill at (0,0), appending nearby points if their value is zero.
    # 
    indices = [start_point]
    while indices:
        
        # Get the first item in indices.
        #
        index = indices[0]
        # Set the value at that index to 1.
        #
        matrix[index] = 1
        # Get indices for nearby points
        #
        up = tuple(np.add(index,(0,-1)))
        down = tuple(np.add(index,(0,1)))
        left = tuple(np.add(index,(-1,0)))
        right = tuple(np.add(index,(1,0)))
        # Check to see if those points are on the matrix.
        #   If they are, add them to indices.
        # 
        if _in_bounds(up,bottom_right_bnd) and matrix[up] != 1:
            indices.append(up)
        if _in_bounds(down,bottom_right_bnd) and matrix[down] != 1:
            indices.append(down)
        if _in_bounds(left,bottom_right_bnd) and matrix[left] != 1:
            indices.append(left)
        if _in_bounds(right,bottom_right_bnd) and matrix[right] != 1:
            indices.append(right)
        # Continue iterating until all points in indices are exhausted.
        #
        indices.pop(0)
def pad_and_fill(matrix:np.ndarray) -> np.array:
    '''Creates a border around a bit matrix and flood-fills with ones from the border inward.
    
    :param matrix: Input bit matrix.
    :return: Matrix with all interior regions filled in.'''
    # Create a copy of the matrix.
    #
    mask = matrix.copy()
    # Pad the copied matrix edges (top, bottom, sides) with zeroes
    #
    mask = np.pad(mask, 1, 'constant', constant_values=0)
    
    # Flood fill the copy starting at (0,0).
    #
    _flood_fill(mask, (0,0))
    
    # Invert the values of the copied matrix.
    #
    mask = 1-mask
    # Remove padding.
    #
    mask = mask[1:-1,1:-1]
    # Superimpose the original and copied matrices.
    #
    return matrix + mask

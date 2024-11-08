# import python libraries
#
import shapely
import sys
import matplotlib.pyplot

# import NEDC libraries
#
import nedc_image_tools

'''
generate a shapely.Polygon from list of coordinates

Parameters:
    coordinates % list of tuples of coordinates
Return:
    shape % resulting shapely.Polygon
'''
def generatePolygon(pp:list)->shapely.Polygon:
    # compute centroid
    cent=(sum([p[0] for p in pp])/len(pp),sum([p[1] for p in pp])/len(pp))
    # sort by polar angle
    pp.sort(key=lambda p: math.atan2(p[1]-cent[1],p[0]-cent[0]))
    shape = shapely.Polygon(pp)
    return shape

'''
return the border of a shape

Parameters:
   shape % 2 dimensional shapely.Polygon
Return:
   border % tuple of arrays, index 0 is x coordinates,
            and index 1 is y coordinates
'''
def getBorder(shape:shapely.Polygon)->tuple:
    border = shape.exterior.xy
    return border
'''
    return the top left corner of every frame
Parameters:
    imagefile % path to svs file
    framesize % size of frame that will be used to segment image
Return:
    frame_top_left_coordiantes % list of tuples (x,y) containing
                                 the top left corner of each frame
'''
def getFrameCoordinates(imagefile:str,framesize:int)->list:
    
    # open the imagefile
    #
    image_reader = nedc_image_tools.Nil()
    image_reader.open(imagefile)
    
    # Get dimensions
    #
    image_width,image_height= image_reader.get_dimension()

    # Get all the coordinates for each windows
    #
    frame_top_left_coordinates = [(x, image_height-y+framesize) for x in range(0, image_width, framesize) for y in range(0, image_height+framesize, framesize)]

    # return that list of coordinates
    #
    return frame_top_left_coordinates


'''
turn all of the top left coordinates into a list of shapely.Polygons 

Parameters:
    coordinates % list of top left corners
    frame_size % size of the boxes length and width
Return:
    frames %  list of shapely.Polygon each containing a frame
'''
def createFrames(coordinates:list,frame_size:int)->list:

    # create a list for the shapely polygons
    #
    frames = []

    # iterate through all the coordinates
    #
    for x in coordinates:

        # get the four corners of the square
        #
        top_left = [x[0],x[1]]
        top_right = [x[0]+frame_size,x[1]]
        bottom_right = [x[0]+frame_size,x[1]-frame_size]
        bottom_left = [x[0],x[1]-frame_size]
        shape_coordinates = [top_left,top_right,bottom_right,bottom_left]

        # append the polygon square to the list
        #
        frames.append(generatePolygon(shape_coordinates))

    # return the list of shapley polygons
    #
    return frames


'''
Example driver function to show how this code might
be used
'''
def main():
    # declare an image file to use
    #
    image = "/data/isip/data/tuh_dpath_breast/deidentified/v3.0.0/svs/train/00477780_aaaaaagg/s000_2017/breast/00477780_aaaaaagg_s000_0hne_0000_a005_lvl001_t000.svs"

    # declare a frame size
    #
    frame_size = 5000

    # obtain the top left coordinates of each frame
    #
    frame_coordinates = getFrameCoordinates(image,frame_size)

    # generate the frames
    #
    frames = createFrames(frame_coordinates,frame_size)

    # for each of the frames print the corresponding
    # x and y coordinates
    #
    for i,frame in enumerate(frames):
        print("Frame #" + str(i) + ":",
              "\nX coordinates = ", getBorder(frame)[0],
              "\nY coordinates = ", getBorder(frame)[1])

if __name__ == "__main__":
    main()


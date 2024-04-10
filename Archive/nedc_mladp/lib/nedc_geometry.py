# import python libraries
#
import shapely
import sys

# Picone's libraries
#
sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")
import nedc_image_tools as phg

# generate a shape from border coordinates
#
def generate_polygon(coords):
    shape = shapely.Polygon(coords)
    return shape

# return the border of a shape
#
def get_border(shape):
    return (shape.exterior.xy)

# return the top left corner of every frame
#
def getframestart(imagefile:str,frame:int):
    
    # open the imagefile
    #
    NIL = phg.Nil()
    NIL.open(imagefile)
    
    # Get dimensions
    #
    xdim,ydim =NIL.get_dimension()

    # Get all the coordinates for each windows
    #
    coordinates = [(x, ydim-y+frame) for x in range(0, xdim, frame) for y in range(0, ydim+frame, frame)]

    # return that list of coordinates
    #
    return coordinates



# generate a list of shapes that are each uniform boxes
# coords = list of list (return of get frame start)
# frame = [framewidth,frameheight] or (framewidth,frameheight) IE. can be list or tuple
# if frame is different from what you gave getframestart, your boxes will overlay each other
#
'''
Ex:
filepath = "path to svs file that is 100 pixesl by 100 pixels"
framesize = [10,10]
coords = getframestart(filepath,framesize)
squares = createboxshapes(coords,framesize)

square will be a list of polygons from the shapely library
'''
def createboxshapes(coords,frame:int):

    # create a list for the shapely polygons
    #
    boxes = []

    # iterate through all the coordinates
    #
    for x in coords:

        # get the four corners of the square
        #
        topleft = [x[0],x[1]]
        topright = [x[0]+frame,x[1]]
        botright = [x[0]+frame,x[1]-frame]
        botleft = [x[0],x[1]-frame]
        shapecoords = [topleft,topright,botright,botleft]

        # append the polygon square to the list
        #
        boxes.append(generate_polygon(shapecoords))

    # return the list of shapley polygons
    #
    return boxes
# import python libraries
#
import shapely
import sys

# Picone's libraries
#
import nedc_image_tools

# generate a shape from border coordinates
#
def generatePolygon(coords:list)->shapely.Polygon:
    shape = shapely.Polygon(coords)
    return shape

# return the border of a shape
#
def getBorder(shape:shapley.Polygon)->list:
    return (shape.exterior.xy)

# return the top left corner of every frame
#
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
def createBoxes(coordinates:list,frame_size:int)->list:

    # create a list for the shapely polygons
    #
    boxes = []

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
        boxes.append(generate_polygon(shape_coordinates))

    # return the list of shapley polygons
    #
    return boxes

def main():
    annotations = "/data/isip/data/tuh_dpath_breast/deidentified/v3.0.0/svs/train/00477780_aaaaaagg/s000_2017/breast/00477780_aaaaaagg_s000_0hne_0000_b003_lvl001_t000.xml"
    image = "/data/isip/data/tuh_dpath_breast/deidentified/v3.0.0/svs/train/00477780_aaaaaagg/s000_2017/breast/00477780_aaaaaagg_s000_0hne_0000_a005_lvl001_t000.svs"

    print("Hello")
    
    frame_coordinates = getFrameCoordinates(image,1000)
    print(frame_coordinates)

if __name__ == "__main__":
    print("Hello")
    main()

print("Hello")

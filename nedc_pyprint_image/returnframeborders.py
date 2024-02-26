
import nedc_image_tools as phg
import sys
import pointwithin
import matplotlib.pyplot as plt
sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")

# generate a list of lists of the top left corners of each frame
# imagefile = "path to svs file"
# frame = [framewidth,frameheight] or (framewidth,frameheight) IE. can be list or tuple
#
'''
Ex:
filepath = "path to svs file that is 100 pixesl by 100 pixels"
framesize = [10,10]
coords = getframestart(filepath,framesize)

coords will be:

    [[0, 100], [10, 100], ..., [100,100],
     [0, 90], [10, 90], ..., [100,90],
     ...,
     [0, 0], [10, 0], ..., [100,0],
    ]

'''
def getframestart(imagefile,frame=None):
    
    # open the imagefile
    #
    NIL = phg.Nil()
    NIL.open(imagefile)
    
    # Get dimensions
    #
    xdim,ydim =NIL.get_dimension()

    # if frame is None then just make one big frame
    #
    if frame == None:
        frame = [xdim,ydim]

    # Get all the coordinates for each windows
    #
    coordinates = [(x, ydim-y+frame[1]) for x in range(0, xdim, frame[0]) for y in range(0, ydim, frame[1])]
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
def createboxshapes(coords,frame):

    # create a list for the shapely polygons
    #
    boxes = []

    # iterate through all the coordinates
    #
    for x in coords:

        # get the four corners of the square
        #
        topleft = [x[0],x[1]]
        topright = [x[0]+frame[0],x[1]]
        botright = [x[0]+frame[0],x[1]-frame[1]]
        botleft = [x[0],x[1]-frame[1]]
        shapecoords = [topleft,topright,botright,botleft]

        # append the polygon square to the list
        #
        boxes.append(pointwithin.generate_polygon(shapecoords))

    # return the list of shapley polygons
    #
    return boxes

# wrapper function that given the imagefile and frame will
# plot those squares to a matlobplit.pyplot
#
'''
Ex:
filepath = "path to svs file that is 100 pixesl by 100 pixels"
framesize = [10,10]
plt_frames(filepath,framesize)
plt.savefig("name.jpg")

name.jpy will be a file that is a plot of the squares overlayed
on a blank plot
'''
def plt_frames(imagefile,frame=None):
    starts = getframestart(imagefile,frame)
    shapes = createboxshapes(starts,frame)
    for x in shapes:
        points = pointwithin.get_border(x)
        plt.plot(points[0],points[1])



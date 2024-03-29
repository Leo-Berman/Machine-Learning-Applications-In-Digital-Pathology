from PIL import Image
import os
import matplotlib.pyplot as plt

import sys

# picone libraries
sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")
sys.path.append("/data/isip/tools/linux_x64/nfc/class/python/nedc_sys_tools")
import nedc_image_tools as phg
import nedc_file_tools

# our libraries
sys.path.append("../lib")
import nedc_fileio
import nedc_regionid
import nedc_geometry

# convert an svs file to compressed jpeg file
#
def svs_to_jpg(imagefile,output_path,compression):

    # Use Nil library to open up the file
    #
    NIL = phg.Nil()
    NIL.open(imagefile)

    # Get dimensions of the svs image in pixels
    #
    xdim,ydim =NIL.get_dimension()

    # Read the single frame
    #
    windows = NIL.read_data_multithread([[0,0]], xdim, ydim)
    
    # save the images as JPEGS
    # generate the image from RGBA values
    #
    im = Image.fromarray(windows[0])

    # compress the image
    #
    im=im.resize((xdim//compression,ydim//compression))
    
    file,ext = os.path.splitext(output_path)

    # save the image
    #
    output_path = file+'_background'+'.jpg'
    im.save(output_path,"JPEG")
    return output_path

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
        boxes.append(nedc_geometry.generate_polygon(shapecoords))

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
def plt_frames(imagefile,frame):
    starts = getframestart(imagefile,frame)
    shapes = createboxshapes(starts,frame)
    for x in shapes:
        points = nedc_geometry.get_border(x)
        plt.plot(points[0],points[1])
'''
Ex: 
    filepath= *.svs
    name="nameofimage"
    width,height = svs_to_jph(filepath,name)
'''


def main():
    # set argument parsing
    #
    args_usage = "gen_graphics_usage.txt"
    args_help = "gen_graphics_help.txt"
    parameter_file = nedc_fileio.parameters_only_args(args_usage,args_help)

    # parse parameters
    #
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"gen_graphics")
    framesize =  int(parsed_parameters['framesize'])
    output_file = parsed_parameters['output_file']
    image_file = parsed_parameters['imagefile']
    label_file = parsed_parameters['labelfile']
    compression = int(parsed_parameters['compression'])
    show_frames = int(parsed_parameters['showframes'])
    # parse annotations
    #
    header, ids, labels, coordinates = nedc_fileio.parse_annotations(label_file)
        
    # get height and width of image (in pixels) from the header
    #
    height = int(header['height'])
    width = int(header['width'])

    # get labeled regions
    #
    labeled_regions = nedc_regionid.labeled_regions(coordinates)
    for i,z in enumerate(labeled_regions):
        x,y = nedc_geometry.get_border(z)
        plt.plot(x,y)
        plt.text(coordinates[i][0][0],coordinates[i][0][1],labels[i])

    image_loc = svs_to_jpg(image_file,output_file,compression)

    # Set the limits of the plot to be equal to the dimensions of the 
    # svs images
    #
    plt.xlim(0,width)
    plt.ylim(0,height)
    im = plt.imread(image_loc)
    plt.imshow(im,extent=[0,width,0,height])

    if show_frames == 1:
        plt_frames(image_file,framesize)

    plt.savefig(output_file)
if __name__ == "__main__":
    main()
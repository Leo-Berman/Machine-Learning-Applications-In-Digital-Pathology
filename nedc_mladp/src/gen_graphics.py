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
    xdim,ydim = NIL.get_dimension()

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

# plot the projected frames onto an image
#
def plt_frames(imagefile,frame):
    starts = nedc_geometry.getframestart(imagefile,frame)
    shapes = nedc_geometry.createboxshapes(starts,frame)
    for x in shapes:
        points = nedc_geometry.get_border(x)
        plt.plot(points[0],points[1])


def main():

    # set argument parsing
    #
    args_usage = "usagefiles/gen_graphics_usage.txt"
    args_help = "helpfiles/gen_graphics_help.txt"
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

    # plot labeled regions and label them with appropriate text
    #
    for i,z in enumerate(labeled_regions):
        x,y = nedc_geometry.get_border(z)
        plt.plot(x,y)
        plt.text(coordinates[i][0][0],coordinates[i][0][1],labels[i])

    # generate the background image and return the background image's filepath
    #
    image_loc = svs_to_jpg(image_file,output_file,compression)

    # Plot the background image
    #
    plt.xlim(0,width)
    plt.ylim(0,height)
    im = plt.imread(image_loc)
    plt.imshow(im,extent=[0,width,0,height])

    # show the frames of the image
    #
    if show_frames == 1:
        plt_frames(image_file,framesize)

    # save the image
    #
    plt.savefig(output_file)
if __name__ == "__main__":
    main()
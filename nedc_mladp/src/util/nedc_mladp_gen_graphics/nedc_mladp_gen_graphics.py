#!/usr/bin/env python
#
# file: /data/isip/exp/tuh_dpath/exp_0280/nedc_mladp/src/util/nedc_gen_graphics/gen_graphics.py
#
# revision history:
#
# 
#
# This is a Python version of the C++ utility nedc_print_header.
#------------------------------------------------------------------------------
from PIL import Image
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# picone libraries
import nedc_image_tools as phg
import nedc_file_tools

# our libraries
import nedc_mladp_fileio_tools as fileio_tools
import nedc_mladp_ann_tools as ann_tools
import nedc_mladp_geometry_tools as geometry_tools



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
    starts = geometry_tools.getframestart(imagefile,frame)
    shapes = geometry_tools.createboxshapes(starts,frame)
    for x in shapes:
        points = geometry_tools.getBorder(x)
        plt.plot(points[0],points[1])


def main():

    # set argument parsing
    #
    args_usage = "nedc_mladp_gen_graphics.usage"
    args_help = "nedc_mladp_gen_graphics.help"
    parameter_file = fileio_tools.parseArguments(args_usage,args_help)

    # parse parameters
    #
    parsed_parameters = nedc_file_tools.load_parameters(parameter_file,"gen_graphics")
    framesize =  int(parsed_parameters['framesize'])
    output_file = parsed_parameters['output_file']
    image_file = parsed_parameters['imagefile']
    label_file = parsed_parameters['labelfile']
    compression = int(parsed_parameters['compression'])
    show_frames = int(parsed_parameters['showframes'])
    show_decisions=int(parsed_parameters['decisions'])
    decisions_filepath=parsed_parameters['decisions_path']

    # parse annotations
    #
    header, ids, labels, coordinates = fileio_tools.parseAnnotations(label_file)
        
    # get height and width of image (in pixels) from the header
    #
    height = int(header['height'])
    width = int(header['width'])

    # get labeled regions
    #
    labeled_regions = ann_tools.labeledRegions(coordinates)


    '''plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()'''
    
    # plot labeled regions and label them with appropriate text
    #
    for i,z in enumerate(labeled_regions):
        labelsdict = {'artf':'black','nneo':'lightgray','bckg':'lightcoral','norm':'salmon','indc':'orange','nneo':'yellow','null':'green','infl':'aqua','dcis':'deepskyblue','susp':'pink'}
        x,y = geometry_tools.getBorder(z)

        plt.plot(x,y,color = labelsdict[labels[i]])
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

    if show_decisions == 1:

        # get all the label information
        #
        decisions = fileio_tools.read_decisions(decisions_filepath)
        
        # assign the labels to a colors
        labels = {'artf':'black','nneo':'lightgray','bckg':'lightcoral','norm':'salmon','indc':'orange','nneo':'yellow','null':'green','infl':'aqua','dcis':'deepskyblue','susp':'pink'}


        # plot the labelled frames
        currentAxis=plt.gca()
        for i,x in enumerate(decisions):
            if i > 1:
                label = x[0]
                framesize = int(x[1])
                xpos = int(x[2])
                ypos = height-int(x[3])
                currentAxis.add_patch(Rectangle((xpos,ypos),framesize,framesize,facecolor=labels[label]))

    # save the image
    #
    plt.savefig(output_file)
if __name__ == "__main__":
    main()

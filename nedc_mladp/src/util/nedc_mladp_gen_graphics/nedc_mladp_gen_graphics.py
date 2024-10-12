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
import PIL
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# our libraries
import nedc_mladp_fileio_tools as fileio_tools
import nedc_mladp_geometry_tools as geometry_tools
import nedc_mladp_feats_tools as feats_tools

# picone libraries
import nedc_image_tools
import nedc_file_tools




# convert an svs file to compressed jpeg file
#
def svsToJpg(imagefile,output_path,compression):

    # Use Nil library to open up the file
    #
    image_reader = nedc_image_tools.Nil()
    image_reader.open(imagefile)

    # Get dimensions of the svs image in pixels
    #
    width,height = image_reader.get_dimension()

    # Read the single frame
    #
    windows = image_reader.read_data_multithread([[0,0]], width, height)
    
    # save the images as JPEGS
    # generate the image from RGBA values
    #
    image = PIL.Image.fromarray(windows[0])

    # compress the image
    #
    image = image.resize((width//compression,height//compression))
    
    file,extension = os.path.splitext(output_path)

    # save the image
    #
    image.save(output_path,"JPEG")
    return output_path

# plot the projected frames onto an image
#
def plotFrames(imagefile,frame):
    starts = geometry_tools.getFrameStart(imagefile,frame)
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
    frame_x_size =  int(parsed_parameters['frame_x_size'])
    frame_y_size =  int(parsed_parameters['frame_y_size'])
    frame_size = (frame_x_size,frame_y_size)
    
    output_directory = parsed_parameters['output_directory']
    if not (output_directory.endswith("/")):
        output_directory += "/"
    image_file = parsed_parameters['image_file']
    reference_annotation_file = parsed_parameters['reference_annotation_file']
    hypothesis_annotation_file = parsed_parameters['hypothesis_annotation_file']
    compression = int(parsed_parameters['compression'])
    show_frames = int(parsed_parameters['show_frames'])

    os.makedirs(output_directory, exist_ok=True)

    
    # parse annotations
    #
    reference_header, reference_ids, reference_labels, reference_coordinates = fileio_tools.parseAnnotations(reference_annotation_file)

    hypothesis_header, hypothesis_ids, hypothesis_labels, hypothesis_coordinates = fileio_tools.parseAnnotations(hypothesis_annotation_file)
    
    # get height and width of image (in pixels) from the header
    #
    reference_height = int(reference_header['height'])
    reference_width = int(reference_header['width'])

    hypothesis_height = int(hypothesis_header['height'])
    hypothesis_width = int(hypothesis_header['width'])

    if (reference_height != hypothesis_height) or (reference_width != hypothesis_width):
        print("Reference and Hypothesis Sizes Don't Match")
        exit()

    # get labeled regions
    #
    reference_labeled_regions = feats_tools.labeledRegions(reference_coordinates)
    hypothesis_labeled_regions = feats_tools.labeledRegions(hypothesis_coordinates)

    
    # plot labeled regions and label them with appropriate text
    #
    reference_labels_dictionary = {'artf':'black', 'nneo':'red',
                  'bckg':'orange', 'norm':'yellow',
                  'indc':'green','nneo':'blue',
                  'null':'purple','infl':'hotpink',
                  'dcis':'brown','susp':'turquoise'}
    
    hypothesis_labels_dictionary = {'artf':'gray', 'nneo':'lightcoral',
                  'bckg':'bisque', 'norm':'lightyellow',
                  'indc':'lightgreen','nneo':'lightsteelblue',
                  'null':'mediumpurple','infl':'pink',
                  'dcis':'chocolate','susp':'aquamarine'}

    
    for i,z in enumerate(reference_labeled_regions):
        x,y = geometry_tools.getBorder(z)

        plt.plot(x,y,color = reference_labels_dictionary[reference_labels[i]])
        plt.text(reference_coordinates[i][0][0],reference_coordinates[i][0][1],"Reference " + reference_labels[i])

    for i,z in enumerate(hypothesis_labeled_regions):
        x,y = geometry_tools.getBorder(z)

        plt.plot(x,y,color = hypothesis_labels_dictionary[hypothesis_labels[i]])
        plt.text(hypothesis_coordinates[i][0][0],hypothesis_coordinates[i][0][1],"Hypothesis " + hypothesis_labels[i])

    # generate the background image and return the background image's filepath
    #
    background_filepath = svsToJpg(image_file,output_directory+'Background.jpg',compression)

    # Plot the background image
    #
    plt.xlim(0,reference_width)
    plt.ylim(0,reference_height)
    image = plt.imread(background_filepath)

    # show the frames of the image
    #
    if show_frames == 1:
        plotFrames(image_file,frame_size)

    # save the image
    #
    plt.savefig(output_directory+'Output.jpg')
if __name__ == "__main__":
    main()

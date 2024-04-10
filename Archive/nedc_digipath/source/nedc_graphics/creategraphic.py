import sys

sys.path.append('..')
from nedc_digipath_lib import nedc_geometry as pointwithin
from nedc_digipath_lib.nedc_fileio import parse_annotations as annotations

import matplotlib.pyplot as plt
from svstojpg import svs_to_jpg as stj
import returnframeborders as rfb

# function to generate an svs file with label borders drawn and labeled
# needs the annotations in a xml or csv format set as filepath
# and an imagepath set to the correlating svs files
#
def main():

    # set the path to the annotations and image file
    #
    filepath = "/data/isip/data/tuh_dpath_breast/deidentified/v3.0.0/svs/train/00545871_aaaaaagm/s000_2018_01_01/breast/00545871_aaaaaagm_s000_0hne_0000_a001_lvl001_t000.csv"
    imagepath = "/data/isip/data/tuh_dpath_breast/deidentified/v3.0.0/svs/train/00545871_aaaaaagm/s000_2018_01_01/breast/00545871_aaaaaagm_s000_0hne_0000_a001_lvl001_t000.svs"
    
    # Parses the annotations
    #
    HEADER,IDS,LABELS,COORDS = annotations(filepath)
    
    # Gets the dimensions of the image file
    #
    w,h = HEADER['width'],HEADER['height']
    
    # create list for label borders
    #
    shapes = []

    # iterate through the coordinates
    #
    for i in range(len(COORDS)):
        
        # Generate a shape for each list of coordinates
        #
        shapes.append(pointwithin.generate_polygon(COORDS[i]))

        # Plot the borders of each shape
        #
        x,y = pointwithin.get_border(shapes[i])
        plt.plot(x,y)

        # Label each shape with it's appropriate name
        #
        plt.text(COORDS[i][0][0],COORDS[i][0][1],LABELS[i])

    
    # Set the limits of the plot to be equal to the dimensions of the 
    # svs images
    #
    plt.xlim(0,int(w))
    plt.ylim(0,int(h))
    
    # generate the compressed imagefile to be used 
    # in the backgroudn of the plot
    #
    stj(imagepath,"./generatedgraphics/background")

    # Read the image in and set it as the background
    # of the plot
    #
    im = plt.imread("./generatedgraphics/background.jpg")
    plt.imshow(im,extent=[0,int(w),0,int(h)])

    # plot squares of frame size 5000x5000 pixels
    # these represent frames
    #
    # rfb.plt_frames(imagepath,[5000,5000])
    
    # save the plot as demo.png in the DATA directory
    #
    plt.savefig("./generatedgraphics/test.jpg")

main()
#!/usr/bin/env python3
"""! @ Function for reading svs files and their associated csv labels """

# @section description_main Description
# Function for printing out labelled data in the form of feature vectors
#

# @section notes_main Notes
# - This is going to be used in conjuction with a classifier to get patch level analysis of the data going.
#

# Copyright (c) 2024 LYM.  All rights reserved.
#

# @file nedc_pyprint_image.py
#

# @brief Python program for creating feature vectors of labelled data.
#
# @section description_doxygen_example Description
# Driver function
#

# @section nedcy_pyprint_image Imports
# sys is a standard library that allows us to adds the nedc_image_tools to our system path so we could access it.
# nedc_pyprint_image.getargs is for parsing commandline arguments
# nedc_pyprint_image.maketraining is a file that generates labelled data as feature vectors in csv format.
# os is a standard library we are using to check file extensions
#  



# Import sys module for file paths
#
import sys

# Import argungment parsing and image classifier
#
from nedc_pyprint_image import getargs as argspy
from nedc_pyprint_image import maketraining as classify

# For splitting files into their name and extensions
#
import os

# add phuykongs library path
#
sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")

def main():

    # print useful processing message
    #
    print("beginning argument processing...")

    # Parse arguments
    #
    args = argspy.parse_args()
    
    # Load each parsed argument as a variable
    #
    iname =  args.imagefilename
    lname = args.labelfilename
    fsize = args.framesize
    wsize = args.windowsize
    level =  args.level
    xoff =   args.xoff
    yoff =   args.yoff

    # prints parsed arguments
    #
    # print("imagefilename = ",iname,"labelfilename = ",lname,"fsize = ",fsize,"wsize = ",wsize,"level = ",level,"xoff = ",xoff,"yoff = ",yoff)

    # track how many need to be processed and how many need to be processed
    #
    processed = 0
    toprocess = 1

    # splits the file path into name and file extension
    #
    ifile,iextension = os.path.splitext(iname)
    lfile,lextension = os.path.splitext(lname)

    # if file is an svs file and label file is a csv file
    #
    if iextension == ".svs" and (lextension == ".csv" or lextension == ".xml"):
        
        # try classifying it
        #
        if classify.classify_center(iname,lname,wsize,fsize):
            processed+=1

    # print number succesfully classified
    #
    print("\nprocessed {} out of {} files successfully".format(processed,toprocess))

main()

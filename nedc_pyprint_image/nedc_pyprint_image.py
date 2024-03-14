# Imports Phuykong's tools for opening svs file
#
import nedc_image_tools as phg

# Import sys module for file paths
#
import sys

# Import argument parsing
#
import arguments as argspy

# Writing an svs to a jpeg
#
import svstojpg

# Classifies an image
#
import classify

# For splitting files into their name and extensions
#
import os

# For printing svs RGBA values
#
import printrgba as prgba

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
    print("imagefilename = ",iname,"labelfilename = ",lname,"fsize = ",fsize,"wsize = ",wsize,"level = ",level,"xoff = ",xoff,"yoff = ",yoff)

    # printing the image to a jpg
    #

    # svstojpg.svs_to_jpg(iname,"name")

    # closes svs file
    #
    # NIL2.close()


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

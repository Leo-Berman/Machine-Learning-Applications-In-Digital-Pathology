import nedc_image_tools as phg
import sys
import argparse as agp
import numpy as np
from PIL import Image
import argument_parsing as argspy
import nedc_printwindows as winprint 
import nedc_classifycenter as classcent
import os
# add 
sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")

def single_file(filename, height, width, level, xoffset, yoffset):
    NIL = phg.Nil()
    if NIL.is_image(filename) == 2:
        NIL.open(filename)
        window = NIL.read_data((xoffset,yoffset), width, height,"RGBA")
        print(("  {}: ").format(1)+filename)
        for i,x in enumerate(window):
            for j,y in enumerate(x):
                print(("{:>12}").format(str(i*10+j)) + ": (a = " + str(y[3]) + ", r = " + str(y[0]) + ", b = " + str(y[1]) + ", g = " + str(y[2]))
        NIL.close()
        return True
    else:
        NIL.close()
        return False

def file_list(filelist,height,width,level,xoffset,yoffset):
    processed = 0
    for x in filelist:
        if single_file(x,height,width,level,xoffset,yoffset) == True:
            processed+=1
    return processed

def main():

    # print useful processing message
    print("beginning argument processing...")

    args = argspy.parse_args()
    

    # get parses as variables
    iname =  args.imagefilename
    lname = args.labelfilename
    fsize = args.framesize
    wsize = args.windowsize
    level =  args.level
    xoff =   args.xoff
    yoff =   args.yoff

    print("imagefilename = ",iname,"labelfilename = ",lname,"fsize = ",fsize,"wsize = ",wsize,"level = ",level,"xoff = ",xoff,"yoff = ",yoff)

    NIL2 = phg.Nil()
    NIL2.open(iname)

    # printing the image to a jpg
    winprint.windows_to_jpg(NIL2)

    # track how many need to be processed and how many need to be processed
    processed = 0
    toprocess = 1

    # 
    file,extension = os.path.splitext(iname)

    # Pixel size should be the parameter dont worry aobut frame number
    if extension == ".svs":

        # process single file
        #if single_file(fname,height,width,level,xoff,yoff) == True:
        
        if classcent.classify_center(iname,lname,fsize):
            processed+=1

    print("\nprocessed {} out of {} files successfully".format(processed,toprocess))
main()

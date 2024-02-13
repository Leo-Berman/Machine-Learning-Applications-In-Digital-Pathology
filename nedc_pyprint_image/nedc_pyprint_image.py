import nedc_image_tools as phg
import sys
import argparse as agp
import numpy as np
from PIL import Image
import argument_parsing as argspy
import nedc_printwindows as winprint 
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
    fname =  args.filename#"/data/isip/data/fccc_dpath/deidentified/v1.0.0/svs/00000/000000197/001003366/c50.2_c50.2/000000197_001003366_st065_xt1_t000.svs"
    height = args.height#10
    width =  args.width#10
    level =  args.level#0
    xoff =   args.xoff#0
    yoff =   args.yoff#0

    print("filename = ",fname,"height = ",height,"width = ",width,"level = ",level,"xoff = ",xoff,"yoff = ",yoff)

    # Class for using tools from Phuykong's library
    NIL = phg.Nil()

    NIL2 = phg.Nil()
    NIL2.open(fname)
    winprint.windows_to_jpg(NIL2)

    # track how many need to be processed and how many need to be processed
    processed = 0
    toprocess = 1

    # Pixel size should be the parameter dont worry aobut frame number
    if NIL.is_image(fname) == 2:

        # process single file
        if single_file(fname,height,width,level,xoff,yoff) == True:
            processed+=1
    else:
        
        # Process each single file in list
        processed += file_list(file,height,width,level,xoff,yoff)
            
    # final print statement
    print("\nprocessed {} out of {} files successfully".format(processed,toprocess))
main()

import nedc_image_tools as phg
import sys
import argparse as agp
import numpy as np
from PIL import Image

# add 
sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")
def argument_parsing():
    parser = agp.ArgumentParser(
        prog = 'nedc_pypring_image.py',
        description = 'prints the image values for an SVS file'
    )

    parser.add_argument('--height',default=10)
    parser.add_argument('--width',default=10)
    parser.add_argument('-l','--level',default=0)
    parser.add_argument('-x','--xoff',default=0)
    parser.add_argument('-y','--yoff',default=0)
    parser.add_argument('-f','--filename',default="/data/isip/data/fccc_dpath/deidentified/v1.0.0/svs/00000/000000197/001003366/c50.2_c50.2/000000197_001003366_st065_xt1_t000.svs")
    args = parser.parse_args()
    return args

def windows_to_jpg(nil,size):
    # Get dimensions
    xdim,ydim =NIL.get_dimension()
    print("Dimensions = ",xdim,ydim)
    
    # Get all the coordinates for each windows
    coordinates = [(x, y) for x in range(0, xdim, WINDOW_FRAME) for y in range(0, ydim, WINDOW_FRAME)]
    
    # Read all the windows for each coordinate WINDOW_FRAME x WINDOW_FRAME
    windows = NIL.read_data_multithread(coordinates, WINDOW_FRAME, WINDOW_FRAME)
    
    # save all the images as JPEGS
    for index, window in enumerate(windows):
        im = Image.fromarray(window)
        im.save(f'./DATA/window_{index}.jpg', "JPEG")



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

    # get parses as variables
    fname = "/data/isip/data/fccc_dpath/deidentified/v1.0.0/svs/00000/000000197/001003366/c50.2_c50.2/000000197_001003366_st065_xt1_t000.svs"
    height = 10
    width = 10
    level = 0
    xoff = 0
    yoff = 0

    # Class for using tools from Phuykong's library
    NIL = phg.Nil()

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

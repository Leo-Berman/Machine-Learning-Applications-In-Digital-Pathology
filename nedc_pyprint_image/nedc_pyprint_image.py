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

def main():
    # parse arguments
    # Get argument parsing data here /data/isip/tools/linux_x64/nfc/util/python/nedc_pyprint_signal
    args = argument_parsing()

    # print useful processing message
    print("beginning argument processing...")

    # get parses as variables
    fname = str(args.filename)
    height = int(args.height)
    width = int(args.width)
    level = int(args.level)
    xoff = int(args.xoff)
    yoff = int(args.yoff)

    # Class for using tools from Phuykong's library
    NIL = phg.Nil()

    # Pixel size should be the parameter dont worry aobut frame number
    
    if NIL.is_image(fname) == 2:
        # process files
        NIL.open(fname)
        print("passed opening the file")

        # Check if file is an svs file
        if(NIL.is_image(fname) == 2):
            print("is svs image")
        else:
            print("not svs image")
        

        # Read a windows for the coordinate of size WINDOW_FRAME x WINDOW_FRAME
        window = NIL.read_data((xoff,yoff), width, height,"RGBA")
        print(len(window))
        for i,x in enumerate(window):
            for j,y in enumerate(x):
                print(str(i*10+j) + ": (a = " + str(y[0]) + ", r = " + str(y[1]) + ",b = " + str(y[2]) + ",g = " + str(y[3]))

        
        
        
        # if NIL.write("demo","png") == True:
        #     print("Successful Write")
        # else:
        #     print("Failed Write")
        
        NIL.close()
        print("passed closing the data")
        
    else:
        # Treat as list
        pass
main()

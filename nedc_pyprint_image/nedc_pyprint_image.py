import nedc_image_tools as phg
import sys
import argparse as agp
import numpy as npy
# add 
sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")
def argument_parsing():
    parser = agp.ArgumentParser(
        prog = 'nedc_pypring_image.py',
        description = 'prints the image values for an SVS file'
    )

    parser.add_argument('--height')
    parser.add_argument('--width')
    parser.add_argument('-l','--level')
    parser.add_argument('-x','--xoff')
    parser.add_argument('-y','--yoff')
    parser.add_argument('filename')
    args = parser.parse_args()
    return args

def main():
    # parse arguments
    # Get argument parsing data here /data/isip/tools/linux_x64/nfc/util/python/nedc_pyprint_signal
    args = argument_parsing()

    # print useful processing message
    print("beginning argument processing...")

    # get parses as variables
    fname = args.filename
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
        print("in")
        NIL.open(fname)
        NIL.read_data((xoff,yoff),width,height)
        #NIL.read_data()
        NIL.write("demo","jpg")
        NIL.close()
        
        pass
    else:
        # Treat as list
        pass
main()
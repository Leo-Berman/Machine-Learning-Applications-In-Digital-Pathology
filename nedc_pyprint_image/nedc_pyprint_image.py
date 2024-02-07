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

    parser.add_argument('--height',default=10)
    parser.add_argument('--width',default=10)
    parser.add_argument('-l','--level',default=1)
    parser.add_argument('-x','--xoff',default=0)
    parser.add_argument('-y','--yoff',default=0)
    parser.add_argument('-f','--filename',default="/data/isip/data/fccc_dpath/deidentified/v1.0.0/svs/00000/000000197/001003366/c50.2_c50.2/000000197_001003366_st065_xt1_t000.svs")
    args = parser.parse_args()
    return args

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

        if(NIL.is_image(fname) == 2):
            print("is svs image")
        else:
            print("not svs image")


        #print(NIL.read_metadata())
        #print(NIL.read_data((100,100),100,100))
        read_DATA = NIL.read_data()
        print("passed reading the data")



        #if NIL.write("demo","JPEG") == False:
        #    print("Writing Failed")
        #print("passed the writing phase")

        
        print("Dimensions = ",NIL.get_dimension())
    

        NIL.close()
        
        print("passed closing the data")
        
        if NIL.write("demo","JPEG") == True:
            print("Successful Write")
        else:
            print("Failed Write")
        
        pass
    else:
        # Treat as list
        pass
main()

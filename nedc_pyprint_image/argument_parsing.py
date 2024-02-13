'''Make a parsing system using the help function form the C program
and use this directory /data/isip/tools/linux_x64/nfc/util/python/nedc_pyprint_signal 
as a framework on how to use their tools for our argument parsing. All we're doing is
implimenting their argument parsing tool.'''

import nedc_cmdl_parser as ncp
import argparse as agp

#---------------------------------------------------------------------------------------
# parse arguments in this function
#---------------------------------------------------------------------------------------


# define the location of the help files #

HELP_FILE = \
    "/data/isip/tools/linux_x64/nfc/util/cpp/nedc_print_image/nedc_print_image.help"

USAGE_FILE = \
    "/data/isip/tools/linux_x64/nfc/util/cpp/nedc_print_image/nedc_print_image.usage"

# define default argument values #

# number of pixels in the vertical direction [-1=all]
ARG_HEIGHT = "--height"

# number of pixels in the horizontal direction [-1=all]
ARG_WIDTH = "--width"

# the level to be read [0]
ARG_LVL = "--level"
ARG_ABRV_LVL = "-l"

# the upper left horizontal coordinate of the rectangle [0]
ARG_XOFF = "--xoff"
ARG_ABR_XOFF = "-x"

# the upper left vertical coordinate of the rectangle [0]
ARG_YOFF = "--yoff"
ARG_ABR_YOFF = "-y"

# define default argument values #

DEF_HEIGHT = int(-1)
DEF_WIDTH = int(-1)
DEF_LVL = int(0)
DEF_XOFF = float(0)
DEF_YOFF = float(0)

#---------------------------------------------------------------------------------------
# test function from here
#---------------------------------------------------------------------------------------

def main(argv):

    parser = agp.ArgumentParser(
        prog = 'nedc_pypring_image.py',
        description = 'prints the image values for an SVS file'
    )

    # create a command line parser

    parser.add_argument(ARG_HEIGHT, type=int)
    parser.add_argument(ARG_WIDTH, type=int)
    parser.add_argument(ARG_LVL, ARG_ABRV_LVL, type=int)
    parser.add_argument(ARG_XOFF, ARG_ABR_XOFF, type=float)
    parser.add_argument(ARG_YOFF, ARG_ABR_YOFF, type=float)
    parser.add_argument('-f','--filename',default="/data/isip/data/fccc_dpath/deidentified/v1.0.0/svs/00000/000000197/001003366/c50.2_c50.2/000000197_001003366_st065_xt1_t000.svs")
    args = parser.parse_args()

    # get the parameter values
    
    if args.height is None:
        args.height = DEF_HEIGHT

    if args.width is None:
        args.width = DEF_WIDTH
    
    if args.level is None:
        args.level = DEF_LVL
    
    if args.xoff is None:
        args.xoff = DEF_XOFF
    
    if args.yoff is None:
        args.yoff = DEF_YOFF

    # print parsed arguments from here 
    print("height: {}, width: {}, level: {}, xoff: {}, yoff: {}".format(args.width,
                                                                          args.height,
                                                                          args.level,
                                                                          args.xoff,
                                                                          args.yoff))

main()

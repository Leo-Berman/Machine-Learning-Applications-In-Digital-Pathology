"""
    The **getargs.py** module will parse through the arguments for each for each input in the input file and return the arguments to the calling function.
"""

import argparse as agp

#---------------------------------------------------------------------------------------
# parse arguments in this function
#---------------------------------------------------------------------------------------

def parse_args():
    """
        :Objective:
            This function declares the arguments with keywords and default values.\n
            Through the commandline, it reads the arguments inputted. If any argument is not inputted, it is equal to its default value.

        :Arguments: 
            - **- -framesize** (*int*) - Length and width of the frame. Default: -1 (all)
            - **- -windowsize** (*int*) - Length and width of the window. Default: -1 (all)
            - **- -level** or **-l** (*float*) - Level to be read. Default: 0
            - **- -xoff** or **-x** (*float*) - Offset on the x-axis. Default: 0
            - **- -yoff** or **-y** (*float*) - Offset on the y-axis. Default: 0
            *Note: No space between the two hyphens. Only set there for visual purposes.*
        :return: Returns the values of the parsed arguments.
        :rtype: arguments of ints and floats
    """
    # define the location of the help and usage files
    HELP_FILE = \
        "/data/isip/tools/linux_x64/nfc/util/cpp/nedc_print_image/nedc_print_image.help"

    USAGE_FILE = \
        "/data/isip/tools/linux_x64/nfc/util/cpp/nedc_print_image/nedc_print_image.usage"

    # define argument keywords

    # the frame size in pixels [-1=all]
    ARG_FRAMESIZE = "--framesize"

    # the window size in pixels [-1=all]
    ARG_WINDOWSIZE = "--windowsize"

    # the level to be read [0]
    ARG_LVL = "--level"
    ARG_ABRV_LVL = "-l"

    # the offset in the x-axis [0]
    ARG_XOFF = "--xoff"
    ARG_ABR_XOFF = "-x"

    # the offset in the y-axis [0]
    ARG_YOFF = "--yoff"
    ARG_ABR_YOFF = "-y"

    # define default argument values
    DEF_FRAMESIZE = int(-1)
    DEF_WINDOWSIZE = int(-1)
    DEF_LVL = int(0)
    DEF_XOFF = float(0)
    DEF_YOFF = float(0)

    #---------------------------------------------------------------------------------------
    # test function from here
    #---------------------------------------------------------------------------------------

    parser = agp.ArgumentParser(
        prog = 'nedc_pyprint_image.py',
        description = 'prints the image values for an SVS file'
    )

    # create a command line parser
    parser.add_argument(ARG_FRAMESIZE, type=int)
    parser.add_argument(ARG_WINDOWSIZE, type=int)
    parser.add_argument(ARG_LVL, ARG_ABRV_LVL, type=int)
    parser.add_argument(ARG_XOFF, ARG_ABR_XOFF, type=int)
    parser.add_argument(ARG_YOFF, ARG_ABR_YOFF, type=int)
    parser.add_argument('-if','--imagefilename',default="/data/isip/data/fccc_dpath/deidentified/v1.0.0/svs/00000/000000197/001003366/c50.2_c50.2/000000197_001003366_st065_xt1_t000.svs")
    parser.add_argument('-lf','--labelfilename',default="/data/isip/data/fccc_dpath/deidentified/v1.0.0/svs/00000/000000197/001003366/c50.2_c50.2/000000197_001003366_st065_xt1_t000.csv")
    args = parser.parse_args()

    # get the parameter values
    if args.framesize is None:
        args.framesize = DEF_FRAMESIZE

    if args.windowsize is None:
        args.windowsize = DEF_WINDOWSIZE
    
    if args.level is None:
        args.level = DEF_LVL
    
    if args.xoff is None:
        args.xoff = DEF_XOFF
    
    if args.yoff is None:
        args.yoff = DEF_YOFF

    return(args)
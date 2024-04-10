#!/usr/bin/env python
#
# file: $NEDC_NFC/util/python/nedc_pyprint_header/nedc_pyprint_header.py
#
# revision history:
#
# 20200607 (JP): first version
#
# This is a Python version of the C++ utility nedc_print_header.
#------------------------------------------------------------------------------

# import system modules
#
import os
import sys

# import nedc_modules
#
import nedc_cmdl_parser as ncp
import nedc_debug_tools as ndt
import nedc_edf_tools as net
import nedc_file_tools as nft

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# define the location of the help files
#
HELP_FILE = "$MLADP/src/util/nedc_run/nedc_run.help"

USAGE_FILE = "$MLADP/src/util/nedc_run/nedc_run.usage"

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

# declare a global debug object so we can use it in functions
#
dbgl = ndt.Dbgl()

# function: main
#
def main(argv):

    # declare local variables
    #

    # create a command line parser
    #
    cmdl = ncp.Cmdl(USAGE_FILE, HELP_FILE)
    cmdl.add_argument("files", type = str, nargs = '*')

    # parse the command line
    #
    args = cmdl.parse_args()

    # check the number of arguments
    #
    if len(args.files) == int(0):
        cmdl.print_usage('stdout')

    # display an informational message
    #
    print("beginning argument processing...")

    num_files_att = int(0)
    num_files_proc = int(0)

    # display the results
    #
    print("processed %ld out of %ld files successfully" %
	  (num_files_proc, num_files_att))

    # exit gracefully
    #
    return True

# begin gracefully
#
if __name__ == '__main__':
    main(sys.argv[0:])

#
# end of file

import sys
import argparse as agp

def argument_parsing():
    parser = agp.ARgumentParser(
        prog = 'nedc_pypring_image.py',
        description = 'prints the image values for an SVS file'
    )

    parser.add_argument('--height')
    parser.add_argument('--width')
    parser.add_argument('-l','--level')
    parser.add_argument('-x','--xoff')
    parser.add_argument('-y','--yoff')
    args = parser.parse_args()
    return args

def main():
    #args = argument_parsing()
    print("Hello")
    #print(args.height)
from nedc_pyprint_image_lib import getargs as argspy
from nedc_pyprint_image_lib import maketraining as classify

# For splitting files into their name and extensions
#
import os

def main():
    
    """ Program for reading svs files and their associated csv labels
    
        

    """

    '''
    Program for reading svs files and their associated csv labels


    This is the Main driver function for running the program. It takes in command line arguments. It reads those images, and outputs feature vectors in accordance to the defined labels.
        
        Notes:
            This is going to be used in conjuction with a classifier to get patch level analysis of the data going.
    

        Args:
            --imagefilename/-if = image file name
            --labelfilename/-lf = label file name 
            --framesize = frame size 
            --windowsize = window size
            --level/-l = level
            --xoff/-x = x offset 
            --yoff/-y = y offset
    '''

    # print useful processing message
    #
    print("beginning argument processing...")

    # Parse arguments
    #
    args = argspy.parse_args()
    
    # Load each parsed argument as a variable
    #
    iname =  args.imagefilename
    lname = args.labelfilename
    fsize = args.framesize
    wsize = args.windowsize
    level =  args.level
    xoff =   args.xoff
    yoff =   args.yoff

    # prints parsed arguments
    #
    # print("imagefilename = ",iname,"labelfilename = ",lname,"fsize = ",fsize,"wsize = ",wsize,"level = ",level,"xoff = ",xoff,"yoff = ",yoff)

    # track how many need to be processed and how many need to be processed
    #
    processed = 0
    toprocess = 1

    # splits the file path into name and file extension
    #
    ifile,iextension = os.path.splitext(iname)
    lfile,lextension = os.path.splitext(lname)

    # if file is an svs file and label file is a csv file
    #
    if iextension == ".svs" and (lextension == ".csv" or lextension == ".xml"):
        
        # try classifying it
        #
        if classify.classify_center(iname,lname,wsize,fsize):
            processed+=1

    # print number succesfully classified
    #
    print("\nprocessed {} out of {} files successfully".format(processed,toprocess))

main()

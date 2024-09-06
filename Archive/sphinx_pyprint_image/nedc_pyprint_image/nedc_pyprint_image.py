import nedc_pyprint_image_lib
import os

def main():   
    """

        objective:
            This is the Main driver function for running the program. It takes in command line arguments. It reads those images, and outputs feature vectors in accordance to the defined labels.
        
        Return:
            Feature vectors in the form of dct coefficients for svs files
        
        Param:
            --imagefilename/-if = image file name
        
        Param:
            --labelfilename/-lf = label file name 
        
        Param:
            --framesize = frame size 
        
        Param:
            --windowsize = window size
        
        Param:
            --level/-l = level
        
        Param:
            --xoff/-x = x offset 
        
        Param:
            --yoff/-y = y offset

        Example:
            
    """

    # print useful processing message
    #
    print("beginning argument processing...")

    # Parse arguments
    #
    args = nedc_pyprint_image_lib.getargs.parse_args()
    
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
        if nedc_pyprint_image_lib.maketraining.classify_center(iname,lname,wsize,fsize):
            processed+=1

    # print number succesfully classified
    #
    print("\nprocessed {} out of {} files successfully".format(processed,toprocess))

if __name__ == "__main__":
    main()
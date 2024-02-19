from PIL import Image
import nedc_image_tools as phg
import sys
sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")

def classify_center(file,window_frame=[-1,-1]):

    NIL = phg.Nil(file)
    # Get dimensions
    xdim,ydim =NIL.get_dimension()
    #print("Dimensions = ",xdim,ydim)
    
    if window_frame == [-1,-1]:
        window_frame = [xdim,ydim]

    # Get all the coordinates for each windows
    coordinates = [(x, y) for x in range(0, xdim, window_frame[0]) for y in range(0, ydim, window_frame[1])]
    
    # Read all the windows for each coordinate WINDOW_FRAME x WINDOW_FRAME
    # windows = NIL.read_data_multithread(coordinates, window_frame[0], window_frame[1])
    
    # save all the images as JPEGS
    '''
    for x in coordinates:
        if x in border of cancer:
            note own coordinate and cancer type
    '''
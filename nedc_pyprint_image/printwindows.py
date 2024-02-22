from PIL import Image
import nedc_image_tools as phg
import sys
sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")

def windows_to_jpg(NIL,window_frame=[-1,-1]):

    # Get dimensions
    xdim,ydim =NIL.get_dimension()

    #
    if window_frame == [-1,-1]:
        window_frame = [xdim,ydim]

    # Get all the coordinates for each windows
    coordinates = [(x, y) for x in range(0, xdim, window_frame[0]) for y in range(0, ydim, window_frame[1])]
    
    # Read all the windows for each coordinate WINDOW_FRAME x WINDOW_FRAME
    windows = NIL.read_data_multithread(coordinates, window_frame[0], window_frame[1])
    
    # save all the images as JPEGS
    for index, window in enumerate(windows):
        im = Image.fromarray(window)
        im.save(f'./DATA/window_{index}.jpg', "JPEG")
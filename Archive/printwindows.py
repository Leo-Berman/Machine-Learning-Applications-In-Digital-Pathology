from PIL import Image
import nedc_image_tools as phg
import sys
sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")

def windows_to_jpg(imagefile,window_frame=[-1,-1]):

    # open the imagefile
    NIL = phg.Nil()
    NIL.open(imagefile)

    # Get dimensions
    xdim,ydim =NIL.get_dimension()

    #
    if window_frame == [-1,-1]:
        window_frame = [xdim,ydim]

    # Get all the coordinates for each windows
    coordinates = [(x, y) for x in range(0, xdim, window_frame[0]) for y in range(0, ydim, window_frame[1])]
    print(coordinates)
    # Read all the windows for each coordinate WINDOW_FRAME x WINDOW_FRAME
    windows = NIL.read_data_multithread(coordinates, window_frame[0], window_frame[1])
    
    # save all the images as JPEGS
    for index, window in enumerate(windows):
        im = Image.fromarray(window)
        im.save(f'./DATA/window_{index}.jpg', "JPEG")

windows_to_jpg("/data/isip/data/tuh_dpath_breast/deidentified/v2.0.0/svs/train/00707578/s000_2015_04_01/breast/00707578_s000_0hne_0000_a001_lvl000_t000.svs")
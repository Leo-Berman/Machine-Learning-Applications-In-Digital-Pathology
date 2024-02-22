from PIL import Image
import nedc_image_tools as phg
import sys
sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")

def svs_to_jpg(imagefile,name,compress=True):

    NIL = phg.Nil()
    NIL.open(imagefile)

    # Get dimensions
    xdim,ydim =NIL.get_dimension()

    # window_frame = [xdim,ydim]

    # Read the single frame
    windows = NIL.read_data_multithread([[0,0]], xdim, ydim)
    
    # save all the images as JPEGS
    for index, window in enumerate(windows):
        im = Image.fromarray(window)
        if compress == True:
            im=im.resize((xdim//40,ydim//40))
        im.save(name+'.jpg', "JPEG")
    
    return xdim,ydim
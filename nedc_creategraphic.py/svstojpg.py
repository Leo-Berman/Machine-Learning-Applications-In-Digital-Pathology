from PIL import Image
import nedc_image_tools as phg
import sys
sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")

# convert an svs file to compressed jpeg file
#
def svs_to_jpg(imagefile,name):

    # Use Nil library to open up the file
    #
    NIL = phg.Nil()
    NIL.open(imagefile)

    # Get dimensions of the svs image in pixels
    #
    xdim,ydim =NIL.get_dimension()

    # Read the single frame
    #
    windows = NIL.read_data_multithread([[0,0]], xdim, ydim)
    
    # save the images as JPEGS
    # generate the image from RGBA values
    #
    im = Image.fromarray(windows[0])

    # compress the image
    #
    im=im.resize((xdim//40,ydim//40))
    
    # save the image
    #
    im.save(name+'.jpg', "JPEG")
    
    # return the dimensions of the file for use in plotting
    #
    return xdim,ydim

'''
Ex: 
    filepath= *.svs
    name="nameofimage"
    width,height = svs_to_jph(filepath,name)
'''

# svs_to_jpg("/data/isip/data/tuh_dpath_breast/deidentified/v2.0.0/svs/train/00707578/s000_2015_04_01/breast/00707578_s000_0hne_0000_a001_lvl000_t000.svs","compressed")


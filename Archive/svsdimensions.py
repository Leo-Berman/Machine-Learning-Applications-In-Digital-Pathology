import nedc_image_tools as phg
import sys
sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")

# returns the dimensions of an svs file
def get_dimensions(iname):

    # opens the file using nedc_image_tools
    NIL = phg.Nil()
    NIL.open(iname)

    # returns the dimensions as a tuple of pixel measurements (width,height)
    return NIL.get_dimension()

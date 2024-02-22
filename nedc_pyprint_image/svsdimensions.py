import nedc_image_tools as phg
import sys
sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")

def get_dimensions(iname):
    NIL = phg.Nil()
    NIL.open(iname)
    return NIL.get_dimension()
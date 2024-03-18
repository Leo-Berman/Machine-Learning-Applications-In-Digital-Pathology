import nedc_image_tools as phg
import sys
# import printwindows as winprint 
# add 
sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")

def single_file(filename, height, width, level, xoffset, yoffset):
    NIL = phg.Nil()
    if NIL.is_image(filename) == 2:
        NIL.open(filename)
        window = NIL.read_data((xoffset,yoffset), width, height,"RGBA")
        print(("  {}: ").format(1)+filename)
        for i,x in enumerate(window):
            for j,y in enumerate(x):
                print(("{:>12}").format(str(i*10+j)) + ": (a = " + str(y[3]) + ", r = " + str(y[0]) + ", b = " + str(y[1]) + ", g = " + str(y[2]))
        NIL.close()
        return True
    else:
        NIL.close()
        return False

def file_list(filelist,height,width,level,xoffset,yoffset):
    processed = 0
    for x in filelist:
        if single_file(x,height,width,level,xoffset,yoffset) == True:
            processed+=1
    return processed
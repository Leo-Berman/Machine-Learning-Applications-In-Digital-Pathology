# Import phuykongs library
import nedc_image_tools as phg
import sys
import csv

sys.path.insert(0,"/data/isip/tools/linux_x64/nfc/class/python/nedc_image_tools/nedc_image_tools.py")

# convert a single window to a list of rgb values
# window will be a list of list of lists
# the first layer of list will encompass the whole window
# each underlaying list will be a row
# each underlaying list of the row will be a pixel
# the label is the label Yuan will feed the code
# This function is going to take a frame and a label and write that frame
# and label to a csv file. That csv file is going to be in the format
# of relativepath (./DATA/dcttest.csv)
#
def window_to_rgb(imagefile,labels,coords = [(0,0)], window_frame = [50,50],name=""):

    # open the imagefile
    NIL = phg.Nil()
    NIL.open(imagefile)
    xdim,ydim = NIL.get_dimension()
#
    window = NIL.read_data_multithread(coords,npixx = window_frame[0],npixy = window_frame[1],color_mode="RGBA")
    window_list = []
    # save all the images as JPEGS
    for i in range(len(window)):
        workwindow = [labels[i]]
        for j in window[i]:
            for k in j:
                workwindow.extend(k.tolist())
        window_list.append(workwindow)


    return window_list


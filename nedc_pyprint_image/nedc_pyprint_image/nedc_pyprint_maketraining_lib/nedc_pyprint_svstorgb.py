# Import phuykongs library
import nedc_image_tools as phg
import sys
import csv
import numpy as np
from scipy.fftpack import dct

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



def rgba_to_dct(framevalues):

    # Create individual lists for each value of RGBA
    #

    red = []
    green = []
    blue = []
    alpha = []

    # Append corresponding list values in separate RGBA lists
    #

    for i in range(1,len(framevalues)-1,4):
        red.append(framevalues[i])
        green.append(framevalues[i+1])
        blue.append(framevalues[i+2])
        alpha.append(framevalues[i+3])
    
    # concatenate the dcts of each vector
    # probably need to index each DCT for the most
    # signifcant terms
    #
    
    vector = []
    vector.extend(dct(red)[0:10])
    vector.extend(dct(green)[0:10])
    vector.extend(dct(blue)[0:10])
    vector.extend(dct(alpha)[0:10])

    # Convert vector to numpy array
    vector_numpy = np.array(vector).tolist()

    vector_numpy.insert(0,framevalues[0])



    file = open("DATA/QDATrain1.csv",'a')
    writer = csv.writer(file)
    writer.writerow(vector_numpy)

# if __name__ == "__main__":
    # main()





    

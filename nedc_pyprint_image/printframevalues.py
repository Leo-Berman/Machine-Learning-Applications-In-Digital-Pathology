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
def window_to_rgb(imagefile,label,coord = (0,0), window_frame = [50,50]):

    # open the imagefile
    NIL = phg.Nil()
    NIL.open(imagefile)
    xdim,ydim = NIL.get_dimension()
    coord = (coord[0],ydim - coord[1] + window_frame[1])


    # read the single frame
    window = NIL.read_data_multithread([coord],window_frame[0],window_frame[1],color_mode="RGBA")
    
    # save all the images as JPEGS
    window_list = []
    for i in window[0]:
        for j in i:
            window_list.extend(j)

    print(window_list)

    print(int(len(window[0][0])))
    print(type(window))
    column_labels = ["label"]
    for i in range (0, int(len(window_list)/4) - 1):
        column_labels.append(f"r{i}")
        column_labels.append(f"g{i}")
        column_labels.append(f"b{i}")
        column_labels.append(f"a{i}")
    
    #print(column_labels)


    with open('DATA/dctoutput.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_labels)
        writer.writerow(window_list)


# call the function
window_to_rgb("/data/isip/data/tuh_dpath_breast/deidentified/v2.0.0/svs/train/00707578/s000_2015_04_01/breast/00707578_s000_0hne_0000_a001_lvl000_t000.svs",1,(0,0),[50,50])
# Import phuykongs library
import nedc_image_tools as phg
import sys
import csv
from rgbatodct import rgba_to_dct

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
    # print(coords)
    # coord = (coord[0],ydim - coord[1] + window_frame[1])
    # print(label,type(coord))
    # read the single frame
    # this is the proble I think the frame might need to be put in differently ?
    #
    window = NIL.read_data_multithread(coords,npixx = window_frame[0],npixy = window_frame[1],color_mode="RGBA")
    # print("past")
    # print(len(window[0][0][0]))
    window_list = []
    # save all the images as JPEGS
    for i in range(len(window)):
        workwindow = [labels[i]]
        for j in window[i]:
            for k in j:
                workwindow.extend(k.tolist())
        window_list.append(workwindow)


    for x in window_list:
        rbga_to_dct(x)
    # print(len(window_list[0]))
    # print("past")
    # print(window_list)

    # print(int(len(window[0][0])))
    # print(type(window))
    # column_labels = ["label"]
    # for i in range (0, int(len(window_list[0])/4) - 1):
    #     column_labels.append(f"r{i}")
    #     column_labels.append(f"g{i}")
    #     column_labels.append(f"b{i}")
    #     column_labels.append(f"a{i}")
    
    #print(column_labels)


    # with open("DATA/"+name + "dctoutput.csv", 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(column_labels)
    #     writer.writerow(window_list)

    # print("Wrote to DATA/"+name + "dctoutput.csv")

    # print(labels,window_list)
# call the function
# window_to_rgb("/data/isip/data/tuh_dpath_breast/deidentified/v2.0.0/svs/train/00707578/s000_2015_04_01/breast/00707578_s000_0hne_0000_a001_lvl000_t000.svs",1,(0,0),[50,50])
# window_to_rgb("/data/isip/data/tuh_dpath_breast/deidentified/v2.0.0/svs/train/00707578/s000_2015_04_01/breast/00707578_s000_0hne_0000_a001_lvl000_t000.svs",1,(0,0),[50,50])

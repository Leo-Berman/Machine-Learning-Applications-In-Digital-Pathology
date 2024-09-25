# import python libraries
#
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn
import matplotlib.pyplot as plt
import polars
import os
import shapely
from enum import Enum

import nedc_mladp_ann_tools as ann_tools
import nedc_dpath_ann_tools

def plot_histogram(labels,histogram_output):
    '''
    do the thing
    '''

    types = ["norm", "bckg", "artf", "null", "nneo", "infl", "susp", "dcis", "indc"]
    colors = ["lightpink", "peachpuff", "#CBC3DB", "#BAD9BB", "lightblue", "thistle", "#BED4E9", "pink", "#C5CDBA"]

    label_count = {}

    for label in labels:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1

    for t in types:
        if t not in label_count:
            label_count[t] = 0

    l_types = list(label_count.keys())
    counts = list(label_count.values())

    plt.bar(l_types, counts, color=colors)
    plt.xlabel('Label Types')
    plt.ylabel('Number of Labels')

    plt.savefig(histogram_output)
    plt.cla()
# plot confusion matrix 
#
def plot_confusion_matrix(model,inlabels,data,outputpath):
    """
        Objective:
            Plots the confusion matrix.

        :param model: Sklearn model type.
        :type model: sklearn model

        :param labels: List of labels of labeled windows.
        :type labels: list of strings

        :param data: x
        :type data: x

        :param outputpath: Directory path for the output to be stored.
        :type outputpath: path
    """
    
    # generate model predicitions
    #
    predictions = model.predict(data)
    
    # generate confusion matrix with labels and predictions
    #
    conf_mat = confusion_matrix(inlabels, predictions,labels=list(set(inlabels)))
    print(list(set(inlabels)))
    # heatmap the confusion matrix
    #
    seaborn.heatmap(conf_mat, cmap='Blues',yticklabels=list(set(inlabels)),xticklabels=list(set(inlabels)))
    
    # save the figure
    #
    plt.ylabel("Actual Values")
    plt.xlabel("Predicted Values")
    plt.savefig(outputpath)
    plt.cla()
# find the mean confidence %
#
def mean_confidence(model,data):
    """
        Objective:
            Finds the mean of all confidence percentages.

        :param model: Sklearn model type.
        :type model: sklearn model

        :param data: x
        :type data: x
        
        :return: confidence average
        :rtype: float
    """

    # find predictions
    #
    class_predictions=model.predict_proba(data)

    # number for total max predictions
    #
    total_max_predictions = 0
    for x in class_predictions:
        total_max_predictions+=max(x)

    # return the mean of all the confidences
    #
    return total_max_predictions/len(class_predictions)

def generate_frame_decisions(model,data,output_path,frame_locs,framesizes,header):
    """
        Objective:
            Plots regions on an image with associated labels based on predictions.

        :param model: Sklearn model type.
        :type model: sklearn model

        :param output_path: Directory path for the output to be stored.
        :type output_path: path

        :param frame_locs: x
        :type frame_locs: x

        :param framesizes: x
        :type framesizes: x

    """
    
    # get the predictions
    #
    class_predictions=model.predict(data)
    rows = []

    # append the predictions and the guesses
    #
    for i,x in enumerate(class_predictions):
        rows.append([x,framesizes[i],frame_locs[i][0],frame_locs[i][1]])

    # set the column titles and write to csv
    #

    if os.path.exists(output_path):
        os.remove(output_path)

    MySchem = ["labels",'framesizes','top_left_x','top_left_y']
    df = polars.DataFrame(rows,schema=MySchem)
        
    with open(output_path,'a') as file:
        file.write(header + '% ')
        df.write_csv(file)

    return df



def generateRegionDecisions(input_array,framesize):
    label_order = Enum('label_order', 'unlab bckg norm null artf nneo infl susp ndic dcis', start = 0)
    
    # declare dictionaries for patches and frames
    #
    my_regions = {}
    
    # iterate through the labels making a to hold lists
    #
    for x in label_order:
        my_regions[x.value] = []

    # iterate through the numpy array
    #
    for i,row in enumerate(input_array):
        for j,point in enumerate(row):

            # and append to the proper list
            #
            my_regions[point].append(shapely.Polygon([(i,j),(i+1,j),(i+1,j+1),(i,j+1)]))

    # keep track of number of patches written
    #
    patches_written = 0

    # return dictionary
    #
    return_dictionary = {}
    
    # iterate through all the label's list of frames
    #
    for label in my_regions:

        # if that list isn't empty
        #
        if len(my_regions[label]) > 0:


            
            # create a list of indexes to get rid of
            #
            pop_list = []

            # iterate through each patch
            #
            for i,patch in enumerate(my_regions[label]):

                # iterate through every other patch ahead of it in the list
                # and if it intersects with one of those, mark it to be removed and union
                # it to the patch it intersected with and break
                #
                for j,subpatch in enumerate(my_regions[label][i+1:]):
                    if shapely.intersects(patch,my_regions[label][j]):
                        my_regions[label][j+i+1] = shapely.union(patch,subpatch,grid_size=1)
                        pop_list.append(i)
                        break

            # remove the redundant shapes
            #
            for i,x in enumerate(set(sorted(pop_list))):
                my_regions[label].pop(x-i)

            new_patch = {}
                
            # iterate through all the patches
            #
            for patch in my_regions[label]:

                # create a list to hold coordinates
                #
                coordinates = []

                # if it's a multipolygon
                #
                if type(patch) == shapely.geometry.multipolygon.MultiPolygon:

                    # iterate through added the coordinates
                    #
                    for polygon in patch.geoms:
                        extend_list = [ x + (0,) for x in patch.exterior.coords[:]]
                        coordinates.extend(extend_list)

                # if it's not a multipolygon add the coordinates
                #
                else:
                    extend_list = [ x + (0,) for x in patch.exterior.coords[:]]
                    coordinates.extend(extend_list)

                    #print("Patch = ",patch.exterior.coords[:])
                    


                return_dictionary[patches_written] = { 'region_id':patches_written + 1,
                                                       'text':label_order(label).name,
                                                       'coordinates':coordinates,
                                                       'confidence':1,
                                                       'tissue_type':'breast',
                                                       'geometric_properties' : {'Length' : 0.0,
                                                                                 'Area' : 0.0,
                                                                                 'LengthMicrons' : 0.0,
                                                                                 'AreaMicrons' : 0.0}                                                      }
                    
                # and keep track of the number of patches written
                #
                patches_written+=1
                
    # return the dataframe
    #
    return return_dictionary

def generateAnnotationsHeader(input_header:str) -> dict:
    split_items = input_header.split(" = ")
    return_dict = {}
    for i in range(len(0,split_items,2)):
        return_dict[split_items[i]] = split_item[i+1]

    return return_dict
def test():
    test_array = [[0,0],
                  [0,1]]
    graph = generate_region_decisions(test_array,200)
    header = {'bname' : 'test.csv',
              'MicronsPerPixel' : 0,
              'width' : 5000,
              'height' : 5000,
              'tissue' : ['breast']}
    
    annotation_writer = nedc_dpath_ann_tools.AnnDpath()
    annotation_writer.set_type("csv")
    annotation_writer.set_header(header)
    annotation_writer.set_graph(graph)
    annotation_writer.write("test.csv")
if __name__ == "__main__":
    test()

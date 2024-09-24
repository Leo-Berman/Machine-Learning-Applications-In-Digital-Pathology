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



def generate_region_decisions(input_array,framesize):
    label_order = Enum('label_order', 'unlab bckg norm null artf nneo infl susp ndic dcis', start = 0)
    color_order = Enum('color_order', 'white black blue green yellow purple orange pink brown red', start = 0)
    
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

    # set up schema for dataframe
    #
    dataframe_schema = {'index':int,'tissue':str,'label':str,'coord_index':int,'row':int,'column':int,'depth':int,'confidence':float}

    # initialize empty dataframe
    #
    dataframe = polars.DataFrame(schema=dataframe_schema)

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

            # iterate through all the patches
            #
            for patch in my_regions[label]:
                curr_color = color_order(label_order(label).value).name

                # create a list to hold coordinates
                #
                boundary = []

                # if it's a multipolygon
                #
                if type(patch) == shapely.geometry.multipolygon.MultiPolygon:

                    # iterate through added the coordinates
                    #
                    for polygon in patch.geoms:
                        polygon_boundary = polygon.exterior.coords[:]
                        boundary.extend(polygon_boundary)
                        x,y = zip(*polygon_boundary)
                    #print("Boundary = ",boundary)

                # if it's not a multipolygon add the coordinates
                #
                else:
                    #print("Patch = ",patch.exterior.coords[:])
                    boundary.extend(patch.exterior.coords[:])

                # create a list to hold rows
                #
                new_region = []

                # iterate through the coordinates
                #
                for j,y in enumerate(boundary):

                    # get the label from the enumerated type
                    #
                    write_label = label_order(label).name

                    # create the new row
                    #
                    new_region.append([patches_written,"breast",write_label,j,y[0]*framesize,y[1]*framesize,0,1])

                # append the current rows to the frame
                #
                append_frame = polars.DataFrame(new_region,schema=dataframe_schema)
                dataframe.extend(append_frame)

                # and keep track of the number of patches written
                #
                patches_written+=1

    print(dataframe)
                
    # return the dataframe
    #
    return dataframe

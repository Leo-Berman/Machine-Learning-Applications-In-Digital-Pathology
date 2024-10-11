# import python libraries
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn
import matplotlib.pyplot as plt
import polars
import os
import shapely
from enum import Enum

import nedc_mladp_ann_tools as ann_tools
from nedc_mladp_label_enum import label_order

import nedc_dpath_ann_tools

def separateLabels(labels:list,frame_size:tuple,top_left_coordinates:list):
    
    # Get the columns of the dataframe
    #
    labels = dataframe['labels']
    framesizes = df['framesizes']
    top_left_x = df['top_left_x']
    top_left_y = df['top_left_y']

    # Initialize dictionary of labels each containing an empty list
    #
    label_dict = {}
    for x in label_order:
        print(x)
        label_dict[x] = []
        
    # Get the frame size (all frames should be the same size) and the number of rows
    #
    framesize = framesizes[0]

    # Populate dictionary
    #
    for r in range(df.shape[0]-1):
        # Convert frame to bits
        #
        #index_x = (top_left_x[r]//framesize[0])
        #index_y = (top_left_y[r]//framesize[1])
        index_x = int(top_left_x[r])
        index_y = int(top_left_y[r])
        # Append the coordinates to the dictionary with the corresponding labels
        #

        coordinate = (index_x,index_y)
        
        if labels[r] == "bckg":
            label_dict['bckg'].append(coordinate)
        elif labels[r] == "norm":
            label_dict['norm'].append(coordinate)
        elif labels[r] == "null":
            label_dict['null'].append(coordinate)
        elif labels[r] == "artf":
            label_dict['artf'].append(coordinate)
        elif labels[r] == "nneo":
            label_dict['nneo'].append(coordinate)
        elif labels[r] == "infl":
            label_dict['infl'].append(coordinate)
        elif labels[r] == "susp":
            label_dict['susp'].append(coordinate)
        elif labels[r] == "indc":
            label_dict['indc'].append(coordinate)
        elif labels[r] == "dcis":
            label_dict['dcis'].append(coordinate)

    return(label_dict)

def coordinateBitmap(top_left_coordinates:list[tuple], frame_size) -> numpy.array:

    # Convert coords to numpy array for slicing.
    #
    top_left_coordinates = numpy.array(top_left_coordinates)

    # Find the highest x and y dimensions.
    #
    max_x = top_left_coordinates[:,0].max()
    max_y = top_left_coordinates[:,1].max()

    # Divide by frame_dx and frame_dy to get matrix size.
    #
    frame_dx,frame_dy = frame_size
    rows = int(max_y/frame_dy) + 1
    cols = int(max_x/frame_dx) + 1
    
    # Create a matrix.
    #
    m = numpy.zeros((rows,cols), dtype=numpy.uint8)

    # Convert coords to matrix indices.
    #   Coords should always be on frame boundaries or problems will result.
    #
    coords[:,0] = coords[:,0] // frame_dx
    coords[:,1] = coords[:,1] // frame_dy

    # Populate matrix.
    #
    for [col,row] in coords:
        m[row,col] = 1
    
    return m

def regionPredictions(frame_decisions:list, top_left_coordinates:list[tuple],
                      frame_confidences:list, frame_size:tuple):

    separateLabels(
    
    
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
                        extend_list = [ x + (0,) for x in polygon.exterior.coords[:]]
                        coordinates.extend(extend_list)

                # if it's not a multipolygon add the coordinates
                #
                else:
                    extend_list = [ x + (0,) for x in patch.exterior.coords[:]]
                    coordinates.extend(extend_list)

                    #print("Patch = ",patch.exterior.coords[:])
                    

                if label_order(label).name != 'unlab':
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

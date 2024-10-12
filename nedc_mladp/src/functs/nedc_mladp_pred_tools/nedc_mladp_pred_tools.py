# import python libraries
from numba import njit
import os
import shapely
import numpy

from nedc_mladp_label_enum import label_order

import nedc_dpath_ann_tools

def generateSparseMatrixes(labels:list,frame_size:tuple,top_left_coordinates:list,confidences:list):
    
    # Initialize dictionary of labels each containing an empty list
    #
    label_dict = {}
    for x in label_order:
        label_dict[x.name] = {'Confidences':[],
                              'Coordinates':[],
                              }
        
    # Populate dictionary
    #
    for label,top_left_coordinate,confidence in zip(labels,top_left_coordinates,confidences):

        # Convert frame to bits
        #        
        label_dict[label]['Coordinates'].append(top_left_coordinate)
        label_dict[label]['Confidences'].append(confidence)

    return label_dict


def coordinateBitmap(top_left_coordinates:list[tuple], frame_size) -> numpy.array:

    # Convert coords to numpy array for slicing.
    #
    top_left_coordinates = numpy.array(top_left_coordinates).astype(int)

    # Find the highest x and y dimensions.
    #
    max_x = top_left_coordinates[:,0].max()
    max_y = top_left_coordinates[:,1].max()

    # Divide by frame_dx and frame_dy to get matrix size.
    #
    frame_dx,frame_dy = frame_size
    rows = max_y//frame_dy + 1
    cols = max_x//frame_dx + 1
    
    # Create a matrix.
    #
    matrix = numpy.zeros((rows,cols), dtype=numpy.uint8)

    # Convert coords to matrix indices.
    #   Coords should always be on frame boundaries or problems will result.
    #
    top_left_coordinates[:,0] = top_left_coordinates[:,0] // frame_dx
    top_left_coordinates[:,1] = top_left_coordinates[:,1] // frame_dy

    # Populate matrix.
    #
    for [column,row] in top_left_coordinates:
        matrix[row,column] = 1
    
    return matrix

@njit
def inBounds(point:tuple, bottom_right_bnd:tuple, top_left_bnd:tuple = (0,0)) -> bool:

    # Compare columns.
    #
    if top_left_bnd[0] <= point[0] <= bottom_right_bnd[0]:

        # Compare rows.
        #
        if top_left_bnd[1] <= point[1] <= bottom_right_bnd[1]:
            return True
        
    return False


def floodFill(matrix:numpy.ndarray, start_point:tuple) -> None:
    
    # Get the bottom-rightmost point index. 
    #
    bottom_right_bnd = numpy.subtract(matrix.shape,(1,1))
    
    # Start the flood-fill at (0,0), appending nearby points if their value is zero.
    # 
    indices = [start_point]
    while indices:
        
        # Get the first item in indices.
        #
        index = indices[0]

        # Check and set the value at that index to 1.
        #
        if matrix[index] != 1:
            matrix[index] = 1
        else:
            # No need to continue the loop body if the bit is already set.
            indices.pop(0)
            continue

        # Get indices for nearby points
        #
        left = tuple(numpy.add(index,(0,-1)))
        right = tuple(numpy.add(index,(0,1)))
        up = tuple(numpy.add(index,(-1,0)))
        down = tuple(numpy.add(index,(1,0)))

        # Check to see if those points are on the matrix.
        #   If they are, add them to indices.
        # 
        if inBounds(up,bottom_right_bnd) and matrix[up] != 1:
            indices.append(up)
        if inBounds(down,bottom_right_bnd) and matrix[down] != 1:
            indices.append(down)
        if inBounds(left,bottom_right_bnd) and matrix[left] != 1:
            indices.append(left)
        if inBounds(right,bottom_right_bnd) and matrix[right] != 1:
            indices.append(right)

        # Continue iterating until all points in indices are exhausted.
        #
        indices.pop(0)

def padAndFill(matrix:numpy.ndarray) -> numpy.array:

    # Create a copy of the matrix.
    #
    mask = matrix.copy()

    # Pad the copied matrix edges (top, bottom, sides) with zeroes
    #
    mask = numpy.pad(mask, 1, 'constant', constant_values=0)
    
    # Flood fill the copy starting at (0,0).
    #
    floodFill(mask, (0,0))
    
    # Invert the values of the copied matrix.
    #
    mask = 1-mask

    # Remove padding.
    #
    mask = mask[1:-1,1:-1]
    
    # Superimpose the original and copied matrices.
    #
    return matrix + mask

def resizeMatrices(matrix_1:numpy.array, matrix_2:numpy.array) -> tuple[numpy.array]:

    # Get differences in dimensions.
    #
    size_difference = numpy.array(matrix_1.shape) - numpy.array(matrix_2.shape)

    # Resize along each axis an amount diff. 
    #
    for axis,difference in enumerate(size_difference):

        # numpy.pad requires a pad_width in the format ((left,right),(up,down)),
        #   using pad[0] resizes to the right, 
        #   using pad[1] resizes downward. 
        #
        pad = [((0,abs(difference)),(0,0)), ((0,0),(0,abs(difference)))]

        # If m2 is smaller, resize m2. 
        #
        if difference > 0:
            matrix_2 = numpy.pad(matrix_2, pad[axis], 'constant', constant_values=0)

        # If m1 is smaller, resize m1. 
        #
        if difference < 0:
            matrix_1 = numpy.pad(matrix_1, pad[axis], 'constant', constant_values=0)

    return (matrix_1,matrix_2)

def generateHeatmap(sparse_matrixes:dict, framesize:tuple) -> numpy.array:

    # Initialize 2D return array.
    #
    return_matrix = numpy.array([[0]], dtype=numpy.uint8)
    
    # Iterate through all labels and coordinates.
    #
    for label,coordinates_confidences in sparse_matrixes.items():
        # For each (label,coordinates) pair...

        if len(coordinates_confidences['Coordinates']) > 0:
        
            # Convert coordinates for the label to a bit matrix.
            #
            matrix = coordinateBitmap(coordinates_confidences['Coordinates'], framesize)
            
            # Fill in regions that are bounded on 4 sides.
            #
            padAndFill(matrix)
            
            # Resize return and label matrices so they are equal dimensions.
            #
            matrix,return_matrix = resizeMatrices(matrix, return_matrix)
            
            # Squash the matrices together.
            #
            label_num = label_order[label].value
            return_matrix = numpy.maximum(return_matrix, matrix*label_num)
            
            # Repeat.
            
    return return_matrix


def regionPredictions(frame_decisions:list, top_left_coordinates:list[tuple],
                      frame_confidences:list, frame_size:tuple):

    sparse_matrixes = generateSparseMatrixes(frame_decisions,frame_size,top_left_coordinates,frame_confidences)

    heatmap = generateHeatmap(sparse_matrixes,frame_size)
    
    # declare dictionaries for patches and frames
    #
    my_regions = {}
    
    # iterate through the labels making a to hold lists
    #
    for x in label_order:
        my_regions[x.value] = []

    # iterate through the numpy array
    #
    for i,row in enumerate(heatmap):
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


            
            # iterate through every other patch ahead of it in the list
            # and if it intersects with one of those, mark it to be removed and union
            # it to the patch it intersected with and break
            #
            i = 0
            while i < len(my_regions[label])-1:
                if shapely.intersects(my_regions[label][i],my_regions[label][i+1]):
                    my_regions[label][i+1] = shapely.union(my_regions[label][i],my_regions[label][i+1],grid_size=1)
                    my_regions[label].pop(i)
                else:
                    i+=1
                
            new_patch = {}

            confidence = sum(sparse_matrixes[label_order(label).name]['Confidences'])/(len(sparse_matrixes[label_order(label).name]['Confidences'])+1)
            
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
                        extend_list = [ (x[0]*frame_size[0], x[1]*frame_size[1], 0) for x in polygon.exterior.coords[:]]
                        coordinates.extend(extend_list)

                # if it's not a multipolygon add the coordinates
                #
                else:
                    extend_list = [ (x[0]*frame_size[0], x[1]*frame_size[1], 0) for x in patch.exterior.coords[:]]
                    coordinates.extend(extend_list)

                    #print("Patch = ",patch.exterior.coords[:])
                    

                if label_order(label).name != 'unlab':
                    return_dictionary[patches_written] = { 'region_id':patches_written+1,
                                                           'text':label_order(label).name,
                                                           'coordinates':coordinates,
                                                           'confidence':confidence,
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

# import python libraries
import numpy
import scipy
import shapely
import math
# import picones libraries
#
import nedc_image_tools

def generateTopLeftFrameCoordinates(height:int, width:int,
                                    frame_size:tuple)->list:
    return_list = []
    for x in range(0,width,frame_size[0]):
        for y in range(0,height,frame_size[1]):
            return_list.append( (x,y) )
    return return_list

def generateWindows(coordinates:list, frame_size:tuple,
                    window_size:tuple)->list:

    windows = []

    window_width_offset = (window_size[0]-frame_size[0])//2
    window_height_offset = (window_size[1]-frame_size[1])//2
    
    for x in coordinates:

        up_most = x[1] - window_height_offset
        down_most = x[1] + frame_size[0] + window_height_offset

        right_most = x[0] + frame_size[1] + window_width_offset
        left_most = x[0] - window_width_offset

        top_left = [left_most, up_most]
        top_right = [right_most, up_most]
        bottom_right = [right_most, down_most]
        bottom_left = [left_most, down_most]

        windows.append(shapely.Polygon([top_left,top_right,
                                        bottom_right,bottom_left]))

    return windows


def classifyFrames(labels:list, height:int, width:int, window_size:tuple,
                   frame_size:tuple, regions:list, overlap_threshold:float):
    
    return_labels = []

    top_left_frame_coords = generateTopLeftFrameCoordinates(height, width,
                                                            frame_size)
    
    windows = generateWindows(top_left_frame_coords,frame_size, window_size)

    return_top_left_frame_coords = []

    for w,x in zip(top_left_frame_coords,windows):
        for y,z in zip(regions,labels):
            overlap = shapely.intersection(x,y)
            if overlap.area/x.area >= overlap_threshold:
                return_labels.append(z)
                return_top_left_frame_coords.append(w)
                break

    return return_top_left_frame_coords,return_labels


def windowRGBValues(image_file:str, frame_top_left_coordinates:list,
                    window_size:tuple):
    
    # open the imagefile
    # 
    image_reader = nedc_image_tools.Nil()
    image_reader.open(image_file)

    # read all of the windows into memory
    #
    windows = image_reader.read_data_multithread(frame_top_left_coordinates,
                                                 npixy = window_size[1],
                                                 npixx = window_size[0],
                                                 color_mode="RGB")
    
    # return list of lists of rgba values
    #
    return windows


# perform discrete cosine transform on rgba values
#
def windowDCT(window_RGBs:list):
    window_DCTs = []
    
    for i,window in enumerate(window_RGBs):

        window_DCTs.append(numpy.array([scipy.fftpack.dctn(color) for color in window]).flatten())

        
    return window_DCTs

def labeledRegions(coordinates:list):
    # generate polygon of regions within the image
    #
    ret_shapes = []
    for pp in coordinates:
        # compute centroid
        cent=(sum([p[0] for p in pp])/len(pp),sum([p[1] for p in pp])/len(pp))
        # sort by polar angle
        pp.sort(key=lambda p: math.atan2(p[1]-cent[1],p[0]-cent[0]))
        ret_shapes.append(shapely.Polygon(pp))

    return ret_shapes


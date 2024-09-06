# import python libraries
#
import numpy
import scipy

# perform discrete cosine transform on rgba values
#
def rgba_to_dct(framelist:list,frame_coord_list:list,framesize:int):

    # list of final rows to return
    #
    list_of_rows = []
    
    # iterate through each frame
    #
    for i,framevalues in enumerate(framelist):        
        red = []
        green = []
        blue = []
        alpha = []

        # Append corresponding list values in separate RGBA lists
        #
        for j in range(1,len(framevalues)-1,4):
            red.append(framevalues[j])
            green.append(framevalues[j+1])
            blue.append(framevalues[j+2])
            alpha.append(framevalues[j+3])

        # concatenate the dcts of each vector
        # probably need to index each DCT for the most
        # signifcant terms
        #
        vector = []
        vector.extend(scipy.fftpack.dct(red)[0:10])
        vector.extend(scipy.fftpack.dct(green)[0:10])
        vector.extend(scipy.fftpack.dct(blue)[0:10])
        vector.extend(scipy.fftpack.dct(alpha)[0:10])

        # Convert vector to numpy array to a list for a reason I'm not sure of
        #
        vector_numpy = numpy.array(vector).tolist()

        # insert the framesize, x and y coordinates, and the dct coefficients
        #
        vector_numpy.insert(0,framesize)
        vector_numpy.insert(0,frame_coord_list[i][1])
        vector_numpy.insert(0,frame_coord_list[i][0])
        vector_numpy.insert(0,framevalues[0])

        # append that to the final row list
        #
        list_of_rows.append(vector_numpy)

    # return that list of rows
    #
    return list_of_rows

def even_data(indata,inlabels):
    retdata = []
    retlabels = []

    inlabels=list(inlabels)
    imbalanced_label = max(set(inlabels), key=inlabels.count)
    mycount = int((len(inlabels) - inlabels.count(imbalanced_label)))
    for x,y in list(zip(inlabels,indata)):
        if x == imbalanced_label and mycount > 0:
            retdata.append(y)
            retlabels.append(x)
            mycount-=1

        elif x != imbalanced_label:
            retdata.append(y)
            retlabels.append(x)
        else:
            pass
    return retdata,retlabels

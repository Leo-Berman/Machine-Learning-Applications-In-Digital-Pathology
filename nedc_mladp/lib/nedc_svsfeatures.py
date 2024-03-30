import numpy
import scipy

def rgba_to_dct(framelist:list,frame_coord_list:list,framesize:int):

    list_of_rows = []
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

        # Convert vector to numpy array
        vector_numpy = numpy.array(vector).tolist()

        vector_numpy.insert(0,framesize)
        vector_numpy.insert(0,frame_coord_list[i][0])
        vector_numpy.insert(0,frame_coord_list[i][1])
        vector_numpy.insert(0,framevalues[0])

        list_of_rows.append(vector_numpy)

    return list_of_rows
import numpy
import scipy

def rgba_to_dct(framelist:list):
    
    list_of_rows = []
    for framevalues in framelist:        
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
        vector.extend(scipy.fftpack.dct(red)[0:10])
        vector.extend(scipy.fftpack.dct(green)[0:10])
        vector.extend(scipy.fftpack.dct(blue)[0:10])
        vector.extend(scipy.fftpack.dct(alpha)[0:10])

        # Convert vector to numpy array
        vector_numpy = numpy.array(vector).tolist()

        vector_numpy.insert(0,framevalues[0])

        list_of_rows.append(vector_numpy)

    return list_of_rows
import csv
import numpy as np
#import sys
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt


def rgba_to_dct(framevalues):
    # path = 'DATA/dctoutput.csv'

    # Open CSV file in read mode with newline to skip "/n"
    # Create a list from the CSV reader object




    # Create individual lists for each value of RGBA
    #

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

  
    #Display the RGBA lists
    #
        
    #print(f"Red: {red}")
    #print(f"Green: {green}")
    #print(f"Blue: {blue}")
    #print(f"Alpha: {alpha}")
    
    # concatenate the dcts of each vector
    # probably need to index each DCT for the most
    # signifcant terms
    #
    
    vector = []
    vector.extend(dct(red)[0:10])
    vector.extend(dct(green)[0:10])
    vector.extend(dct(blue)[0:10])
    vector.extend(dct(alpha)[0:10])

    #print(len(vector))

    # Convert vector to numpy array
    vector_numpy = np.array(vector).tolist()

    # Get absolute values of coefficients
    #

    # magnitude_coeffs = np.abs(vector)
    #print(magnitude_coeffs)

    # Determine a high pass threshold
    # 75th percentile
    #

    # threshold = np.percentile(magnitude_coeffs, 99.999)
    # print(threshold)


    # Performing an Inverse DCT to reconstruct original vector
    # Many of the transformed DCT coefficients are close to zero
    # After inverse DCT those coefficients are automatically getting discarded
    
    # reconstructed_vector = idct(vector)

    # print(type(reconstructed_vector))

    # High Pass Filter
    # Preserving Higher Frequencies (Lower Magnitudes/ Energies)
    # As Higher Frequencies denote texture boundaries

    # High_Freq_Coeffs = vector_numpy[magnitude_coeffs < threshold]
    # High_Freq_Coeffs = High_Freq_Coeffs.tolist()
    # print(High_Freq_Coeffs)
    vector_numpy.insert(0,framevalues[0])
    # High_Freq_Coeffs.insert(0,framevalues[0])
    # Do an Inverse DCT 
    #

    #High_Freq_Reconstruct = idct(High_Freq_Coeffs)
    #print(len(High_Freq_Reconstruct))



    
    


    '''plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(reconstructed_vector)
    plt.title('DCT Reconstructed Vector')
    plt.xlabel('Index')
    plt.ylabel('Value')

    plt.tight_layout()
    plt.show()'''

    # write the vector to a file
    #

    file = open("DATA/QDAEval2.csv",'a')
    writer = csv.writer(file)
    writer.writerow(vector_numpy)

# if __name__ == "__main__":
    # main()





    

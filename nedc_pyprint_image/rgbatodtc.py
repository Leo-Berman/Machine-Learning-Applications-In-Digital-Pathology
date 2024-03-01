import csv
import numpy as np
#import sys
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt


def rgba_to_dct(framevalues):
    path = 'DATA/dctoutput.csv'

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

    for i in range(0,len(framevalues)-1,4):
        red.append(int(read[1][i]))
        green.append(int(read[1][i+1]))
        blue.append(int(read[1][i+2]))
        alpha.append(int(read[1][i+3]))

  
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
    vector.extend(dct(red))
    vector.extend(dct(green))
    vector.extend(dct(blue))
    vector.extend(dct(alpha))

    #print(len(vector))

    # Convert vector to numpy array
    vector_numpy = np.array(vector)

    # Get absolute values of coefficients
    #

    magnitude_coeffs = np.abs(vector)
    #print(magnitude_coeffs)

    # Determine a high pass threshold
    # 75th percentile
    #

    threshold = np.percentile(magnitude_coeffs, 90)
    print(threshold)


    # Performing an Inverse DCT to reconstruct original vector
    # Many of the transformed DCT coefficients are close to zero
    # After inverse DCT those coefficients are automatically getting discarded
    
    # reconstructed_vector = idct(vector)

    # print(type(reconstructed_vector))

    # High Pass Filter
    # Preserving Higher Frequencies (Lower Magnitudes/ Energies)
    # As Higher Frequencies denote texture boundaries

    High_Freq_Coeffs = vector_numpy[magnitude_coeffs < threshold]
    print(len(High_Freq_Coeffs))

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

    file = open("DATA/PCATest.csv",'w')
    writer = csv.writer(file)
    writer.writerow(High_Freq_Coeffs)

# if __name__ == "__main__":
    # main()





    

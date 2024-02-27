import csv
from scipy.fftpack import dct


def main(relpath = "./DATA/dcttest.csv"):
    
    # open the csv and read in each row
    #
    file = open(relpath,'r')
    data = csv.reader(file)
    read = []
    for row in data:
        read.append(row)

    # create a list for each value of RGBA
    #
    r = []
    g = []
    b = []
    a = []
    
    # iterate through each row  except for the first one
    # and append each value to it's appropriate vector
    #
    for i in range(1,len(read[1])-1,4):
        r.append(int(read[1][i]))
        g.append(int(read[1][i+1]))
        b.append(int(read[1][i+2]))
        a.append(int(read[1][i+3]))
    
    # concatenate the dcts of each vector
    # probably need to index each DCT for the most
    # signifcant terms
    #
    vector = []
    vector.extend(dct(r))
    vector.extend(dct(g))
    vector.extend(dct(b))
    
    # write the vector to a file
    #
    file = open("DATA/PCA.csv",'w')
    writer = csv.writer(file)
    writer.writerow(vector)

main()
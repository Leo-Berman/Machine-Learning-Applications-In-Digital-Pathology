import csv
from scipy.fftpack import dct
relpath = "./DATA/dcttest.csv"
def main():
    '''do a discrete cosine transform of the data in relpath'''
    file = open(relpath,'r')
    data = csv.reader(file)
    read = []
    for row in data:
        read.append(row)

    r = []
    g = []
    b = []
    a = []
    for i in range(1,len(read[1])-1,4):
        r.append(int(read[1][i]))
        g.append(int(read[1][i+1]))
        b.append(int(read[1][i+2]))
        a.append(int(read[1][i+3]))
    
    vector = []
    vector.extend(dct(r))
    vector.extend(dct(g))
    vector.extend(dct(b))
    print(vector)
    file = open("DATA/PCA.csv",'w')
    writer = csv.writer(file)
    writer.writerow(vector)
main()
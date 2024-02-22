from sklearn.decomposition import PCA
import csv
import numpy as np
def main():
    relpath = "./DATA/PCA.csv"
    file = open(relpath,'r')
    data = csv.reader(file)
    read = []
    for row in data:
        read.append(row)
    pca = PCA(n_components=1)
    pca.fit(np.array(read[0]).reshape(-1,1))
    print(read)

main()
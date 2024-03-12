from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import csv
import numpy as np
def main():

    # read the rows into memory
    #
    relpath = "./DATA/PCATest.csv"
    file = open(relpath,'r')
    data = csv.reader(file)
    read = []
    for row in data:
        read.append(row)

    # list for holding the labels and data
    #
    mydata = []
    labels = []
    
    #  iterate through the data
    #
    for i,x in enumerate(read):

        # add the label to a list
        #
        label = x.pop(0)
        labels.append(label)

        # add the data into a separate list
        #
        mydata.append(x)
        # row_length = (len(x))//4
        # red = x[0:row_length]
        # green = x[row_length:row_length*2]
        # blue = x[row_length*2:row_length*3]
        # alpha = x[row_length*3:row_length*4]

        # applist = [red,green,blue,alpha]
        # mydata.append(applist)

    # reshape the arrays
    #
    labels = np.array(labels).ravel()
    print(len(labels))
    mydata = np.array(mydata)
    print(len(mydata))

    # Fit the model
    #
    QDA = QuadraticDiscriminantAnalysis()
    QDA.fit(mydata, labels)
    
    # print the score 
    #
    print(QDA.score(mydata,labels))
    guesses = []

    # get a list of guesses
    #
    for x in mydata:
        guesses.append(QDA.predict(x.reshape(1,-1))[0])
    print(guesses)

    # print("label = ", mydata[0][0])
    # print("red length = ",len(mydata[0][1]))
    # print("green length = ",len(mydata[0][2]))
    # print("blue length = ",len(mydata[0][3]))
    # print("alpha length = ",len(mydata[0][4]))


main()
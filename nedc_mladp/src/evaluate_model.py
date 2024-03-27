import joblib
import os
import csv
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
def main():
    trained_model = joblib.load('../dat/TRAINED_MODELS/Trained_QDA.joblib')
    
    # change directory to the appropriate eval data file
    #
    data_file = "../dat/EVAL_DATA"
    
    # set the directory to that data path
    #
    os.chdir(data_file)
    
    # set the list of datapoints to all the files within that directory
    #
    eval_list = os.listdir()
    
    # list to hold vectors in [name,feature1,feature2, .. featuren] format
    #
    read = []
    
    # iterate through all the datafiles
    #
    for x in eval_list:
        
        # read the rows into memory
        #
        file = open(x,'r')
        data = csv.reader(file)
        for row in data:
            for j in range(len(row)):
                row[j] = row[j][1:len(row[j])-1].replace("'","").split(",")
                read.append(row[j])
        file.close()

    # list for holding the labels and data
    #
    mydata = []
    labels = []

    #  iterate through the data
    #
    for x in read:

        # add the label to a list
        #
        label = x.pop(0)
        labels.append(label)

        # add the data into a separate list
        #
        mydata.append(np.array([float(z) for z in x]))

    # reshape the arrays
    #
    labels = np.array(labels).ravel()
    mydata = np.array(mydata)

    # print the overall score
    #
    print(trained_model.score(mydata,labels))
if __name__ == "__main__":
    main()
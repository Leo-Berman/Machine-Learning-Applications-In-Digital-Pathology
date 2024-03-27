#!/usr/bin/env python3
"""! @brief Clasifies the DCT values into different labeled regions"""
##
# @file classify_dct_values.py
#
# @brief Classify the DCT values into different labeled regions
#
# 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier as RNF
import csv
import numpy as np
import os
import joblib
def main():

    # change directory to the appropriate train data file
    #
    data_file = "../dat/TRAIN_DATA"
    os.chdir(data_file)
    
    # set the list of datapoints to all the files within that directory
    #
    train_list = os.listdir()
    
    # list to hold vectors in [name,feature1,feature2, .. featuren] format
    #
    read = []

    # iterate through the entire training list
    #
    for x in train_list:

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
    
    # Fit the model
    #
    model = RNF()
    model.fit(mydata, labels)
    print(model.score(mydata,labels))

    # change the directory to output themodel
    #
    os.chdir("../TRAINED_MODELS")

    # dump the model there
    #
    joblib.dump(model,'Trained_RNF.joblib')

    

main()
#!/usr/bin/env python3
"""! @brief Clasifies the DCT values into different labeled regions"""
##
# @file classify_dct_values.py
#
# @brief Classify the DCT values into different labeled regions
#
# 
import sklearn
import csv
import numpy as np
import os
import jolib
def main():

    # change directory to the appropriate train data file
    #
    data_file = "../nedc_gen_feats/TRAIN_DATA"
    os.chdir(data_file)
    
    # set the list of datapoints to all the files within that directory
    #
    train_list = os.listdir()
    
    # train_list = ["QDATrain2"]
    read = []
    
    # iterate through the entire training list
    #
    for x in train_list:

        # read the rows into memory
        #
        relpath = "./DATA/"+x+".csv"
        file = open(relpath,'r')
        data = csv.reader(file)
        for row in data:
            read.append(row)
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
    print(len(labels))
    
    # Fit the model
    #
    QDA = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()
    QDA.fit(mydata, labels)
    
    os.chdir("../../nedc_train_model/trained_models")

    jolib.dump(QDA,'Trained_QDA.joblib')

    

main()
# import system modules
import os
import sys
import numpy as np
import pandas as pd
import random
import PIL
import torch
import torch.utils as utils

TRAIN, DEV, EVAL = 'train', 'dev', 'eval'
IMG_EXT = '*.tif'
DEF_MODEL_FNAME = "model.pckl"

# import project specific libraries
from nedc_mladp_label_enum import label_order

def parsePCA(filelist):
    '''
    description: 
        parse PCA file for the labels and PCA features.

    arguments:
        :filelist: list of CSV files.
    
    return:
        :totaldata: numpy array containing labels, PCA features 0-n
    '''
    all_data = []

    with open(filelist, "r") as files:
        for x in files:
            x = x.strip()
            df = pd.read_csv(x, skiprows=9, dtype=str, keep_default_na=False)
            pca_columns = df.filter(like='PCA')
            df_parsed = pd.concat([df[['Label']], pca_columns], axis=1)
            all_data.append(df_parsed.to_numpy())

        
    totaldata = np.vstack(all_data)

    return totaldata

def parsePCA_file(file):
    '''
    description: 
        parse PCA file for the labels and PCA features.

    arguments:
        :file: CSV file
    
    return:
        :totaldata: numpy array containing labels, PCA features 0-n
    '''
    all_data = []

    df = pd.read_csv(file, skiprows=9)
    pca_columns = df.filter(like='PCA')
    df_parsed = pd.concat([df[['Label']], pca_columns], axis=1)
    all_data.append(df_parsed.to_numpy())
        
    totaldata = np.vstack(all_data)

    return totaldata

def randomData(num_frames:int):

    # Set the random seed for reproducibility
    np.random.seed(42)

    # Create a NumPy array of random 32x32x4 arrays (IMAGES)
    # data = np.random.rand(num_frames, 32, 32, 4)

    # Create a NumPy array of random 1x1000 arrays (PCA)
    data = np.random.rand(num_frames, 1, 1000)

    # Generate random label digits (as strings) corresponding to the number of arrays
    labels = np.array([str(np.random.randint(0, 10)) for _ in range(num_frames)])


    # Convert the labels and features to the correct types
    data = data.astype(float)
    labels = labels.astype(int)

    print(np.shape(data))
    print(np.shape(labels))

    return data, labels

def correctType(data, labels):
    '''
    convert the list of labels as a list of digits (int) with label_order.

    arguments:
        :data: numpy-type 3D array.
        :labels: numpy-type 2D array.

    return:
        :data: (float) numpy-type 3D array.
        :labels: (int) numpy-type 2D array.
    '''
    
    i = 0
    
    for l in labels:
        labels[i] = label_order[l].value
        i += 1

    data = data.astype(float)
    labels = labels.astype(int)

    return data,labels

def fillClasses(classes, counts):
    '''
    Fill the list of unique classes  with the missing classes.
    Initialize those missing classes to zero counts.

    arguments:
        :classes: array of unique classes.
        :counts: array of counts of the unique classes.

    return:
        :all_classes: array of all classes (0-8).
        :all_counts: array of counts for all classes.
    '''

    # Initialize arrays for classes and zero counts
    all_classes = np.arange(0,9)
    all_counts = np.zeros_like(all_classes)

    # Fill the arrays with the counts to the corresponding indices
    for i,cls in enumerate(classes):
        all_counts[cls] = counts[i]

    return all_classes, all_counts

def getClasses(digits):
    labels = []
    for d in digits:
        labels.append(label_order(int(d)+1).name)
    return labels
# import system modules
import os
import sys
import numpy as np
import pandas as pd
import random
import PIL
import torch
import torch.utils as utils

# import project specific libraries
from nedc_mladp_ann_tools import label_order

def parsePCA(filelist):
    '''
    Description: Parse PCA file for the labels and PCA features.

    :param file: list of CSV files.
    :return: Numpy array containing labels, PCA features 0-n
    '''
    all_data = []

    for x in filelist:
        df = pd.read_csv(x, skiprows=9)
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

    return data, labels

def correctType(data, labels):
    '''
    Convert the list of labels as a list of digits (int) with label_order.

    :param labels: string numpy-type 2D array.
    :return labels: int numpy-type 2D array.
    '''
    
    i = 0
    for l in labels:
        labels[i] = label_order[l].value
        i += 1

    data = data.astype(float)
    labels = labels.astype(int)

    return data,labels

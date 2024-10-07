#!/usr/bin/env python
import os
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision as tv
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

from nedc_mladp_ann_tools import label_order

def getDataLoaders(data,labels):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # # Load the training data from your directory
    # train_data = tv.datasets.ImageFolder(root='./data/train', transform=transform)
    # train_loader = tv.DataLoader(train_data, batch_size=4, shuffle=True)

    # # Load the testing data
    # test_data = tv.datasets.ImageFolder(root='./data/test', transform=transform)
    # test_loader = tv.DataLoader(test_data, batch_size=4, shuffle=False)

    # Convert the list of labels as a list of digits with label order.
    #
    i = 0
    for l in labels:
        labels[i] = label_order[l].value
        i += 1

    # Load the extracted features into a 3 dimensional tensor.
    #
    features_tensor = torch.tensor(data, dtype=torch.float32)

    # Load the labels into a 2 dimensional tensor.
    #
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Reshape the tensor -- # Shape: [num_frames, 4 (channels), width, height]
    #
    num_frames = features_tensor.shape[0]
    num_pixels = features_tensor.shape[1] // 4
    features_tensor = features_tensor.view(num_frames, 4, num_pixels, 1)

    print(features_tensor.shape)
    print(len(labels))


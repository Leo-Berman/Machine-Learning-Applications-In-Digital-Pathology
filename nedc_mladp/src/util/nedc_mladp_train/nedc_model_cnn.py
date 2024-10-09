#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

# Torch library and modules
import torch
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Local tools
from nedc_mladp_ann_tools import label_order
from nedc_cnn_tools import random_data

def model_CNN(data,labels):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data,labels = random_data(5)

    # Convert the list of labels as a list of digits with label order.
    #
    # i = 0
    # for l in labels:
    #     labels[i] = label_order[l].value
    #     i += 1
        
    # Convert the numpy types to numbers
    #
    data = data.astype(float)
    labels = labels.astype(int)

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
    features_tensor = features_tensor.view(num_frames, 4, 32, 32)

    # Create the Dataset
    #
    dataset = TensorDataset(features_tensor,labels_tensor)

    # Create a DataLoader (batch size 1 for now)
    #
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Define the CNN model as a sequence of layers
    #
    conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride = 2, padding=1)
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    fc_inputsize = 16 * 8 * 8
    fc1 = nn.Linear(fc_inputsize, 10) # (# of output layers * height * width, # of classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(conv1.parameters()) + list(fc1.parameters()))

    epochs = 2

    # Training loop
    for epoch in range(epochs):  # Loop over the dataset multiple times
        for inputs, labels in dataloader:
            # Forward pass
            x = conv1(inputs)  # Pass through convolutional layer
            x = pool(torch.relu(x))  # Activation and pooling
            x = x.view(-1, fc_inputsize)  # Flatten the output (-1: pytorch determines size based on data given)
            outputs = fc1(x)  # Pass through fully connected layer

            # Calculate loss function
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()   # Zero the gradients
            loss.backward()         # Backpropagation
            optimizer.step()        # Update weights

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    print("Finished training")



    # # Iterate through the DataLoader
    # #
    # for batch_features, batch_labels in dataloader:
    #     print("Batch Features:")
    #     print(batch_features)
    #     print("Batch Labels:")
    #     print(batch_labels)
    #     print()
import matplotlib.pyplot as plt
import numpy as np
import sys

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
from nedc_mladp_cnn_class import MladpTrainCNN
sys.path.append('/home/tul16619/SD1/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/src/functs/nedc_mladp_model_tools')
import nedc_mladp_model_tools as tools

def trainModel(data,labels):

    # Convert the labels and features to the correct types
    labels = tools.correctType(data,labels)

    # Create training model object
    my_model = MladpTrainCNN(data_features=data, data_labels=labels)

    # Reshape the tensor -- # Shape: [num_frames, 4 (channels), width, height]
    #
    num_frames = features_tensor.shape[0]
    num_pixels = features_tensor.shape[1] // 4
    features_tensor = features_tensor.view(num_frames, 4, 32, 32)

    # Create the Dataset
    #
    

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
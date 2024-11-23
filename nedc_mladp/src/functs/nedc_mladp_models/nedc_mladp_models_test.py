#!/usr/bin/env python
import os
import torch
import torch.nn
import torch.optim
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import ray
import math
import gc
gc.enable()

import nedc_image_tools
import nedc_file_tools
import nedc_dpath_ann_tools

# import project specific libraries
from nedc_mladp_label_enum import label_order
import nedc_mladp_fileio_tools as fileio_tools
import nedc_mladp_feats_tools as feats_tools

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

class ImageDataset(Dataset):

    @ray.remote
    def readFrames(self, annotation_file, image_file):

        # Get image dimensions
        image_reader = nedc_image_tools.Nil()
        image_reader.open(image_file)
        width, height = image_reader.get_dimension()

        # Get all frame top left coordinates
        top_left_coordinates = feats_tools.generateTopLeftFrameCoordinates(height, width, self.frame_size)

        # Get the labeled regions coordinates and cooresponding labels
        _, _, labels, region_coordinates = fileio_tools.parseAnnotations(annotation_file)

        # Generate the shapes
        labeled_regions = feats_tools.labeledRegions(region_coordinates)

        # Get the top left coordinates of the frames within labeled regions
        frame_top_left_coordinates,window_labels = feats_tools.classifyFrames(labels,
                                                                             height,
                                                                             width,
                                                                             self.window_size,
                                                                             self.frame_size,
                                                                             labeled_regions,
                                                                             self.overlap_threshold)

        # Get the RGB values of those windows
        window_RGBs = feats_tools.windowRGBValues(image_file, frame_top_left_coordinates, self.window_size)

        return (window_RGBs, window_labels)


    @ray.remote
    def countFrames(self, annotation_file, image_file):

        # Get image dimensions
        image_reader = nedc_image_tools.Nil()
        image_reader.open(image_file)
        width, height = image_reader.get_dimension()

        # Get all frame top left coordinates
        top_left_coordinates = feats_tools.generateTopLeftFrameCoordinates(height, width, self.frame_size)

        # Get the labeled regions coordinates and cooresponding labels
        _, _, labels, region_coordinates = fileio_tools.parseAnnotations(annotation_file)

        # Generate the shapes
        labeled_regions = feats_tools.labeledRegions(region_coordinates)

        # Get the top left coordinates of the frames within labeled regions
        _,window_labels = feats_tools.classifyFrames(labels,
                                                     height,
                                                     width,
                                                     self.window_size,
                                                     self.frame_size,
                                                     labeled_regions,
                                                     self.overlap_threshold)

        # return number of features in the image
        return len(window_labels)

    def __init__(self, image_files,
                 annotation_files,
                 frame_size:tuple,
                 window_size:tuple,
                 overlap_threshold,
                 cpus_per_batch=1,
                 gpus_per_batch=0,
                 memory=1,
                 image_cache_size=50,
                 feature_cache_threshold=1200):
        self.image_files = image_files
        self.annotation_files = annotation_files
        self.memory = memory*1024 * 1024 * 1024
        self.cpus_per_batch = cpus_per_batch
        self.gpus_per_batch = 0
        self.image_cache_size = image_cache_size
        self.current_file_index = 0
        self.frame_size = frame_size
        self.window_size = window_size
        self.overlap_threshold = overlap_threshold
        self.feature_cache_threshold = feature_cache_threshold
        # initialize ray 
        ray.init()

        # Count the number of labelled windows
        print(f"Counting labelled windows in {len(self.image_files)} images")
        ray_process = [self.countFrames.options(num_cpus=cpus_per_batch,
                                               num_gpus=gpus_per_batch,
                                               memory = memory).remote(self, annotation_file, image_file) for annotation_file, image_file in zip(self.annotation_files, self.image_files)]

        # Fill the cache with feature vectors
        print(f"Filling cache")
        self.current_process = [self.readFrames.options(num_cpus=cpus_per_batch,
                                                        num_gpus=gpus_per_batch,
                                                        memory = memory).remote(self, annotation_file, image_file) for annotation_file, image_file in zip(self.annotation_files[0:self.image_cache_size], self.image_files[0:image_cache_size])]

        # Get the number of features
        frames_per_image = ray.get(ray_process)
        self.data_length = sum(frames_per_image)
        del ray_process
        del frames_per_image
        print("Number of features = ",self.data_length)

        # Load the cache
        windows = ray.get(self.current_process)
        self.RGBs = []
        self.labels = []
        for RGBs, labels in windows:
            self.RGBs.extend(RGBs)
            self.labels.extend(labels)
        del windows
        del self.current_process
        print("Cached Filled")
        
        # Start the process for the next part of the cache
        self.current_process = [self.readFrames.options(num_cpus=cpus_per_batch,
                                                        num_gpus=gpus_per_batch,
                                                        memory = memory).remote(self, annotation_file, image_file) for annotation_file, image_file in zip(self.annotation_files[0:self.image_cache_size], self.image_files[0:image_cache_size])]
        
        # Set the current file index
        self.current_file_index += 2 * self.image_cache_size    


        

    def __len__(self):
        return self.data_length

    
    def __getitem__(self, idx):

        # return the RGB value and the label
        return_point = {'Data':self.RGBs.pop(0), 'Label':self.labels.pop(0)}

        # If almost out of features, start loading the next leg of the cahce
        if len(self.labels) <= self.feature_cache_threshold:

            # if it has ran through the file list, reset it and load the remaining files
            if self.current_file_index + self.image_cache_size >= len(self.image_files):
                to_add = len(self.image_files) - 1 - self.current_file_index
                self.current_file_index = 0

            # otherwise increment by the cache size
            else:
                to_add = self.image_cache_size
                self.current_file_index += self.image_cache_size

            # get the read values
            windows = ray.get(self.current_process)
            del self.current_process

            # start the next reading process
            self.current_process = [self.readFrames.options(num_cpus=self.cpus_per_batch,
                                                            num_gpus=self.gpus_per_batch,
                                                            memory=self.memory).remote(self, annotation_file, image_file) for annotation_file, image_file in zip(self.annotation_files[self.current_file_index:self.current_file_index+to_add], self.image_files[self.current_file_index:self.current_file_index+to_add])]

            # Append the values and clean up
            for RGBs, labels in windows:
                self.RGBs.extend(RGBs)
                self.labels.extend(labels)
            del windows
                
        return return_point



class CNN_2D(torch.nn.Module):

    def __init__(self, number_of_classes:int,
                 layer_01_convolution_output_channels:int=32,
                 layer_01_convolution_kernel_size:int=3,
                 layer_01_max_pooling_kernel_size:int=3,
                 layer_01_max_pooling_stride:int=1,
                 
                 
                 layer_02_convolution_output_channels:int=64,
                 layer_02_convolution_kernel_size:int=3,
                 layer_02_max_pooling_kernel_size:int=3,
                 layer_02_max_pooling_stride:int=1,
                 
                 layer_03_convolution_output_channels:int=128,
                 layer_03_convolution_kernel_size:int=3,
                 layer_03_max_pooling_kernel_size:int=3,
                 layer_03_max_pooling_stride:int=1,
                 dropout_coefficient:float=.5,
                 window_size:tuple = (256,256)):
    


        super(CNN_2D, self).__init__()
        
        ''' declare first convolutional layer
        3 input channels for R, G, and B
        '''
        self.convolution_01 = torch.nn.Conv2d(in_channels = 3,
                                              out_channels=layer_01_convolution_output_channels,
                                              kernel_size = layer_01_convolution_kernel_size)

        # Calculate shape based on kernel size
        shape = (window_size[0] + 1 - layer_01_convolution_kernel_size, window_size[1] + 1 - layer_01_convolution_kernel_size)

        # declare first max pooling layer
        self.max_pooling_01 = torch.nn.MaxPool2d(kernel_size = layer_01_max_pooling_kernel_size,
                                                 stride = layer_01_max_pooling_stride)

        # Calculate shape based on kernel size and stride
        max_pooling_0_steps = (shape[0] -  layer_01_max_pooling_kernel_size ) // layer_01_max_pooling_stride + 1
        max_pooling_1_steps = (shape[1] -  layer_01_max_pooling_kernel_size ) // layer_01_max_pooling_stride + 1
        shape = (max_pooling_0_steps, max_pooling_1_steps)
                
        # declare second convolutional layer
        self.convolution_02 = torch.nn.Conv2d(in_channels =layer_01_convolution_output_channels,
                                              out_channels=layer_02_convolution_output_channels,
                                              kernel_size = layer_02_convolution_kernel_size)

        # Calculate shape based on kernel size
        shape = (shape[0] + 1 - layer_02_convolution_kernel_size, shape[1] + 1 - layer_02_convolution_kernel_size)


        # declare second max pooling layer
        self.max_pooling_02 = torch.nn.MaxPool2d(kernel_size = layer_02_max_pooling_kernel_size,
                                                 stride = layer_02_max_pooling_stride)

        # Calculate shape based on kernel size and stride
        max_pooling_0_steps = (shape[0] -  layer_02_max_pooling_kernel_size ) // layer_02_max_pooling_stride + 1
        max_pooling_1_steps = (shape[1] -  layer_02_max_pooling_kernel_size ) // layer_02_max_pooling_stride + 1
        shape = (max_pooling_0_steps, max_pooling_1_steps)
        
        # declare third convolution layer
        self.convolution_03 = torch.nn.Conv2d(in_channels = layer_02_convolution_output_channels,
                                              out_channels=layer_03_convolution_output_channels,
                                              kernel_size = layer_03_convolution_kernel_size)

        # calculate shape based on kernel size
        shape = (shape[0] + 1 - layer_03_convolution_kernel_size, shape[1] + 1 - layer_03_convolution_kernel_size)

        # declare third max pooling layer
        self.max_pooling_03 = torch.nn.MaxPool2d(kernel_size = layer_03_max_pooling_kernel_size,
                                                 stride = layer_03_max_pooling_stride)

        # calculate shape based on kernel size and stride
        max_pooling_0_steps = (shape[0] -  layer_03_max_pooling_kernel_size ) // layer_03_max_pooling_stride + 1
        max_pooling_1_steps = (shape[1] -  layer_03_max_pooling_kernel_size ) // layer_03_max_pooling_stride + 1
        shape = (max_pooling_0_steps, max_pooling_1_steps)

        # Calculate fully_connected_input_size
        fully_connected_input_size = shape[0] * shape[1] * layer_03_convolution_output_channels

        # condense flattened vector down
        self.fully_connected_01 = torch.nn.Linear(fully_connected_input_size,
                                                  layer_03_convolution_output_channels)

        # dropout layer
        self.dropout_01 = torch.nn.Dropout(p = dropout_coefficient)

        # condense to class predictions
        self.fully_connected_02 = torch.nn.Linear(layer_03_convolution_output_channels,
                                                  number_of_classes)

        print("CNN Successfully Initialized")

    def forward(self, x):
        x = self.convolution_01(x)

        x = torch.relu(x)
        
        x = self.max_pooling_01(x)
        
        x = self.convolution_02(x)

        x = torch.relu(x)
        
        x = self.max_pooling_02(x)
        
        x = self.convolution_03(x)
        
        x = torch.relu(x)

        x = self.max_pooling_03(x)
        
        x = x.view(x.size(0), -1)

        x = self.fully_connected_01(x)

        x = torch.relu(x)

        x = self.dropout_01(x)

        x = self.fully_connected_02(x)

        return x

def frameCollate(batch):

    # create lists to hold data and labels
    labels = []
    data = []

    # normalize uint8 to float32
    transform = transforms.ToTensor()

    # iterate through batch fillings lists
    for sample in batch:
        data.append(transform(sample['Data']))
        labels.append(sample['Label'])

    # convert to tensors
    tensors = torch.stack(data)
    tensor_labels = torch.tensor([label_order[label].value for label in labels])
    return tensors, tensor_labels


def main():
    
    '''image_files = "/data/isip/exp/tuh_dpath/exp_0288/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/data/tuh_exp/train/train_images.list"
    annotation_files = "/data/isip/exp/tuh_dpath/exp_0288/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/data/tuh_exp/train/train_annotations.list"'''
    image_files = "/data/isip/exp/tuh_dpath/exp_0288/Machine-Learning-Applications-In-Digital-Pathology/ExampleImages.list"
    annotation_files = "/data/isip/exp/tuh_dpath/exp_0288/Machine-Learning-Applications-In-Digital-Pathology/ExampleAnnotations.list"

    # read the lists of files
    with open(image_files, "r") as f:
        image_files = [file.strip() for file in f.readlines()]
    with open(annotation_files, "r") as f:
        annotation_files = [file.strip() for file in f.readlines()]

    # declare parameters for genreating features
    frame_size = (128,128)
    window_size = (256,256)
    overlap_threshold = .5

    # declare CNN parameters
    epochs = 20
    learning_rate = .001
    batch_size = 300
    
    # Declare dataset
    dataset = ImageDataset(image_files, annotation_files, frame_size,
                           window_size, overlap_threshold, cpus_per_batch = .25,
                           memory = .5)

    # Initialize CNN
    CNN = CNN_2D(9)

    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If there is more than 1 GPU, parallelize
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        CNN = torch.nn.DataParallel(CNN)

    # Send the model to the appropriate device
    CNN = CNN.to(device)

    # Set the criteria and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(CNN.parameters(), lr = learning_rate)

    # Calculate the number of batches and initialize the dataloader
    number_of_batches = math.ceil(len(dataset)/batch_size)
    dataloader = DataLoader(dataset, batch_size = batch_size, collate_fn=frameCollate)

    # iterate through the epochs
    for epoch in range(epochs):
        batch_number = 1
        total_labels = 0
        labels_correct = 0

        # iterate through the batches
        for data, labels in dataloader:

            # send the data and labls to the proper device
            device_data = data.to(device)
            device_labels = labels.to(device)

            # get the class predictions
            outputs = CNN(device_data)

            # perform lsos and optimization
            loss = criterion(outputs,device_labels)
            loss.backward()
            optimizer.step()

            # track for epoch accuracy
            _, predicted_classes = torch.max(outputs,1)
            total = predicted_classes.size(0)
            correct = sum([(predicted == actual) for predicted,actual in zip(predicted_classes,device_labels)])
            labels_correct+=correct
            total_labels+=total

            # update on batch
            print(f"Batch {batch_number} of {number_of_batches}", flush=True)
            batch_number+=1

        # save the model
        torch.save(CNN, f"CNN{epoch}.pth")

        # print epoch information
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Accuracy: {labels_correct/total_labels}', flush=True)
    
if __name__ == "__main__":
    main()

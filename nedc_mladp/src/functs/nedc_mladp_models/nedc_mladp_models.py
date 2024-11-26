#!/usr/bin/env python
import os
import torch
import torch.nn
import torch.optim
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pandas
import mmap
import ray
import math
import json
import gc
gc.enable()

import nedc_image_tools
import nedc_file_tools
import nedc_dpath_ann_tools

# import project specific libraries
from nedc_mladp_label_enum import label_order
import nedc_mladp_fileio_tools as fileio_tools
import nedc_mladp_feats_tools as feats_tools



class CSVDataset(Dataset):

    
    def my_parse(self,csv_file):
        lines = [line.split(',') for line in fileio_tools.readLines(csv_file) if ':' not in line]
        dataframe = pandas.DataFrame(lines[1:],columns=lines[0])
        labels = dataframe['Label'].to_list()
        dataframe = dataframe.drop(['Label','TopLeftRow','TopLeftColumn'],axis=1)
        array = dataframe.to_numpy()
        array_columns = array.shape[1]
        return_point = {'Data':array[:,array_columns-self.PCA_count:array_columns].astype(np.float32),'Labels':labels}
        return return_point

    @ray.remote(num_cpus=1)
    def count_lines_mmap(self,file_number,filepath):
        with open(filepath, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), 0)
            lines = 0
            while mm.readline():
                lines += 1
            mm.close()
            print(file_number,"Completed")
            return lines - 10
    
    def __init__(self, csv_files, PCA_count=3782, transform = None):
        self.csv_files = csv_files
        self.PCA_count = PCA_count
        self.current_file = self.my_parse(csv_files[0])
        self.current_file_index = 0
        
        ray.init(ignore_reinit_error=True)
        print(f"Counting lines in {len(self.csv_files)}")
        lines = ray.get([self.count_lines_mmap.remote(self,i,csv_file) for i,csv_file in enumerate(self.csv_files)])
        self.data_length = sum(lines)
        ray.shutdown()
        print("Number of features = ",self.data_length)

    def __len__(self):
        return self.data_length

    
    def __getitem__(self, idx):
        return_point = {'Data':self.current_file['Data'][0,:], 'Label':self.current_file['Labels'].pop(0)}
        self.current_file['Data'] = np.delete(self.current_file['Data'],(0), axis = 0)
        if len(self.current_file['Labels']) == 0:
            self.current_file_index+=1
            if self.current_file_index == len(self.csv_files):
                self.current_file_index = 0
            self.current_file = self.my_parse(self.csv_files[self.current_file_index])
        return return_point


class CNN_1D(torch.nn.Module):

    def __init__(self, PCA_components:int, number_of_classes:int, model_output_path:str,
                 layer_01_convolution_output_channels:int=32,
                 layer_01_convolution_kernel_size:int=3,
                 layer_01_max_pooling_kernel_size:int=3,
                 
                 layer_02_convolution_output_channels:int=64,
                 layer_02_convolution_kernel_size:int=3,
                 layer_02_max_pooling_kernel_size:int=3,
                 
                 layer_03_convolution_output_channels:int=128,
                 layer_03_convolution_kernel_size:int=3,
                 layer_03_max_pooling_kernel_size:int=3,

                 dropout_coefficient:float=.5,
                 number_of_epochs:int=20, learning_rate:float=.001):

        super(convolutional_neural_network, self).__init__()
        
        self.getDevice()            
        self.PCA_components = PCA_components
        self.number_of_classes = number_of_classes
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs
        self.model_output_path = model_output_path

        self.convolution_01 = torch.nn.Conv1d(1,
                                              layer_01_convolution_output_channels,
                                              layer_01_convolution_kernel_size)

        self.max_pooling_01 = torch.nn.MaxPool1d(layer_01_max_pooling_kernel_size)

        max_pooling_01_output_size = (self.PCA_components + 1 - layer_01_convolution_kernel_size) // layer_01_max_pooling_kernel_size
        
        self.convolution_02 = torch.nn.Conv1d(layer_01_convolution_output_channels,
                                              layer_02_convolution_output_channels,
                                              layer_02_convolution_kernel_size)

        self.max_pooling_02 = torch.nn.MaxPool1d(layer_02_max_pooling_kernel_size)
        
        max_pooling_02_output_size = (max_pooling_01_output_size + 1 - layer_02_convolution_kernel_size) // layer_02_max_pooling_kernel_size
        
        self.convolution_03 = torch.nn.Conv1d(layer_02_convolution_output_channels,
                                              layer_03_convolution_output_channels,
                                              layer_03_convolution_kernel_size)

        self.max_pooling_03 = torch.nn.MaxPool1d(layer_03_max_pooling_kernel_size)

        fully_connected_input_size = (max_pooling_02_output_size + 1 - layer_03_convolution_kernel_size) // layer_03_max_pooling_kernel_size
        
        '''fully_connected_input_size = (self.PCA_components + 6
                                      - layer_01_convolution_kernel_size
                                      - layer_02_convolution_kernel_size
                                      - layer_03_convolution_kernel_size
                                      - layer_01_max_pooling_kernel_size
                                      - layer_02_max_pooling_kernel_size
                                      - layer_03_max_pooling_kernel_size)'''
        
        self.fully_connected_01 = torch.nn.Linear(layer_03_convolution_output_channels * fully_connected_input_size,
                                                  layer_03_convolution_output_channels)

        self.fully_connected_02 = torch.nn.Linear(layer_03_convolution_output_channels,
                                                  self.number_of_classes)

        self.dropout_01 = torch.nn.Dropout(p = dropout_coefficient)

        print("CNN Successfully Initialized")

    def my_collate(self, batch):
        labels = []
        data = []
        for sample in batch:
            data.append(sample['Data'])
            labels.append(sample['Label'])
        return np.vstack(data), labels 

        
    def forward(self, x):
        x = self.convolution_01(x)

        x = torch.relu(x)

        #print("Convolution 1", x.size())
        
        x = self.max_pooling_01(x)

        #print("Max Pooling 1", x.size())
        
        x = self.convolution_02(x)

        x = torch.relu(x)

        #print("Convolution 2", x.size())
        
        x = self.max_pooling_02(x)

        #print("Max Pooling 3", x.size())
        
        x = self.convolution_03(x)

        #print("Convolution 3", x.size())
        
        x = torch.relu(x)

        x = self.max_pooling_03(x)
        
        #print("Max Pooling 3", x.size())
        
        x = x.view(x.size(0), -1)

        x = self.fully_connected_01(x)

        x = torch.relu(x)

        x = self.dropout_01(x)

        x = self.fully_connected_02(x)

        return x

    def fit(self, list_of_files, batch_size = 4096):
        #if (self.device_type == "gpu"):
        #    data = data.to(self.device_type)
        #    labels = labels.to(self.device_type)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        print("Criterion and Optimizer initialized")

        dataset = CSVDataset(list_of_files, self.PCA_components)
        number_of_batches = math.ceil(len(dataset)/batch_size)
        dataloader = DataLoader(dataset, batch_size = batch_size, collate_fn=self.my_collate)
        
        for epoch in range(self.number_of_epochs):
            print(f"Processing epoch {epoch}")
            batch_number = 1
            for data, labels in dataloader:
                #batch_data_gpu = torch.tensor([tmp_data] for tmp_data in data_point['Data']).to(self.device_type)
                #batch_labels_gpu = labelToTensor(data_point['Label']).to(self.device_type)                                

                optimizer.zero_grad()

                
                outputs = self.forward(self.dataToTensor(data))
                loss = criterion(outputs, self.labelToTensor(labels))
                loss.backward()
                optimizer.step()
                print(f"Batch {batch_number} of {number_of_batches}", flush=True)
                batch_number+=1
            torch.save(self, self.model_output_path + f"CNN{epoch}.pth")
            print(f'Epoch {epoch+1}/{self.number_of_epochs}, Loss: {loss.item()}', flush=True)

    def getDevice(self):
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device_type == "cuda":
            self.cuda()
        print(f"Training utilizing {'GPU' if self.device_type == 'cuda' else 'CPU'}", flush=True)

        
    def predict(self, data):
        self.eval()
        tensor_features = self.dataToTensor(data).to(self.device_type)
        with torch.no_grad():
            predictions = self.forward(tensor_features)

        _, predicted_classes = torch.max(predictions,1)
        class_names = [label_order(label+1).name for label in predicted_classes.tolist()]

        return class_names

    def predict_proba(self, data):
        self.eval()
        tensor_features = self.dataToTensor(data).to(self.device_type)
        with torch.no_grad():
            predictions = self.forward(tensor_features).to(self.device_type)

            probabilities = torch.nn.functional.softmax(predictions, dim=1).tolist()
        
        return probabilities

    # takes in a list of windows where each window is represented by a list of PCs
    def dataToTensor(self, data):
        data = [[window] for window in data.tolist()]
        tensors = torch.tensor(data).to(self.device_type)
        return tensors

    # takes in a list of labels where each label is represented by a string
    def labelToTensor(self, labels):
        int_labels = [label_order[label].value-1 for label in labels]
        return torch.tensor(int_labels).to(self.device_type)




def CNN_1D_main():
    list_of_csv_files = "/data/isip/exp/tuh_dpath/exp_0288/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/data/tuh_exp/example_features_output/feature_files.list"
    with open(list_of_csv_files, "r") as f:
        files = [file.strip() for file in f.readlines()]


    CNN =convolutional_neural_network(3782, 9, "./models",
                                      1, 3, 1, 3, 1, 3)
    CNN = CNN.to(CNN.getDevice())
    
    CNN.fit(files, batch_size = 512)
    
class ImageDataset(Dataset):

    @ray.remote
    def readFrames(self, annotation_file, image_file, prediction_bool):

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

        if prediction_bool == True:
            self.frame_top_left_coordinates = frame_top_left_coordinates
            print("ON")
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
                 memory_per_batch=1,
                 image_cache_size=50,
                 feature_cache_threshold=1200,
                 prediction_bool=False,
                 object_store_memory:float=20):
        self.image_files = image_files
        self.annotation_files = annotation_files
        self.memory_per_batch = memory_per_batch
        self.cpus_per_batch = cpus_per_batch
        self.gpus_per_batch = 0
        self.image_cache_size = image_cache_size
        self.current_file_index = 0
        self.frame_size = frame_size
        self.window_size = window_size
        self.overlap_threshold = overlap_threshold
        self.feature_cache_threshold = feature_cache_threshold
        self.prediction_bool=prediction_bool
        self.frame_top_left_coordinates = None
        # initialize ray 
        ray.init(ignore_reinit_error=True, object_store_memory=object_store_memory*1024*1024*1024,
                 _system_config={
                     "object_spilling_config": json.dumps(
                         {"type": "filesystem", "params": {"directory_path": "./spill"}},
                     )
                 })

        # Count the number of labelled windows
        print(f"Counting labelled windows in {len(self.image_files)} images")
        ray_process = [self.countFrames.options(num_cpus=cpus_per_batch,
                                               num_gpus=gpus_per_batch,
                                                memory = memory_per_batch).remote(self, annotation_file, image_file) for annotation_file, image_file in zip(self.annotation_files, self.image_files)]

        # Fill the cache with feature vectors
        print(f"Filling cache")
        self.current_process = [self.readFrames.options(num_cpus=cpus_per_batch,
                                                        num_gpus=gpus_per_batch,
                                                        memory = memory_per_batch).remote(self, annotation_file, image_file, self.prediction_bool) for annotation_file, image_file in zip(self.annotation_files[0:self.image_cache_size], self.image_files[0:image_cache_size])]

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
                                                        memory = memory_per_batch).remote(self, annotation_file, image_file, self.prediction_bool) for annotation_file, image_file in zip(self.annotation_files[0:self.image_cache_size], self.image_files[0:image_cache_size])]
        
        # Set the current file index
        self.current_file_index += 2 * self.image_cache_size    


    def returnCoordinates(self):
        return self.frame_top_left_coordinates


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
                                                            memory=self.memory_per_batch).remote(self, annotation_file, image_file, self.prediction_bool) for annotation_file, image_file in zip(self.annotation_files[self.current_file_index:self.current_file_index+to_add], self.image_files[self.current_file_index:self.current_file_index+to_add])]

            # Append the values and clean up
            for RGBs, labels in windows:
                self.RGBs.extend(RGBs)
                self.labels.extend(labels)
            del windows
                
        return return_point



class CNN_2D_internal(torch.nn.Module):

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
                 window_size:tuple = (256,256),
                 object_store_memory:float = 20):
    


        super(CNN_2D_internal, self).__init__()
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "1"
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
    
class CNN_2D:
    
    def predict(self, image_file:str, annotation_file:str, frame_size:tuple, overlap_threshold:float):
        self.model.eval()
        # Declare dataset and dataloader
        print(image_file, annotation_file)
        dataset = ImageDataset([image_file], [annotation_file], frame_size,
                               self.window_size, overlap_threshold, cpus_per_batch = .25,
                               memory = .5)
        
        dataloader = DataLoader(dataset, batch_size = len(dataset), collate_fn=frameCollate)

        # iterate through the batches
        for data, labels in dataloader:
            
            # send the data and labls to the proper device
            device_data = data.to(self.device)
            device_labels = labels.to(self.device)
            
            # get the class predictions
            outputs = self.model(device_data)

            # track for epoch accuracy
            confidences, predicted_classes = torch.max(outputs,1)
            total = predicted_classes.size(0)
            correct = sum([(predicted == actual) for predicted,actual in zip(predicted_classes,device_labels)])
            labels_correct=correct
            total_labels=total
            print(confidences, predicted_classes)
    
    def getDevice(self):
        # Get the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # If there is more than 1 GPU, parallelize
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)

        # Send the model to the appropriate device
        self.model = self.model.to(self.device)
    
    def __init__(self, number_of_classes:int,
                 layer_01_convolution_output_channels:int,
                 layer_01_convolution_kernel_size:int,
                 layer_01_max_pooling_kernel_size:int,
                 layer_01_max_pooling_stride:int,
                 layer_02_convolution_output_channels:int,
                 layer_02_convolution_kernel_size:int,
                 layer_02_max_pooling_kernel_size:int,
                 layer_02_max_pooling_stride:int,
                 layer_03_convolution_output_channels:int,
                 layer_03_convolution_kernel_size:int,
                 layer_03_max_pooling_kernel_size:int,
                 layer_03_max_pooling_stride:int,
                 dropout_coefficient:float,
                 window_size:tuple):
        
        self.model = CNN_2D_internal(number_of_classes,
                 layer_01_convolution_output_channels,
                 layer_01_convolution_kernel_size,
                 layer_01_max_pooling_kernel_size,
                 layer_01_max_pooling_stride,
                 
                 
                 layer_02_convolution_output_channels,
                 layer_02_convolution_kernel_size,
                 layer_02_max_pooling_kernel_size,
                 layer_02_max_pooling_stride,
                 
                 layer_03_convolution_output_channels,
                 layer_03_convolution_kernel_size,
                 layer_03_max_pooling_kernel_size,
                 layer_03_max_pooling_stride,
                 dropout_coefficient,
                 window_size)

        self.getDevice()
        
        self.window_size = window_size
        


        
    
        
    def fit(self, image_files:list, annotation_files:list,
            epochs:int, learning_rate:float, batch_size:int,
            overlap_threshold:float, frame_size:tuple,
            output_directory:str,cpus_per_batch:float,
            memory_per_batch:float, image_cache_size:int,
            object_store_memory:float):

        # Check outupt directory
        if not (output_directory.endswith("/")):
            output_directory += "/"
        
        # read the lists of files
        with open(image_files, "r") as f:
            image_files = [file.strip() for file in f.readlines()]
        with open(annotation_files, "r") as f:
            annotation_files = [file.strip() for file in f.readlines()]
        
        # Declare dataset and dataloader
        dataset = ImageDataset(image_files, annotation_files, frame_size,
                               self.window_size, overlap_threshold,
                               cpus_per_batch = cpus_per_batch,
                               memory_per_batch = memory_per_batch,
                               image_cache_size = image_cache_size,
                               object_store_memory = object_store_memory,
                               feature_cache_threshold = 8 * batch_size)
        dataloader = DataLoader(dataset, batch_size = batch_size, collate_fn=frameCollate)


        # Set the criterion and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)

        # find the number of batches
        number_of_batches = math.ceil(len(dataset)/batch_size)

        # iterate through the epochs
        for epoch in range(epochs):
            batch_number = 1
            total_labels = 0
            labels_correct = 0
            
            # iterate through the batches
            for data, labels in dataloader:

                # send the data and labls to the proper device
                device_data = data.to(self.device)
                device_labels = labels.to(self.device)

                # get the class predictions
                outputs = self.model(device_data)

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
            torch.save(self.model, f"{output_directory}CNN_2D{epoch}.pth")

            # print epoch information
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Accuracy: {labels_correct/total_labels}', flush=True)

def frameCollateResize(batch):

    # create lists to hold data and labels
    labels = []
    data = []
    
    # normalize uint8 to float32
    transform_1 = transforms.ToTensor()
    transform_2 = transforms.Resize((128,128))
    # iterate through batch fillings lists
    for sample in batch:
        data.append(transform_2(transform_1(sample['Data'])))
        labels.append(sample['Label'])
        
    # convert to tensors
    tensors = torch.stack(data)
    tensor_labels = torch.tensor([label_order[label].value for label in labels])
    return tensors, tensor_labels

class CNN_2D_internal_claudia(torch.nn.Module):

    def __init__(self, number_of_classes:int,
                 
                 layer_01_convolution_output_channels:int=32,
                 layer_01_convolution_kernel_size:int=3,
                 layer_01_max_pooling_kernel_size:int=3,
                 layer_01_max_pooling_stride:int=1,
                 
                 
                 layer_02_convolution_output_channels:int=64,
                 layer_02_convolution_kernel_size:int=3,
                 layer_02_max_pooling_kernel_size:int=3,
                 layer_02_max_pooling_stride:int=1,
                 dropout_coefficient:float=.5,                 
                 window_size:tuple = (256,256)):

    


        super(CNN_2D_internal_claudia, self).__init__()
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "1"
        ''' declare first convolutional layer
        3 input channels for R, G, and B
        '''
        self.convolution_01 = torch.nn.Conv2d(in_channels = 3,
                                              out_channels=layer_01_convolution_output_channels,
                                              kernel_size = layer_01_convolution_kernel_size)

        # Calculate shape based on kernel size
        shape = (128 + 1 - layer_01_convolution_kernel_size, 128 + 1 - layer_01_convolution_kernel_size)

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
        
        # Calculate fully_connected_input_size
        fully_connected_input_size = shape[0] * shape[1] * layer_02_convolution_output_channels

        # condense flattened vector down
        self.fully_connected_01 = torch.nn.Linear(fully_connected_input_size,
                                                  layer_02_convolution_output_channels)

        # dropout layer
        self.dropout_01 = torch.nn.Dropout(p = dropout_coefficient)

        # condense to class predictions
        self.fully_connected_02 = torch.nn.Linear(layer_02_convolution_output_channels,
                                                  number_of_classes)

        # sofmax layer
        self.softmax = torch.nn.Softmax(dim=1)
        
        
        print("CNN Successfully Initialized")

    def forward(self, x):

        x = self.convolution_01(x)

        x = torch.relu(x)
        
        x = self.max_pooling_01(x)
        
        x = self.convolution_02(x)

        x = torch.relu(x)
        
        x = self.max_pooling_02(x)
                
        x = x.view(x.size(0), -1)

        x = self.fully_connected_01(x)

        x = torch.relu(x)

        x = self.dropout_01(x)

        x = self.fully_connected_02(x)

        x = self.softmax(x)
        
        return x



class CNN_2D_claudia:
    
    def predict(self, image_file:str, annotation_file:str, frame_size:tuple, overlap_threshold:float):
        self.model.eval()
        # Declare dataset and dataloader
        print(image_file, annotation_file)
        dataset = ImageDataset([image_file], [annotation_file], frame_size,
                               self.window_size, overlap_threshold, cpus_per_batch = .25,
                               memory = .5)
        
        dataloader = DataLoader(dataset, batch_size = len(dataset), collate_fn=frameCollateResize)

        # iterate through the batches
        for data, labels in dataloader:
            
            # send the data and labls to the proper device
            device_data = data.to(self.device)
            device_labels = labels.to(self.device)
            
            # get the class predictions
            outputs = self.model(device_data)

            # track for epoch accuracy
            confidences, predicted_classes = torch.max(outputs,1)
            total = predicted_classes.size(0)
            correct = sum([(predicted == actual) for predicted,actual in zip(predicted_classes,device_labels)])
            labels_correct=correct
            total_labels=total
            print(confidences, predicted_classes)
    
    def getDevice(self):
        # Get the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # If there is more than 1 GPU, parallelize
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)

        # Send the model to the appropriate device
        self.model = self.model.to(self.device)
    
    def __init__(self, number_of_classes:int,
                 layer_01_convolution_output_channels:int,
                 layer_01_convolution_kernel_size:int,
                 layer_01_max_pooling_kernel_size:int,
                 layer_01_max_pooling_stride:int,
                 layer_02_convolution_output_channels:int,
                 layer_02_convolution_kernel_size:int,
                 layer_02_max_pooling_kernel_size:int,
                 layer_02_max_pooling_stride:int,
                 dropout_coefficient:float,
                 window_size:tuple,
                 load_model_path:str=None):
        if load_model_path != None:
            self.model = torch.load(load_model_path)
            print("Model Loaded",flush=True)
        else:
            self.model = CNN_2D_internal_claudia(number_of_classes,
                                                 layer_01_convolution_output_channels,
                                                 layer_01_convolution_kernel_size,
                                                 layer_01_max_pooling_kernel_size,
                                                 layer_01_max_pooling_stride,
                                                 
                                                 
                                                 layer_02_convolution_output_channels,
                                                 layer_02_convolution_kernel_size,
                                                 layer_02_max_pooling_kernel_size,
                                                 layer_02_max_pooling_stride,
                                                 
                                                 dropout_coefficient,
                                                 window_size)

        self.getDevice()
        
        self.window_size = window_size
        


        
    
        
    def fit(self, image_files:list, annotation_files:list,
            epochs:int, learning_rate:float, batch_size:int,
            overlap_threshold:float, frame_size:tuple,
            output_directory:str,cpus_per_batch:float,
            memory_per_batch:float, image_cache_size:int,
            object_store_memory:float):

        # Check outupt directory
        if not (output_directory.endswith("/")):
            output_directory += "/"
        
        # read the lists of files
        with open(image_files, "r") as f:
            image_files = [file.strip() for file in f.readlines()]
        with open(annotation_files, "r") as f:
            annotation_files = [file.strip() for file in f.readlines()]
        
        # Declare dataset and dataloader
        dataset = ImageDataset(image_files, annotation_files, frame_size,
                               self.window_size, overlap_threshold,
                               cpus_per_batch = cpus_per_batch,
                               memory_per_batch = memory_per_batch,
                               image_cache_size = image_cache_size,
                               object_store_memory = object_store_memory)
        dataloader = DataLoader(dataset, batch_size = batch_size, collate_fn=frameCollateResize)


        # Set the criterion and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        
        # find the number of batches
        number_of_batches = math.ceil(len(dataset)/batch_size)

        # iterate through the epochs
        for epoch in range(epochs):
            batch_number = 1
            total_labels = 0
            labels_correct = 0
            epoch_loss = 0
            # iterate through the batches
            for data, labels in dataloader:

                # send the data and labls to the proper device
                device_data = data.to(self.device)
                device_labels = labels.to(self.device)

                # get the class predictions
                outputs = self.model(device_data)

                # perform lsos and optimization
                loss = criterion(outputs,device_labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
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
            torch.save(self.model, f"{output_directory}CNN_2D{epoch}.pth")

            # print epoch information
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/number_of_batches}, Accuracy: {labels_correct/total_labels}', flush=True)

            
def CNN_2D_main():
    
    '''image_files = "/data/isip/exp/tuh_dpath/exp_0288/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/data/tuh_exp/train/train_images.list"
    annotation_files = "/data/isip/exp/tuh_dpath/exp_0288/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/data/tuh_exp/train/train_annotations.list"'''
    image_files = "/data/isip/exp/tuh_dpath/exp_0288/Machine-Learning-Applications-In-Digital-Pathology/ExampleImages.list"
    annotation_files = "/data/isip/exp/tuh_dpath/exp_0288/Machine-Learning-Applications-In-Digital-Pathology/ExampleAnnotations.list"

    # declare parameters for genreating features
    number_of_classes = 9
    layer_01_convolution_output_channels = 32
    layer_01_convolution_kernel_size = 3
    layer_01_max_pooling_kernel_size = 3
    layer_01_max_pooling_stride = 1
    layer_02_convolution_output_channels = 64
    layer_02_convolution_kernel_size = 3
    layer_02_max_pooling_kernel_size = 3
    layer_02_max_pooling_stride = 1
    dropout_coefficient = 0.5
    overlap_threshold = .5
    frame_size = (128,128)
    window_size = (256,256)
    number_of_epochs = 20
    learning_rate = .001
    batch_size = 300
    output_directory = "./"
    
    model = CNN_2D_claudia(number_of_classes,
                           layer_01_convolution_output_channels,
                           layer_01_convolution_kernel_size,
                           layer_01_max_pooling_kernel_size,
                           layer_01_max_pooling_stride,
                           layer_02_convolution_output_channels,
                           layer_02_convolution_kernel_size,
                           layer_02_max_pooling_kernel_size,
                           layer_02_max_pooling_stride,    
                           dropout_coefficient,
                           window_size)
    
    model.fit(image_files, annotation_files,
              number_of_epochs, learning_rate, batch_size,
              overlap_threshold, frame_size, output_directory,
              .5,.5, 10)
    
    
if __name__ == "__main__":
    CNN_2D_main()

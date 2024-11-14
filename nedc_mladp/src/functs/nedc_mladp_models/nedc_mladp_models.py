#!/usr/bin/env python
import torch
import torch.nn
import torch.optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pandas
import mmap


import gc
gc.enable()

# import project specific libraries
from nedc_mladp_label_enum import label_order
import nedc_mladp_fileio_tools as fileio_tools



class CSVDataset(Dataset):

    
    def my_parse(self,csv_file):
        lines = [line.split(',') for line in fileio_tools.readLines(csv_file) if ':' not in line]
        dataframe = pandas.DataFrame(lines[1:],columns=lines[0])
        labels = dataframe['Label'].to_list()
        dataframe = dataframe.drop(['Label','TopLeftRow','TopLeftColumn'],axis=1)
        return_point = {'Data':dataframe.to_numpy().astype(np.float32),'Labels':labels}
        return return_point
    
    def count_lines_mmap(self,filepath):
        with open(filepath, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), 0)
            lines = 0
            while mm.readline():
                lines += 1
            mm.close()
            return lines
    
    def __init__(self, csv_files, transform = None):
        self.csv_files = csv_files
        self.current_file = self.my_parse(csv_files[0])
        self.data_length = 0
        self.current_file_index = 0
        for i,csv_file in enumerate(csv_files):
            self.data_length+=self.count_lines_mmap(csv_file) - 10
            print(self.data_length)
            print(f"{i} of {len(csv_files)} files lines counted")
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


class convolutional_neural_network(torch.nn.Module):

    def __init__(self, PCA_components:int, number_of_classes:int, model_output_path:str,
                 layer_01_output_channels:int=32, layer_01_kernel_size:int=3,
                 layer_02_output_channels:int=64, layer_02_kernel_size:int=3,
                 layer_03_output_channels:int=128, layer_03_kernel_size:int=3,
                 dropout_coefficient:float=.5, number_of_epochs:int=20,
                 learning_rate:float=.001):

        super(convolutional_neural_network, self).__init__()
        
        self.getDevice()            
        self.PCA_components = PCA_components
        self.number_of_classes = number_of_classes
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs
        self.model_output_path = model_output_path
        
        self.convolution_01 = torch.nn.Conv1d(1,
                                              layer_01_output_channels,
                                              layer_01_kernel_size)

        self.convolution_02 = torch.nn.Conv1d(layer_01_output_channels,
                                              layer_02_output_channels,
                                              layer_02_kernel_size)

        self.convolution_03 = torch.nn.Conv1d(layer_02_output_channels,
                                              layer_03_output_channels,
                                              layer_03_kernel_size)

        fully_connected_input_size = (self.PCA_components + 3
                                      - layer_01_kernel_size
                                      - layer_02_kernel_size
                                      - layer_03_kernel_size)
                                      
        self.fully_connected_01 = torch.nn.Linear(layer_03_output_channels * fully_connected_input_size,
                                                  layer_03_output_channels)

        self.fully_connected_02 = torch.nn.Linear(layer_03_output_channels,
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

        x = self.convolution_02(x)

        x = torch.relu(x)

        x = self.convolution_03(x)

        
        x = torch.relu(x)

        x = x.view(x.size(0), -1)

        x = self.fully_connected_01(x)

        x = torch.relu(x)

        x = self.dropout_01(x)

        x = self.fully_connected_02(x)

        return x

    def fit(self, list_of_files):
        #if (self.device_type == "gpu"):
        #    data = data.to(self.device_type)
        #    labels = labels.to(self.device_type)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        print("Criterion and Optimizer initialized")

        dataloader = DataLoader(CSVDataset(list_of_files), batch_size = 64,collate_fn=self.my_collate)
        
        for epoch in range(self.number_of_epochs):
            print(f"Processing epoch {epoch}")
            batch_number = 1
            for data, labels in dataloader:
                #batch_data_gpu = torch.tensor([tmp_data] for tmp_data in data_point['Data']).to(self.device_type)
                #batch_labels_gpu = labelToTensor(data_point['Label']).to(self.device_type)                                

                print(f"Processing batch {batch_number} of {batch_size}")
                batch_number+=1
                optimizer.zero_grad()

                
                outputs = self.forward(self.dataToTensor(data))
                loss = criterion(outputs, self.labelToTensor(labels))
                loss.backward()
                optimizer.step()
                
                torch.save(self, self.model_output_path + f"CNN{epoch}.pth")
                print(f'Epoch {epoch+1}/{self.number_of_epochs}, Loss: {loss.item()}')

    def getDevice(self):
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training utilizing {'GPU' if self.device_type == 'cuda' else 'CPU'}", flush=True)
        self.to(self.device_type)

        
    def predict(self, data):
        self.eval()
        tensor_features = self.dataToTensor(data)
        with torch.no_grad():
            predictions = self.forward(tensor_features)

        _, predicted_classes = torch.max(predictions,1)
        class_names = [label_order(label).name for label in predicted_classes.tolist()]

        return class_names

    def predict_proba(self, data):
        self.eval()
        tensor_features = self.dataToTensor(data)
        with torch.no_grad():
            predictions = self.forward(tensor_features)

        probabilities = torch.nn.functional.softmax(predictions, dim=1)

        return probabilities

    # takes in a list of windows where each window is represented by a list of PCs
    def dataToTensor(self, data):
        data = [[window] for window in data.tolist()]
        tensors = torch.tensor(data)
        return tensors

    # takes in a list of labels where each label is represented by a string
    def labelToTensor(self, labels):
        int_labels = [label_order[label].value-1 for label in labels]
        return torch.tensor(int_labels)

    def createDataloader(self, data, labels):
        data_set = TensorDataset(data,labels)
        data_loader = DataLoader(data_set, batch_size=100)
        return data_loader
    
def main():
    list_of_csv_files = "/data/isip/exp/tuh_dpath/exp_0288/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/data/tuh_exp/example_features_output/feature_files.list"
    with open(list_of_csv_files, "r") as f:
        files = [file.strip() for file in f.readlines()]


    CNN =convolutional_neural_network(3782, 9, "./models",
                                      1, 3, 1, 3, 1, 3)
    CNN.fit(files)
        
    
if __name__ == "__main__":
    main()

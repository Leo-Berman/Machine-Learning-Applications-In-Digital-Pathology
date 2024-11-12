import torch
import torch.nn
import torch.optim
import pandas as pd
import numpy as np

# import project specific libraries
from nedc_mladp_label_enum import label_order


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

    def fit(self, data, labels):
        if (self.device_type == "gpu"):
            data = data.to(self.device_type)
            labels = labels.to(self.device_type)
        tensor_features = self.dataToTensor(data)
        tensor_labels = self.labelToTensor(labels)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        
        for epoch in range(self.number_of_epochs):
            
            optimizer.zero_grad()
            outputs = self.forward(tensor_features)
            loss = criterion(outputs, tensor_labels)
            
            loss.backward()
            optimizer.step()
            torch.save(self, self.model_output_path + f"CNN{epoch}.pth")
            print(f'Epoch {epoch+1}/{self.number_of_epochs}, Loss: {loss.item()}')

    def getDevice(self):
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training utilizing {'GPU' if self.device_type == 'cuda' else 'CPU'}")
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
    
        
if __name__ == "__main__":
    main()

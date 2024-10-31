import sys
import torch
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils
import numpy as np

sys.path.append('/home/tul16619/SD1/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/src/functs/nedc_mladp_model_tools')
import nedc_mladp_model_tools as tools


def load_data(filelist):

    # tools.randomData(50)

    totaldata = tools.parsePCA(filelist)
    # totaldata = tools.parsePCA_file(filelist)
    # print(totaldata)

    labels = totaldata[:,0]
    
    data = totaldata[:,1:]

    # print(np.shape(labels))
    # print(np.shape(data))

    # labels[-1] = 'unlab'
    print(labels)

    data, labels = tools.correctType(data,labels)

    print(labels)

    feats_tensor = torch.tensor(data, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    # dataset = utils.data.TensorDataset(feats_tensor,label_tensor)

    print(feats_tensor.shape)
    print(label_tensor.shape)

    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # model = torch.load("/data/isip/exp/tuh_dpath/exp_0286/v2.0.0/models/model.pth")

    # # print(model)

    # # Update the first convolutional layer for a 1-channel input and a smaller kernel
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    # # Update the maxpool to handle the smaller input dimension
    # model.maxpool = nn.Identity()

    # # Verify the change
    # print(model)

    # # Adjust your data to match the required input shape
    # # Example dummy data
    # # data = torch.randn(1045, 40)  # Your (1045, 40) data

    # # Reshape each data point to (1, 8, 5) and prepare for the model
    # feats_tensor = feats_tensor.reshape(-1, 1, 8, 5)  # (batch_size, channels, height, width)

    # # data = utils.data.TensorDataset(feats_tensor,label_tensor)

    # # feats_tensor = feats_tensor.to(device)

    # # Split data into training and validation sets (e.g., 80-20 split)
    # train_size = int(0.8 * len(feats_tensor))
    # val_size = len(feats_tensor) - train_size
    # train_data, val_data = utils.data.random_split(TensorDataset(feats_tensor, label_tensor), [train_size, val_size])

    # # Data loaders
    # train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    # # Move model to the GPU if available
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = model.to(device)

    # # Define the loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # Training loop
    # num_epochs = 2
    # for epoch in range(num_epochs):
    #     model.train()
    #     running_loss = 0.0
    #     for inputs, labels in train_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)

    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         print(np.shape(outputs))
    #         print(np.shape(labels))
    #         # exit(100)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item() * inputs.size(0)

    #     epoch_loss = running_loss / len(train_loader.dataset)
    #     print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    #     # Validation
    #     model.eval()
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for inputs, labels in val_loader:
    #             inputs, labels = inputs.to(device), labels.to(device)
    #             outputs = model(inputs)
    #             _, predicted = torch.max(outputs, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()
        
    #     accuracy = 100 * correct / total
    #     print(f'Validation Accuracy: {accuracy:.2f}%')

    # print("Training complete.")

load_data("/data/isip/exp/tuh_dpath/exp_0288/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/data/example_output/features/feature_files.list")
# load_data("/data/isip/exp/tuh_dpath/exp_0288/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/data/example_output/features/00477780_aaaaaagg_s000_0hne_0000_a001_lvl001_t000_FEATS.csv")

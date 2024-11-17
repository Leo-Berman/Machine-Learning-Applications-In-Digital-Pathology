import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler
from torch import optim

feats = np.load('feats.npy')
print("feats: ", np.shape(feats))
labels = np.load('labels.npy')
print("labels: ", np.shape(labels))


num_ftrs = labels.max() # encoding starts from 1

# Adjust labels to start from 0
labels -= 1


# Load model
model = torch.load('../model.pth')

model.fc = nn.Linear(512, num_ftrs) # number of input and output features

# Splitting the data
# TODO: Change it with the actual subsets, with paths from a list file
feats_train, feats_dev, labels_train, labels_dev = train_test_split(
    feats, labels, test_size=0.2, random_state=42, stratify=labels
)
#divide_number = 38
divide_number = 1

# =======================================================================
# ====================== Taking care of data ============================
# =======================================================================

feats_train_samples = feats_train.shape[0]//divide_number*divide_number
feats_dev_samples = feats_dev.shape[0]//divide_number*divide_number
labels_train_samples = labels_train.shape[0]//divide_number*divide_number
labels_dev_samples = labels_dev.shape[0]//divide_number*divide_number

feats_train = feats_train[:feats_train_samples,:].reshape(-1,1,40,divide_number)
feats_dev = feats_dev[:feats_dev_samples,:].reshape(-1,1,40,divide_number)

# Convert to tensors
feats_train_tensor = torch.tensor(feats_train, dtype=torch.float32).repeat(1,3,1,1)
labels_train_tensor = torch.tensor(labels_train, dtype=torch.long)
feats_dev_tensor = torch.tensor(feats_dev, dtype=torch.float32).repeat(1,3,1,1)
labels_dev_tensor = torch.tensor(labels_dev, dtype=torch.long)

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(feats_train_tensor, labels_train_tensor)
dev_dataset = TensorDataset(feats_dev_tensor, labels_dev_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

# Calculate class weights for training set
train_unique, train_counts = np.unique(labels_train, return_counts=True)
train_class_counts = torch.tensor(train_counts, dtype=torch.float32)
train_class_weights = max(train_class_counts) / train_class_counts
train_class_weights = train_class_weights / train_class_weights.sum()  # Normalization

# Calculate class weights for validation set
dev_unique, dev_counts = np.unique(labels_dev, return_counts=True)
dev_class_counts = torch.tensor(dev_counts, dtype=torch.float32)
dev_class_weights = max(dev_class_counts) / dev_class_counts
dev_class_weights = dev_class_weights / dev_class_weights.sum()  # Normalization

# =======================================================================
# ====================== Taking care of model ===========================
# =======================================================================


# Define loss functions with respective class weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_criterion = nn.CrossEntropyLoss(weight=train_class_weights.to(device))
dev_criterion = nn.CrossEntropyLoss(weight=dev_class_weights.to(device))

# Define optimizer
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Define learning rate scheduler
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#import time
import copy
#import torch

def simple_train_model(model, train_loader, dev_loader, train_criterion, dev_criterion, optimizer, scheduler, device, num_epochs):
    # Track best model weights and performance
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')

    # Start the training process
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Set model to training mode
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Iterate over training data
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()
            print(inputs.shape)
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            #preds = copy.copy(outputs)
            print(outputs.shape)
            print("Outputs",outputs)
            print(labels.shape)
            print("Labels",labels)
            loss = train_criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            # TODO: Make it prettier to see (I used 222222).
            print("2222222222",preds)
            print("2222222222",labels.data)
            running_corrects += torch.sum(preds == labels.data)

        # Step the scheduler
        scheduler.step()

        # Calculate training loss and accuracy
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        # Iterate over validation data
        for inputs, labels in dev_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():  # No need to track gradients for validation
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                #preds = copy.copy(outputs)
                loss = dev_criterion(outputs, labels)

            # Track statistics
            running_loss += loss.item() * inputs.size(0)

            running_corrects += torch.sum(preds == labels.data)

        # Calculate validation loss and accuracy
        epoch_loss = running_loss / len(dev_loader.dataset)
        epoch_acc = running_corrects.double() / len(dev_loader.dataset)
        print(f'Dev Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Deep copy the model if it is the best one so far
        if epoch_loss < best_loss:
            best_acc = epoch_acc
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    # Load best model weights
    model.load_state_dict(best_model_wts)

num_epochs = 1
# TODO: give the model to the CPU/GPU (check what's available)
model.to(device) 
# TODO: get rid of the prints, I added them so that you also see 
model = simple_train_model(model,
                           train_loader,
                           dev_loader,
                           train_criterion,
                           dev_criterion,
                           optimizer_ft,
                           exp_lr_scheduler,
                           device,
                           num_epochs)






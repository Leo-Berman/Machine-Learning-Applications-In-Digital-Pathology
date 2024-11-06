import sys
import copy
import numpy as np

from sklearn.model_selection import train_test_split

# PYTORCH LIBRARY
import torch
from torch import optim, utils, nn
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader

# IMPORT LOCAL LIBRARIES
sys.path.append('/home/tul16619/SD1/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/src/functs/nedc_mladp_model_tools')
import nedc_mladp_model_tools as tools


# Load model
model = torch.load('model.pth')

class modelCNN:
    def __init__(self, model_path, num_epochs, batch_size, device):
        
        self.model_path = model_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def input_data(self, filelist):
        # get the total data from CSV file(s)
        totaldata = tools.parsePCA(filelist)          # filelist
        
        # separate the labels and features
        labels = totaldata[:,0]
        feats = totaldata[:,1:]
        feats, labels = tools.correctType(feats,labels)

        # create the tensors
        feats_tensor = torch.tensor(feats, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.long)

        self.feats_train, self.feats_dev, self.labels_train, self.labels_dev = train_test_split(
            feats, labels, test_size=0.2, random_state=42, stratify=labels
        )

    def model_prep(self):

        divide_num = 1

        # Decide number of samples for train data and development data
        
        feats_train_samples = self.feats_train.shape[0]//divide_num*divide_num
        feats_dev_samples = self.feats_dev.shape[0]//divide_num*divide_num
        labels_train_samples = self.labels_train.shape[0]//divide_num*divide_num
        labels_dev_samples = self.labels_dev.shape[0]//divide_num*divide_num

        self.feats_train = self.feats_train[:feats_train_samples,:].reshape(-1,1,40,divide_num)
        self.feats_dev = self.feats_dev[:feats_dev_samples,:].reshape(-1,1,40,divide_num)

        # Convert to tensors
        #
        feats_train_tensor = torch.tensor(self.feats_train, dtype=torch.float32).repeat(1,3,1,1)
        labels_train_tensor = torch.tensor(self.labels_train, dtype=torch.long)
        feats_dev_tensor = torch.tensor(self.feats_dev, dtype=torch.float32).repeat(1,3,1,1)
        labels_dev_tensor = torch.tensor(self.labels_dev, dtype=torch.long)

        # Create TensorDatasets and DataLoaders
        #
        train_dataset = TensorDataset(feats_train_tensor, labels_train_tensor)
        dev_dataset = TensorDataset(feats_dev_tensor, labels_dev_tensor)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True)
        self.dev_loader = DataLoader(dev_dataset, batch_size=self.batchsize, shuffle=False)

        # Calculate class weights for training set
        #
        train_unique, train_counts = np.unique(self.labels_train, return_counts=True)
        train_class_counts = torch.tensor(train_counts, dtype=torch.float32)
        train_class_weights = max(train_class_counts) / train_class_counts
        train_class_weights = train_class_weights / train_class_weights.sum()  # Normalization

        # Calculate class weights for validation set
        #
        dev_unique, dev_counts = np.unique(self.labels_dev, return_counts=True)
        dev_class_counts = torch.tensor(dev_counts, dtype=torch.float32)
        dev_class_weights = max(dev_class_counts) / dev_class_counts
        dev_class_weights = dev_class_weights / dev_class_weights.sum()  # Normalization

        # Define loss functions with respective class weights
        self.train_criterion = nn.CrossEntropyLoss(weight=train_class_weights.to(self.device))
        self.dev_criterion = nn.CrossEntropyLoss(weight=dev_class_weights.to(self.device))

        # Define optimizer
        self.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # Define learning rate scheduler
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def simple_train_model(self):
        # Track best model weights and performance
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_loss = float('inf')

        # Start the training process
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch + 1}/{self.num_epochs}')
            print('-' * 10)

            # Set model to training mode
            model.train()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over training data
            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Zero the gradients
                self.optimizer.zero_grad()
                print(inputs.shape)
                
                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                print(outputs.shape)
                print("Outputs",outputs)
                print(labels.shape)
                print("Labels",labels)
                loss = self.train_criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # Track statistics
                running_loss += loss.item() * inputs.size(0)
                # TODO: Make it prettier to see (I used 222222).
                print("2222222222",preds)
                print("2222222222",labels.data)
                running_corrects += torch.sum(preds == labels.data)

            # Step the scheduler
            self.scheduler.step()

            # Calculate training loss and accuracy
            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = running_corrects.double() / len(self.train_loader.dataset)
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Validation phase
            model.eval()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over validation data
            for inputs, labels in self.dev_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():  # No need to track gradients for validation
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    #preds = copy.copy(outputs)
                    loss = self.dev_criterion(outputs, labels)

                # Track statistics
                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)

            # Calculate validation loss and accuracy
            epoch_loss = running_loss / len(self.dev_loader.dataset)
            epoch_acc = running_corrects.double() / len(self.dev_loader.dataset)
            print(f'Dev Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it is the best one so far
            if epoch_loss < best_loss:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        # Load best model weights
        model.load_state_dict(best_model_wts)
    
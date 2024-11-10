import sys
import copy
import numpy as np
import time

from sklearn.model_selection import train_test_split

# PYTORCH LIBRARY
import torch
from torch import optim, utils, nn
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader

# IMPORT LOCAL LIBRARIES
sys.path.append('/home/tul16619/SD1/Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/src/functs/nedc_mladp_model_tools')
import nedc_mladp_model_tools as tools


class modelCNN:
    def __init__(self, num_epochs, batch_size, num_cls):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_cls = num_cls

    def input_data(self, filelist):
        # get the to        tal data from CSV file(s)
        totaldata = tools.parsePCA(filelist)          # filelist
        
        # separate the labels and features
        labels = totaldata[:,0]
        feats = totaldata[:,1:]
        feats, labels = tools.correctType(feats,labels)

        # create the tensors
        feats_tensor = torch.tensor(feats, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.long)

        # labels contain digits [1-9]
        num_cls = len(set(label_tensor.tolist()))
        # decrement the labels by 1 to contain [0:8]
        label_tensor -= 1

        return feats_tensor, label_tensor, num_cls

    def prep_model(self, feats_train, labels_train, feats_dev, labels_dev):

        self.feats_train = feats_train
        self.labels_train = labels_train
        self.feats_dev = feats_dev
        self.labels_dev = labels_dev

        # Decide number of samples for train data and development data
        reshaped_feats_train = self.feats_train[:self.feats_train.shape[0],:].reshape(-1,1,self.feats_train.shape[1],1)
        reshaped_feats_dev = self.feats_dev[:self.feats_dev.shape[0],:].reshape(-1,1,self.feats_dev.shape[1],1)

        # Convert to tensors
        #
        feats_train_tensor = reshaped_feats_train.clone().detach().to(torch.float32).repeat(1, 3, 1, 1)
        labels_train_tensor = self.labels_train.clone().detach().to(torch.long)
        feats_dev_tensor = reshaped_feats_dev.clone().detach().to(torch.float32).repeat(1, 3, 1, 1)
        labels_dev_tensor = self.labels_dev.clone().detach().to(torch.long)

        # Create TensorDatasets and DataLoaders
        #
        train_dataset = TensorDataset(feats_train_tensor, labels_train_tensor)
        dev_dataset = TensorDataset(feats_dev_tensor, labels_dev_tensor)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False)

        # Calculate class weights for training set
        #
        train_unique, train_counts = np.unique(labels_train, return_counts=True)
        self.train_unique, train_counts = tools.fillClasses(train_unique, train_counts)
        train_class_counts = torch.tensor(train_counts, dtype=torch.float32)
        train_class_weights = max(train_class_counts) / train_class_counts
        self.train_class_weights = train_class_weights / train_class_weights.sum()  # Normalization

        # Calculate class weights for validation set
        #
        dev_unique, dev_counts = np.unique(labels_dev, return_counts=True)
        self.dev_unique, dev_counts = tools.fillClasses(dev_unique, dev_counts)
        dev_class_counts = torch.tensor(dev_counts, dtype=torch.float32)
        dev_class_weights = max(dev_class_counts) / dev_class_counts
        self.dev_class_weights = dev_class_weights / dev_class_weights.sum()  # Normalization

    def build_model(self, model_path, train_num_cls, dev_num_cls):

        self.train_num_cls = train_num_cls
        self.dev_num_cls = dev_num_cls

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path)
        self.train_criterion = nn.CrossEntropyLoss(weight=self.train_class_weights.to(self.device))
        self.dev_criterion = nn.CrossEntropyLoss(weight=self.dev_class_weights.to(self.device))

        # Define optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        # Define learning rate scheduler
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

        # QUESTION: WHICH NUM_FTRS SHOULD I USE
        self.model.fc = nn.Linear(512, self.num_cls)

    def simple_train_model(self):
        # Track best model weights and performance
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        best_loss = float('inf')

        # Print the beginning information.
        labels = tools.getClasses(self.train_unique)
        print("Class names: ", labels)
        print("Train subdataset weights: ", self.train_class_weights)
        print("Dev subdataset weights: ", self.dev_class_weights)
        print("Train data:")
        print("--> # of images: ", self.feats_train.shape[1])
        print("--> # of classes: ", self.train_num_cls)
        print("--> # of windows: ", self.feats_train.shape[0])
        print("Eval data:")
        print("--> # of images: ", self.feats_dev.shape[1])
        print("--> # of classes: ", self.dev_num_cls)
        print("--> # of windows: ", self.feats_dev.shape[0])
        print("=============================================")

        # Start the training process
        for epoch in range(self.num_epochs):

            print(f'Epoch {epoch + 1}/{self.num_epochs}')
            print('-' * 10)

            # Set model to training mode
            self.model.train()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over training data
            
            for inputs, labels in self.train_loader:

                # Start the time for the training process
                start_time = time.perf_counter()

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
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

            # End the timer and calculate time elapsed
            end_time = time.perf_counter()
            train_time = end_time - start_time

            # Calculate training loss and accuracy
            train_epoch_loss = running_loss / len(self.train_loader.dataset)
            train_epoch_acc = running_corrects.double() / len(self.train_loader.dataset)
            
            # Validation phase
            self.model.eval()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over validation data
            for inputs, labels in self.dev_loader:

                # Start the time for the training process
                start_time = time.perf_counter()

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():  # No need to track gradients for validation
                    outputs = self.model(inputs)
                    print(outputs)
                    _, preds = torch.max(outputs, 1)
                    #preds = copy.copy(outputs)
                    loss = self.dev_criterion(outputs, labels)

                # Track statistics
                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)

            # End the timer and calculate time elapsed
            end_time = time.perf_counter()
            dev_time = end_time - start_time

            # Calculate validation loss and accuracy
            dev_epoch_loss = running_loss / len(self.dev_loader.dataset)
            dev_epoch_acc = running_corrects.double() / len(self.dev_loader.dataset)

            # Deep copy the model if it is the best one so far
            if dev_epoch_loss < best_loss:
                best_acc = dev_epoch_acc
                best_loss = dev_epoch_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())

            print(f"Train    Elapsed: {train_time:.2f} sec Loss: {train_epoch_loss:.4f} Acc: {train_epoch_acc:.4f}")
            print(f"Dev      Elapsed: {dev_time:.2f} sec Loss: {dev_epoch_loss:.4f} Acc: {dev_epoch_acc:.4f}")

        # Load best model weights
        self.model.load_state_dict(best_model_wts)
    
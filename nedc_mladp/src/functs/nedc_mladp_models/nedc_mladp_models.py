import sys
import os
import copy
import numpy as np
import time

# Machine Learning Libraries
import torch
from torch import optim, utils, nn
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# NEDC Libraries
sys.path.append('Machine-Learning-Applications-In-Digital-Pathology/nedc_mladp/lib/')
import nedc_mladp_train_tools as tools

class convolutional_neural_network:
    def __init__(self, num_epochs, batch_size, num_cls, lr, step_size, momentum, gamma, input_size):

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_cls = num_cls
        self.lr = lr
        self.step_size = step_size
        self.momentum = momentum
        self.gamma = gamma
        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self, filelist):

        # Get the total data from CSV file(s)
        #
        totaldata, image_count = tools.parsePCA(filelist)          # filelist
        labels = totaldata[:,0]
        feats = totaldata[:,1:]
        feats, labels = tools.correctType(feats,labels)

        # Create the tensors
        #
        feats_tensor = torch.tensor(feats, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.long) - 1

        # Labels contain digits [1-9]
        #
        num_cls = len(set(labels))

        return feats_tensor, label_tensor, num_cls, image_count

    def dataloader(self, feats, labels, shuffle_flag):
        '''
        arguments:
            :feats: tensor of features.
            :labels: tensor of labels.
            :shuffle_flag: True (Train) or False (Eval)
        '''

        reshaped = feats[:feats.shape[0],:].reshape(-1,1,feats.shape[1],1)
        feats = reshaped.clone().detach().to(torch.float32).repeat(1, 3, 1, 1)
        dataset = TensorDataset(feats, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle_flag)
        return dataloader

    def compute_weights(self, labels, train):
        unique, counts = np.unique(labels, return_counts=True)
        counts = tools.fillClasses(unique, counts)[1]
        counts = torch.tensor(counts, dtype=torch.float32)
        if train:
            weights = tools.getWeights(counts)
            weights = weights / weights.sum()
        else:
            weights = torch.tensor(np.zeros(9), dtype=torch.float32)
        return weights

    def build_model(self, model_path, train_weights, eval_weights):

        # Load the model.
        #
        self.model = torch.load(model_path, weights_only=False)
        self.model = self.model.to(self.device)
        self.model.fc = nn.Linear(512, self.num_cls)

        # Define hyperparameters
        #
        self.train_criterion = nn.CrossEntropyLoss(weight=train_weights.to(self.device))
        # self.eval_criterion = nn.CrossEntropyLoss(weight=eval_weights.to(self.device))
        self.eval_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

    def load_info(self, train_num_cls, train_images_count, train_feats, eval_num_cls, eval_images_count, eval_feats):

        # Extra information for printing
        #
        self.train_num_cls = train_num_cls
        self.train_images_count = train_images_count
        self.train_feats = train_feats
        self.eval_num_cls = eval_num_cls
        self.eval_images_count = eval_images_count
        self.eval_feats = eval_feats
        
    def train_model(self, train_dataloader, train_weights, eval_dataloader, eval_weights, validate):

        # Track best model weights and performance.
        #
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        best_loss = float('inf')

        # Print the beginning information.
        #
        labels = tools.getClasses(set(range(9)))
        print("Device: ", self.device)
        print("Class names: ", labels)
        print("Train weights: ", train_weights.tolist())
        print("Eval weights: ", eval_weights.tolist())
        print("Train data:")
        print("--> # of images: ", self.train_images_count)
        print("--> # of classes: ", self.train_num_cls)
        print("--> # of windows: ", self.train_feats.shape[0])
        print("Eval data:")
        print("--> # of images: ", self.eval_images_count)
        print("--> # of classes: ", self.eval_num_cls)
        print("--> # of windows: ", self.eval_feats.shape[0])
        print("==================================================")

        # Lists for storing the accuracy for each epoch.
        #
        train_accuracies = []
        eval_accuracies = []

        total_train_time = 0
        total_eval_time = 0

        # Start the training process
        #
        for epoch in range(self.num_epochs):

            print(f'Epoch {epoch + 1}/{self.num_epochs}')
            print("--------------------------------------------------")

            print("---------------------Training---------------------")
            self.model.train()
            running_loss, running_corrects, train_time = self.run_epoch(train_dataloader, self.train_criterion, train=True)

            # Step the scheduler
            #
            self.scheduler.step()

            # Calculate training loss and accuracy
            #
            train_loss = running_loss / len(train_dataloader.dataset)
            train_acc = running_corrects.double() / len(train_dataloader.dataset)

            # Keep track of the accuracy and time for each epoch
            #
            train_accuracies.append(train_acc)
            total_train_time += train_time
            
            if validate:
                print("--------------------Validation--------------------")
                self.model.eval()
                running_loss, running_corrects, eval_time = self.run_epoch(eval_dataloader, self.eval_criterion, train=False)

                # Calculate validation loss and accuracy
                #
                eval_loss = running_loss / len(eval_dataloader.dataset)
                eval_acc = running_corrects.double() / len(eval_dataloader.dataset)

                # Deep copy the model if it is the best one so far
                #
                if eval_loss < best_loss:
                    best_acc = eval_acc
                    best_loss = eval_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())

                # Keep track of the accuracy for each epoch
                #
                eval_accuracies.append(eval_acc)
                total_eval_time += eval_time

            # Load best model weights
            self.model.load_state_dict(best_model_wts)


        print(f"Train    Elapsed: {total_train_time:.2f} sec Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        if validate:
            print(f"Eval     Elapsed: {total_eval_time:.2f} sec Loss: {eval_loss:.4f} Acc: {eval_acc:.4f}")

        if validate:
            self.train_accuracies, self.eval_accuracies = train_accuracies, eval_accuracies

    def run_epoch(self, dataloader, criterion, train):

        running_loss = 0.0
        running_corrects = 0

        # Start the time for the epoch
        start_time = time.perf_counter()

        for inputs, labels in dataloader:

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero the gradients (only for train)
            if train:
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

            else:
                with torch.no_grad():
                    # Forward pass
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            print("**Predictions:\n",preds.tolist())
            print("**Labels:\n",labels.data.tolist())

        # End the timer and calculate time elapsed
        #
        end_time = time.perf_counter()
        run_time = end_time - start_time

        return running_loss, running_corrects, run_time
        

    def plot(self, directory, name):

        tools.plotPerformance(
            perf_train=self.train_accuracies,
            perf_eval=self.eval_accuracies,
            directory=directory,
            name=name,
            num_epochs=self.num_epochs
            )

    def save_model(self, output_directory, output_model_name):
        '''
        Save the model to the output directory.
        '''

        if not (output_directory.endswith("/")):
            output_directory += "/"   
            os.makedirs(output_directory,exist_ok=True)
        output_path = os.path.join(output_directory, output_model_name)

        torch.save(self.model.state_dict(), output_path)
        print("Model saved as: ")
        print(output_path)
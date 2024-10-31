#!/usr/bin/env python
#
# file: $NEDC_NFC/src/class/nedc_dpath_train.py
#
# revision history:
#
# 20230929 (SM): prep v2.0.0 for release
# 20211224 (PM): refactored code
# 20210326 (VK): initial release
#
# Usage:
#  import nedc_dpath_train as ndt
#
# This file contains a class that train the neural network for DPATH
#------------------------------------------------------------------------------

# import system modules
#
import os
import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import torch.utils as utils

# import NEDC support modules
#
# from nedc_dpath_image import ImagesList

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

TRAIN, DEV, EVAL = 'train', 'dev', 'eval'
IMG_EXT = '*.tif'
DEF_MODEL_FNAME = "model.pckl"

#------------------------------------------------------------------------------
#
# classes listed here
#
#------------------------------------------------------------------------------

class MladpTrainCNN:
    """
    class: MladpTrainCNN

    arguments:
     fname: the name of the csv file. the first column should be labels and the
            the second should be files path
     transform: all the transformations that should be done on images

    description:
     this class trains ResNet. for other kind of networks, the transform part
     must be modified   
    """

    def __init__(self, train_data, train_labels, eval_data, eval_labels, nsamples_per_class,
                 train_transforms, dev_transforms, device):
        """
        method: DPathTrain::constructor

        arguments:
         data_features: numpy array of features (float)
         data_labels: numpy array of labels as digits (int)

         train_list: the file names in train dataset
         dev_list: the fiile names in dev dataset
         nsamples_per_class: the maximum number of samples tjat in training will
                             be used
         train_transforms: the transformations on the input image in training
         dev_transforms: the transformations on the input image in development
         device: the device that the training will be run on

        returns:
         None

        description:
         this method will initialize the training object and set up two data loaders.
         
        goal: model.train(train_data, train_labels, eval_data, eval_labels)
        """

        # define the class data
        #        
        self.train_data = train_data
        self.train_labels = train_labels

        # if eval data is not provided, use train data
        #
        if eval_data is None:
            self.eval_data = train_data
            self.eval_labels = train_labels
        else:
            self.eval_data = eval_data
            self.eval_labels = eval_labels

        # self.nsamples_per_class = nsamples_per_class
        # self.device = device

        # create the data loaders for train and evaluation
        #
        train_feats_tensor = torch.tensor(self.train_data, dtype=torch.float32)
        train_label_tensor = torch.tensor(self.train_labels, dtype=torch.long)
        train_dataset = utils.TensorDataset(train_feats_tensor,train_label_tensor)

        eval_feats_tensor = torch.tensor(self.eval_data, dtype=torch.float32)
        eval_label_tensor = torch.tensor(self.eval_labels, dtype=torch.long)
        eval_dataset = utils.TensorDataset(eval_feats_tensor,eval_label_tensor)

        print("feats tensor:\n", train_feats_tensor)
        print("label tensor:\n", train_label_tensor)
        print("dataset:\n", train_dataset)

        exit(100)
        
        #------------------ rest was not written by yuan-------------------------

        # Reduce the number of samples in train
        # 
        (class_names, train_nsamples, train_subdataset,
        train_nsamples_subdataset) = \
            self.select_samples(self.train_dataset, nsamples_per_class[TRAIN])
        
        # reduce the number of samples in dev
        #
        (_, dev_nsamples, dev_subdataset,dev_nsamples_subdataset) = \
            self.select_samples(self.dev_dataset, nsamples_per_class[DEV])
            
        # set remaining class data
        #
        self.class_names = class_names
        self.train_nsamples = train_nsamples
        self.train_subdataset = train_subdataset
        self.train_nsamples_subdataset = train_nsamples_subdataset
        self.dev_nsamples = dev_nsamples
        self.dev_subdataset = dev_subdataset
        self.dev_nsamples_subdataset = dev_nsamples_subdataset

        # Introducing the transforms
        #
        # assign appropriate transforms
        #
        if hasattr(self.train_subdataset.dataset, 'transform'):
            self.train_subdataset.dataset.transform = train_transforms
        else:
            self.train_subdataset.dataset.dataset.transform = train_transforms
        if hasattr(self.dev_subdataset.dataset, 'transform'):
            self.dev_subdataset.dataset.transform = dev_transforms
        else:
            self.dev_subdataset.dataset.dataset.transform = dev_transforms

        # defining weights
        #
        train_weights = (max(self.train_nsamples_subdataset) /
                        torch.Tensor(self.train_nsamples_subdataset))
        self.train_weights = train_weights / train_weights.sum()
        dev_weights = (max(self.dev_nsamples_subdataset) /
                        torch.Tensor(self.dev_nsamples_subdataset))
        self.dev_weights = dev_weights / dev_weights.sum()
    
    # end of method

    def train(self, initial_model, batch_size, num_epochs, nworkers,
              input_model_fname=None):
        """
        method: DPathTrain::train

        argument:
         initial_model: the pretrained model that should be trained
         batch_size: the training batch size
         num_epochs: the number of epochs for training
         nworkers: the number of workers to load data
         input_model_fname: the file to save the trained model (default=None)

        return:
         model: the trained model

        description:
         this method trains a model using the ResNet pretrained model
        """

        if self.train_subdataset and self.eval_subdataset:
            self.train_dataset = self.train_subdataset
            self.eval_dataset = self.eval_subdataset

        # making dataloader for dev and train sets
        #
        self.train_dataloader = \
            utils.data.DataLoader(self.train_dataset,
                                        batch_size, shuffle=True,
                                        num_workers=nworkers)
        self.eval_dataloader = \
            utils.data.DataLoader(self.eval_dataset,
                                        batch_size, shuffle=False,
                                        num_workers=nworkers)
        
        # load model (and weights if available)
        #
        model_ft = initial_model
        if input_model_fname is not None:
            model_ft.load_state_dict(torch.load(input_model_fname,
                                                map_location=self.device))

        # configure the last layer output
        #
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(self.class_names))

        # send the model to device
        #
        model_ft = model_ft.to(self.device)

        # define loss function
        #        
        train_criterion = nn.CrossEntropyLoss(weight=
                                              self.train_weights.to(self.device))
        dev_criterion = nn.CrossEntropyLoss(weight=
                                            self.dev_weights.to(self.device))

        # Observe that all parameters are being optimized
        #
        optimizer_ft = \
            optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        #
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7,
                                                    gamma=0.1)

        # train the pretrained model
        #
        self.model = self.train_model(model_ft, train_criterion, dev_criterion,
                                       optimizer_ft, exp_lr_scheduler,
                                       self.train_dataloader,
                                       self.dev_dataloader,
                                       self.device, num_epochs)
        
        # exit gracefully
        #
        return(self.model)
    #
    # end of method
    
    def train_model(self, model, train_criterion, dev_criterion,
                     optimizer, scheduler,
                     train_dataloader, dev_dataloader, device, num_epochs=1):
        """
        method: DPathTrain::train_model

        arguments:
         model: the neural network model which should be trained
         train_criterion: loss function (such as mse or cross entropy) for train
         dev_criterion: loss function (such as mse or cross entropy) for dev
         optimizer: optimizer function (such as SGD or Adam)
         scheduler: scheduler object ot change the optimizers parameters
         train_dataloader: PyTorch data from the train set
         dev_dataloader: PyTorch data from the dev set
         device: the device to train the model on (CPU or GPU)
         num_epochs: number of epochs which neural network will be trained
                     (default=1)

        return:
         model: a trained model

        description:
         this method accepts a pretrained model and trains it according to dev
         and eval data
        """

        # compute data sizes
        #
        train_len, dev_len = (len(train_dataloader)*train_dataloader.batch_size,
                              len(dev_dataloader)*dev_dataloader.batch_size)

        # keep time trace
        #
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_loss = float('inf')

        # for every epoch
        #
        for epoch in range(num_epochs):

            # print the starting time for the epoch
            #
            epoch_time = time.time()
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            #
            # training phase
            #
            model.train()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over train data
            #
            for inputs, labels, _ in train_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                #
                optimizer.zero_grad()

                # forward
                # track history
                #
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = train_criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    #
                    loss.backward()
                    optimizer.step()

                # statistics
                #
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # finish with the train set
            #
            scheduler.step()
            epoch_loss = running_loss / train_len
            epoch_acc = running_corrects.double() / train_len
            print(f'Train \t Elapsed: {time.time()-epoch_time:.2f} sec '  
                  + f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # validation phase
            #
            model.eval()
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over train data
            #
            for inputs, labels, _ in dev_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                #
                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = dev_criterion(outputs, labels)

                # statistics
                #
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            # finish the dev set
            #
            epoch_loss = running_loss / dev_len
            epoch_acc = running_corrects.double() / dev_len
            print(f'Dev \t Elapsed: {time.time()-epoch_time:.2f} sec '  
                  + f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            #
            if epoch_loss < best_loss:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        # print timing info for training
        #
        print()
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m ' +
              f'{time_elapsed % 60:.0f}s')

        # load best model weights
        #
        model.load_state_dict(best_model_wts)
        
        # exit gracefully
        #
        return model
    
    #
    # end of method

    def decode(self, model, dataloader, device):
        """
        method: DpathTrain::decode

        arguments:
         model: the neural network model
         dataloader: the dataloaders dictionary
         device: the device to decode on (GPU or CPU)

        return:
         all_labels: a dictionary of labels for the confusion matrix 
         all_preds: a dictionary of predictions for the confusion matrix
         accuracy: a total classification accuracy
        """

        # save the model state to restore it at the end
        #
        was_training = model.training
        model.eval()

        # extract all labels and prdeictions for passing to confusion matrix
        #
        all_labels = np.zeros(len(dataloader)*dataloader.batch_size)
        all_preds = np.zeros(len(dataloader)*dataloader.batch_size)

        # loop over all epochs
        #
        lcounter = 0
        with torch.no_grad():
            for i, (inputs, labels, _) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_labels[lcounter:lcounter+len(labels)] = \
                    labels.cpu().numpy()
                all_preds[lcounter:lcounter+len(preds)] = preds.cpu().numpy()
                lcounter += len(labels)

        # change the model state to last state
        #
        model.train(mode=was_training)

        # computing accuracy based on simple mean absolute error
        #
        accuracy = np.mean(all_labels == all_preds)

        # exit gracefully
        #
        return all_labels, all_preds, accuracy
    #
    # end of method
    
#
# end of class

#
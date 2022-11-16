#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:48:09 2022

@author: saurabhraj
"""
#ref: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
#ref: https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627
#ref: https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118
#ref: https://nextjournal.com/gkoehler/pytorch-mnist


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from scipy.sparse.linalg.eigen import test
from sklearn.utils import shuffle
from tqdm import tqdm


 
class NeuralNetClassifier(nn.Module):
    def __init__(self, input_size, hidden_neurons, output_size):
        super(NeuralNetClassifier, self).__init__()
        """The parameteres: 
            input_size - input size of each image
            hidden_neurons - a list of neurons we want in each hidden layer
            output_size - number of distinct labels in output
        """
        self.input_size = input_size
        self.hidden_neurons = hidden_neurons
        self.output_size = output_size
        
        """We will now build out neural network:
            input layer -> hidden_layer1 (Relu)->hidden_layer2 (Sigmoid)-> output_layer
        """
        
        
        self.Model = nn.Sequential(
            nn.Linear(input_size, hidden_neurons[0]),
            nn.ReLU(),
            nn.Linear(hidden_neurons[0], hidden_neurons[1]),
            nn.Sigmoid(),
            nn.Linear(hidden_neurons[1], output_size),
            nn.Softmax(dim=1)
            
            )     
        
    def forward(self,x):
        preds = self.Model(x)
        return preds
    
     

def getData(download=False):
  train_data = datasets.MNIST(root='/content/drive/MyDrive/INM702_CW_MNIST/Data/',
                            train=True,
                            transform =transforms.ToTensor(),
                            download=True)

  test_data = datasets.MNIST(root='/content/drive/MyDrive/INM702_CW_MNIST/Data/',
                           train = False,
                           transform =transforms.ToTensor(),
                           download=True)     

  return train_data, test_data

def visualize(rows, cols):
  plt.figure(figsize=(12,8))
  for i in range(rows*cols):
    plt.subplot(2,3,i+1)
    plt.imshow(train_data.data[i])
    plt.title(train_data.targets[i])


def create_dataloaders(train_data, test_data, batch_size):
  #since the batch size is 100, total training points (60000) would be divided into 60000/100 = 600 batches.
  train_loader = DataLoader(train_data,
                            batch_size = batch_size,
                            shuffle=True,
                            num_workers=1)
  test_loader = DataLoader(test_data, 
                           batch_size = batch_size,
                           shuffle=True, 
                           num_workers=1)
  return (train_loader, test_loader)





def train(model,loss_criterion,optimizer,e,epoch):
  #To report the loss, i keep a loss sum counter which add loss on all the batches in a \n
  #single epoch and then divides that loss sum with total number of batches.
  model.train()
  count = 0
  num_batches = len(train_loader)
  #for e in range(epoch):
  epoch_loss = 0
  for batch, (data, target) in enumerate(train_loader):

    X_train = data.view(data.shape[0], -1)
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_criterion(y_pred, target)
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()
    if (batch+1)%num_batches == 0:
      train_loss.append(epoch_loss/num_batches)
      print('Epoch : {}|{}    =================>    Train Loss : {:.4f}'.format(e,epoch, epoch_loss/num_batches))
      epoch_loss = 0

  

def test(model,e,epoch):
  model.eval()
  correct, total = 0,0
  epoch_loss = 0
  num_batches = len(test_loader)
  with torch.no_grad():
    for batch, (data, target) in enumerate(test_loader):
      
      data = data.view(data.shape[0],-1)
      #print(data.shape)
      output = model(data)
      y_pred = torch.max(output,1) 
      #works same as np.argmax. 1 is for vertical
      #torch.max return values and indices both. We only need indices
      y_pred = y_pred[1]
      #print(y_pred)
      # print(target)
      for i in range(len(y_pred)):
        if y_pred[i] == target[i]:
          correct += 1
      total += batch_size
      epoch_loss += loss_criterion(output, target)

      if (batch+1)%num_batches == 0:
        test_loss.append(epoch_loss/num_batches)
        print('Epoch : {}|{}    =================>    Val Loss : {:.4f}'.format(e,epoch, epoch_loss/num_batches))
        epoch_loss = 0


  print(f'Accuracy : {(correct/total)*100}')

    
def getInfo(train_data, test_data, model):
  print('Train data size : ', train_data.data.size())
  print('Test data size : ', test_data.data.size())
  X, y =  next(iter(train_loader))
  print(f'Shape of X coming out of train loader : {X.shape}')
  X = X.view(X.shape[0],-1)
  print(f'Shape of X going into training : {X.shape}')

  print(f'Model Info : {model}')
  
def start_training(epoch):
    
    for e in range(1,epoch+1):
      train(model,loss_criterion,optimizer,e,epoch)
      test(model,e, epoch)





#parameters and variables initiations:

train_data, test_data = getData()
visualize(2,3)
batch_size = 64
loss_criterion = nn.CrossEntropyLoss()
input_size = 784
hidden_neurons = [128,64] 
output_size = 10

#model
model = NeuralNetClassifier(input_size, hidden_neurons, output_size)
train_loss = []
test_loss = []

(train_loader, test_loader) = create_dataloaders(train_data, test_data, batch_size)
getInfo(train_data, test_data, model)

lr = 0.1 #learning rate
optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.8)  #optimizer
epoch = 10

#start_training(epoch)



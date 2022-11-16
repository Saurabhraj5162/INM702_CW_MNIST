#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:48:09 2022

@author: saurabhraj
"""
#ref: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
#ref: https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import dataloader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


 
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
    
    
        
def download_data():
        
    train_data = datasets.MNIST(root='/Users/saurabhraj/Desktop/CITY AI/Programming & Maths/INM702_CW_MNIST/src/data/',
                                train=True,
                                transform =transforms.ToTensor(),
                                download=False)
    
    test_data = datasets.MNIST(root='Users/saurabhraj/Desktop/CITY AI/Programming & Maths/INM702_CW_MNIST/src/data/',
                               train = False,
                               transform =transforms.ToTensor(),
                               download=False)     

    return (train_data, test_data)

(train_data, test_data) = download_data()

print(train_data.data.size())

# df = pd.read_csv('/Users/saurabhraj/Desktop/CITY AI/Programming & Maths/INM702_CW_MNIST/src/data/mnist_test.csv')
# #print(df.head())
# #print(len(df))

# X = df.drop('label', axis=1)
# y = df['label']

# X_train, X_test, y_train, y_test = train_test_split(X , y,test_size=0.25, random_state=42)
# #print(f'X train shape : {X_train.shape} || Y train shape : {y_train.shape}')
# #print(f'X test shape : {X_test.shape} || Y train shape : {y_train.shape}')

# input_size = X_train.shape
# print(f'input size : {input_size}')   
     

# hidden_neurons = [(7500,256),(7500,256)]
# output_size = len(set(y))
# model1 = NeuralNetClassifier(input_size, hidden_neurons, output_size)
# y_pred = model1.forward(X_train)
        
# print(f'acc : {accuracy_score(y_test, y_pred)}')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
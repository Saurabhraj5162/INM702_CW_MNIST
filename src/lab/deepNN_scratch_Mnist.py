#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 12:25:02 2022

@author: saurabhraj
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical



def sigmoid(Z):
  """ Arg: Linear activation Z
      return: Sigmoid of Z
  """
  return 1/(1+np.exp(-Z))

def sigmoid_derivative(dA,Z):
  """ Arg: Derivative of Loss w.r.t activation (dL/dAl), Linear activation Z
      return: dL/dZl = (dL/dAl)*(dAl/dZl)
              (dAl/dZl) = Derivative of activation w.r.t linear activation(Z)
  """
  dZl = sigmoid(Z)*(1-sigmoid(Z))

  return dA*dZl

def relu(Z):
  """ Arg: Linear activation Z
      return: Relu of Z
  """
  return np.maximum(0,Z)

def relu_derivative(dA,Z):
  """ Arg: Derivative of Loss w.r.t activation (dL/dAl), Linear activation Z
      return: dL/dZl = (dL/dAl)*(dAl/dZl)
              (dAl/dZl) = Derivative of activation w.r.t linear activation(Z)
  """
  
  dZ = Z.copy()

  dZ = np.array(dZ>0, dtype=np.float32)


  return dA*dZ


def softmax(Z):
  #print(Z.shape)
  e = np.exp(Z)
  e_total = np.sum(e, axis=0, keepdims=True)
  #print(e.shape, e_total.shape)
  
  return np.divide(e,e_total)


def initialize_params(layers_list, method):
  """ Args: layers_list : containing the number of neurons (nodes) for each layer in the netwrok.
            method: a tuple. method[0] contains type of initialization (random/trunc_normal)
                              method[1] contains multiplying_factor for random initialization
                                        contains (mean, sd,low,upp) for truncated_normal
      returns: Initialized parameters : Weights (W), Bias (b) for each layers in a dictionary data structure.

  """

  params = {}
  mult_factor = 0.01
  num_layers = len(layers_list)
  if method[0] == 'random':
    for l in range(1,num_layers):
      params['W' + str(l)] = np.random.randn(layers_list[l], layers_list[l-1])*mult_factor
      params['b' + str(l)] = np.zeros((layers_list[l], 1))

  if method[0] == 'trunc_normal':
    mean, sd,low,upp = method[1]
    for l in range(1, num_layers):
      temp = truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
      params['W' + str(l)] = temp.rvs((layers_list[l], layers_list[l-1]))
      params['b' + str(l)] = np.zeros((layers_list[l], 1))

  return params




def single_forward(W,b,A,activation_function):
  """
  Args: W,b - parameters for the current layer; 
        A- linear activation (Z = WA + b) for the current layer;
        activation_function
  returns: A_forward, 
  """

  Z_current = np.dot(W,A) + b
  #Z_current = np.squeeze(Z_current)
  #print(Z_current.shape)
  #print(Z_current)
  linear_memory = (A,W,b)

  if activation_function == 'sigmoid':
    A_current = sigmoid(Z_current)

  if activation_function == 'relu':
    A_current= relu(Z_current)

  if activation_function == 'softmax':
    A_current = softmax(Z_current)

  memory_current = (linear_memory, Z_current)
  return (A_current, memory_current)




def forward_propagation(A0, params, activation_functions):
  """ 
  Args: A0 - X input
        params - dictionary containing W and b for all the layers
        activation_functions containing activation function used in all the layers (['relu',...,'sigmoid','softmax'])
  returns: Activation output of last layer and the memory (W,b,Z,A) in each layer
  """
  num_layers = len(params)//2
  #print(f'shape of num_layers : {num_layers}')
  memory = []
  A_current = A0
  for l in range(1,num_layers):
    W_current = params['W' + str(l)]
    b_current = params['b' + str(l)]
    A_prev = A_current
    #print(f'W : {W_current.shape} A : {A_current.shape}')
    A_current, memory_current = single_forward(W_current, b_current, A_prev, activation_functions[l])
    memory.append(memory_current)
  
  W_current = params['W'+str(num_layers)]
  b_current = params['b'+str(num_layers)]
  #print("Activation for lasy layer : ", activation_functions[num_layers])
  A_last, memory_current = single_forward(W_current, b_current, A_current, activation_functions[num_layers])
  memory.append(memory_current)  
  #print(A_last.shape, A0.shape)
  #print(memory_current)
  #print(memory)
  return A_last, memory


  



def calculate_cost(A_last, Y):
  m = Y.shape[1]
  #print(Y.shape, A_last.shape)
  part1 = Y*np.log(A_last)

  cost = (-1/m)*np.sum(part1)
  cost = np.squeeze(cost)
  return cost

def single_backward(dA, memory, activation_function):
  """ 
  Args: dA - derivative of next layer output w.r.t A of current layer
        memory - contains values of Z,W,b,A for corresponding layer stored while forward propagation.
        activation_function - 'relu'/'sigmoid'
  returns: dA_previous (dL/dAl-1) - derivative of cost w.r.t activation of previous layers, 
           dW (dL/dWl)- derivative of cost w.r.t weight W of current layer
           db (dL/db) -  derivative of cost w.r.t bias b of current layer
  """
  A_previous, W, b = memory[0]
  Z = memory[1]
  m = A_previous.shape[1]

  if activation_function == 'relu':
    dZ = relu_derivative(dA,Z)

  elif activation_function == 'sigmoid':
    dZ = sigmoid_derivative(dA,Z)




  
  dW = (1/m)*np.dot(dZ, A_previous.T)
  db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
  dA_previous = np.dot(W.T,dZ)

  return dA_previous, dW, db



def backward_propagation(A_last, Y, memory, activation_functions):
  
  num_layers = len(memory)
  grads = {}
  Y = Y.reshape(A_last.shape)
  m = Y.shape[1]

  #dA_last = A_last - Y

  memory_current = memory[-1] #(A_L-1,W_L,b_L), Z_last(output of softmax layer)
  dZ_last = A_last - Y

  A_previous, W, b = memory_current[0]
  
  dW_current = (1/m)*np.dot(dZ_last, A_previous.T)
  db_current = (1/m)*np.sum(dZ_last, axis=1, keepdims=True)
  dA_previous = np.dot(W.T,dZ_last)

  #dW_current, db_current = single_backward(dZ_last, memory_current, activation_functions[-1])
    
  grads["dA" + str(num_layers-1)] = dA_previous
  grads["dW" + str(num_layers)] = dW_current
  grads["db" + str(num_layers)] = db_current

  for l in range(num_layers-2,-1,-1):
    memory_current = memory[l]
    #print('flkqenfnejnqn ', activation_functions[l])
    dA_previous = grads["dA" + str(l+1)]
    dA_previous, dW_current, db_current = single_backward(dA_previous, memory_current, activation_functions[l])
    
    grads["dA" + str(l)] = dA_previous
    grads["dW" + str(l+1)] = dW_current
    grads["db" + str(l+1)] = db_current


  return grads




def gradient_descent(params, grads, learning_rate):
  """
  Args: params - initialized parameters (W,b). {W1:__, b1:__}
        grads - calculated gradients dictionary during backward_propagation
        learning_rate - alpha of gradient descent

  return : updated params dictionary
  """
  num_layers = len(params)//2
  updated_params = params.copy()
  for l in range(1,num_layers+1):
    updated_params['W' + str(l)] -= learning_rate*grads['dW' + str(l)]
    updated_params['b' + str(l)] -= learning_rate*grads['db' + str(l)]

  return updated_params






def visualize(rows, cols):
  plt.figure(figsize=(12,8))
  for i in range(rows*cols):
    plt.subplot(2,3,i+1)
    plt.imshow(train_x_orig[i])



def getInfo(train_x_orig, train_y, test_x_orig, test_y):
  print('X Train data size : ', train_x_orig.shape)
  print('Y Train data size : ', train_y.shape)
  print('X Test data size : ', test_x_orig.shape)
  print('Y Test data size : ', test_y.shape)
  print('='*50)
  #print('Total training examples : ', train_y.shape[0] )
  
  
  
def preprocess_data(train_x_orig, train_y, test_x_orig, test_y, n_classes):
  m_train = train_y.shape[0]
  X_train = train_x_orig.reshape(m_train, -1).T
  X_train = X_train/255.
  y_train = to_categorical(train_y,n_classes)
  y_train = y_train.T
  

  m_test = test_y.shape[0]
  X_test = test_x_orig.reshape(m_test, -1).T
  X_test = X_test/255.
  y_test = to_categorical(test_y,n_classes)
  y_test = y_test.T
 

  getInfo(X_train, y_train, X_test, y_test)
  
  return (X_train, y_train, X_test, y_test)
  

  

# MY MODEL:
"""
def initialize_params(layers_list, method):
    ...
    return parameters 
def forward_propagation(A0, params, activation_functions):
    ...
    return A_last, [((A_L-1,W_L,b_L), Z_last),
                    ((A_L-2,W_L-1,b_L-1), Z_last-1),
                    ((A_L-3,W_L-2,b_L-2), Z_last-2)
                    .
                    .
                    .
                    ((A0(X),W_1,b_1), Z_1)]
def calculate_cost(A_last, Y):
    ...
    return cost
def backward_propagation(A_last, Y, memory, activation_functions):
    ...
    return {dAL-1, dWL, dbL,
            dAL-2, dWL-1, dbL-1,
            .
            .
            .
            dA0, dW1, db1}
def gradient_descent(params, grads, learning_rate):
    ...
    return updated_parameters
"""

def L_layer_model(X, Y, layers_list, activation_functions,weight_initializer ,learning_rate = 0.1, num_iterations = 3000, print_cost=False):


    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization.
    
    params = initialize_params(layers_list, weight_initializer)
    #print(params.keys())
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SOFTMAX
        
        A_last, memory = forward_propagation(X, params, activation_functions)
        #print('A last shape : ', A_last.shape)
        #print('memory : ', memory.keys())
        
        # Compute cost.
        cost = calculate_cost(A_last, Y)
        
      
        # Backward propagation.
   
        grads = backward_propagation(A_last, Y, memory, activation_functions)
        
 
        # Update parameters.
        params = gradient_descent(params, grads, learning_rate)
        

                
        # Print the cost every 100 iterations
        if print_cost and i % 10 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 10 == 0 or i == num_iterations:
            costs.append(cost)
    
    return params, costs

(train_x_orig, train_y), (test_x_orig, test_y) = mnist.load_data()
getInfo(train_x_orig, train_y, test_x_orig, test_y)
X_train, y_train, X_test, y_test = preprocess_data(train_x_orig, train_y, test_x_orig, test_y, 10)

input_shape = 784
n_classes = 10
layers_list = [input_shape, 1000, n_classes]
activation_functions = ['relu','relu','softmax']
weight_initializer = ['random',0.01]
#weight_initializer = ['trunc_normal',(0, 1, -0.5, 0.5)]


"""Training: """
parameters, costs = L_layer_model(X_train, y_train, layers_list, activation_functions, weight_initializer, num_iterations = 100, print_cost = True)




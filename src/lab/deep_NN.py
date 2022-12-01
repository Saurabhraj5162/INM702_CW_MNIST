# -*- coding: utf-8 -*-
"""Deep Neural Network INM702CW2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1n9s0cwTMxjiDiXJTcQTATs2awntB16dt
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
#from keras.datasets import mnist
#from tensorflow.keras.utils import to_categorical

class My_Neural_Network():
  def __init__(self, layers_list,
               activation_functions,weight_initializer, 
               learning_rate,num_iterations, lambd,
               drop_rate, sgd):

    self.layers_list = layers_list
    self.activation_functions = activation_functions
    self.weight_initializer = weight_initializer
    self.learning_rate = learning_rate
    self.num_iterations = num_iterations
    self.lambd = lambd
    self.drop_rate = drop_rate
    self.sgd = sgd


  def sigmoid(self,Z):
    """ Arg: Linear activation Z
        return: Sigmoid of Z
    """
    return 1/(1+np.exp(-Z))

  def sigmoid_derivative(self,dA,Z):
    """ Arg: Derivative of Loss w.r.t activation (dL/dAl), Linear activation Z
        return: dL/dZl = (dL/dAl)*(dAl/dZl)
                (dAl/dZl) = Derivative of activation w.r.t linear activation(Z)
    """
    dZl = self.sigmoid(Z)*(1-self.sigmoid(Z))

    return dA*dZl

  def relu(self,Z):
    """ Arg: Linear activation Z
        return: Relu of Z
    """
    return np.maximum(0,Z)

  def relu_derivative(self,dA,Z):
    """ Arg: Derivative of Loss w.r.t activation (dL/dAl), Linear activation Z
        return: dL/dZl = (dL/dAl)*(dAl/dZl)
                (dAl/dZl) = Derivative of activation w.r.t linear activation(Z)
    """
    
    dZ = Z.copy()

    dZ = np.array(dZ>0, dtype=np.float32)


    return dA*dZ


  def softmax(self,Z):
    #print(Z.shape)
    e = np.exp(Z)
    e_total = np.sum(e, axis=0, keepdims=True)
    #print(e.shape, e_total.shape)
    
    return np.divide(e,e_total)




  def initialize_params(self, layers_list, method):
    """ Args: layers_list : containing the number of neurons (nodes) for each layer in the netwrok.
              method: a tuple. method[0] contains type of initialization (random/trunc_normal)
                                method[1] contains multiplying_factor for random initialization
                                          contains (mean, sd,low,upp) for truncated_normal
        returns: Initialized parameters : Weights (W), Bias (b) for each layers in a dictionary data structure.

    """

    params = {}
    num_layers = len(layers_list)
    if method == 'random':
      mult_factor = 0.01
      for l in range(1,num_layers):
        params['W' + str(l)] = np.random.randn(layers_list[l], layers_list[l-1])*mult_factor
        params['b' + str(l)] = np.zeros((layers_list[l], 1))

    if method == 'trunc_normal':
      mean, sd,low,upp = 0, 1, -0.5, 0.5
      for l in range(1, num_layers):
        temp = truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
        params['W' + str(l)] = temp.rvs((layers_list[l], layers_list[l-1]))
        params['b' + str(l)] = np.zeros((layers_list[l], 1))

    if method == 'zeros':
      for l in range(1,num_layers):
        params['W' + str(l)] = np.zeros((layers_list[l], layers_list[l-1]))
        params['b' + str(l)] = np.zeros((layers_list[l], 1))


    if method == 'he':
      for l in range(1,num_layers):
        params['W' + str(l)] = np.random.randn(layers_list[l], layers_list[l-1])*np.sqrt(2/layers_list[l-1])
        params['b' + str(l)] = np.zeros((layers_list[l], 1))

    if method == 'xavier':
      for l in range(1,num_layers):
        params['W' + str(l)] = np.random.randn(layers_list[l], layers_list[l-1])*np.sqrt(1/layers_list[l-1])
        params['b' + str(l)] = np.zeros((layers_list[l], 1))

    return params


  def linear_activation(self, W,b,A,activation_function):
    """
    Args: W,b - parameters for the current layer; 
          A- linear activation (Z = WA_prev + b) for the current layer;
          activation_function
    returns: A_forward, memories of current activation, dropout memory (if dropout)
    """

    Z_current = np.dot(W,A) + b
    #Z_current = np.squeeze(Z_current)
    #print(Z_current.shape)
    #print(Z_current)
    linear_memory = (A,W,b)
    drop_flag = 0
    if activation_function == 'sigmoid':
      A_current = self.sigmoid(Z_current)

    if activation_function == 'relu':
      A_current= self.relu(Z_current)

    if activation_function == 'relu + dropout':
      A_current= self.relu(Z_current)
      D_current = np.random.rand(A_current.shape[0], A_current.shape[1])
      D_current = (D_current < (1- self.drop_rate)).astype(int)
      A_current = A_current*D_current
      A_current = A_current/(1- self.drop_rate)
      drop_flag = 1

    
    if activation_function == 'sigmoid + dropout':
      A_current= self.sigmoid(Z_current)
      D_current = np.random.rand(A_current.shape[0], A_current.shape[1])
      D_current = (D_current < (1- self.drop_rate)).astype(int)
      A_current = A_current*D_current
      A_current = A_current/(1- self.drop_rate)
      drop_flag = 1
      

    if activation_function == 'softmax':
      A_current = self.softmax(Z_current)

    if drop_flag == 0:
      memory_current = (linear_memory, Z_current)
    else:
      memory_current = (linear_memory, Z_current, D_current)
    return (A_current, memory_current)



  def forward_propagation(self, A0, params, activation_functions):
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
      A_current, memory_current = self.linear_activation(W_current, b_current, A_prev, activation_functions[l])
      memory.append(memory_current)
    
    W_current = params['W'+str(num_layers)]
    b_current = params['b'+str(num_layers)]
    #print("Activation for lasy layer : ", activation_functions[num_layers])
    A_last, memory_current = self.linear_activation(W_current, b_current, A_current, activation_functions[num_layers])
    memory.append(memory_current)  
    #print('Output vs input shape of FWD prop:',A_last.shape, A0.shape)
    #print(memory_current)
    #print(memory)
    return A_last, memory





  def calculate_cost(self, A_last, Y):
    #print('cost : ', Y.shape)
    m = Y.shape[1]
    #print(Y.shape, A_last.shape)
    part1 = Y*np.log(A_last)

    cost = (-1/m)*np.sum(part1)
    cost = np.squeeze(cost)
    return cost


  def calculate_cost_L2_regularised(self, A_last, Y, parameters, lambd):
    m = Y.shape[1]
    L = len(parameters)//2
    cost_without_reg = self.calculate_cost(A_last, Y)
    L2_reg = 0
    for l in range(1,L+1):
      W = parameters['W' + str(l)]
      L2_reg += np.sum(np.square(W))

    L2_reg = (lambd/(2*m))*L2_reg

    
    cost = cost_without_reg + L2_reg
    
    return cost

  def linear_backward(self,dA, memory, activation_function):
    """ 
    Args: dA - derivative of next layer output w.r.t A of current layer
          memory - contains values of Z,W,b,A for corresponding layer stored while forward propagation.
          activation_function - 'relu'/'sigmoid'
    returns: dA_previous (dL/dAl-1) - derivative of cost w.r.t activation of previous layers, 
            dW (dL/dWl)- derivative of cost w.r.t weight W of current layer
            db (dL/db) -  derivative of cost w.r.t bias b of current layer
    """
    
    if self.drop_rate ==0:
      A_previous, W, b = memory[0]
      Z = memory[1]
    else:
      A_previous, W, b = memory[0]
      Z = memory[1]
      D = memory[2]

    m = A_previous.shape[1]

    if activation_function == 'relu':
      dZ = self.relu_derivative(dA,Z)

    elif activation_function == 'sigmoid':
      dZ = self.sigmoid_derivative(dA,Z)

    elif activation_function == 'relu + dropout':
      dA = dA*D
      dA = dA/(1-self.drop_rate)
      dZ = self.relu_derivative(dA,Z)


    elif activation_function == 'sigmoid + dropout':
      dA = dA*D
      dA = dA/(1-self.drop_rate)
      dZ = self.sigmoid_derivative(dA,Z)




    
    dW = (1/m)*np.dot(dZ, A_previous.T)
    db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
    dA_previous = np.dot(W.T,dZ)

    return dA_previous, dW, db


  def backward_propagation(self,A_last, Y, memory, activation_functions):
    
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

    #dW_current, db_current = linear_backward(dZ_last, memory_current, activation_functions[-1])
      
    grads["dA" + str(num_layers-1)] = dA_previous
    grads["dW" + str(num_layers)] = dW_current
    grads["db" + str(num_layers)] = db_current

    for l in range(num_layers-2,-1,-1):
      memory_current = memory[l]
      #print('flkqenfnejnqn ', activation_functions[l])
      dA_previous = grads["dA" + str(l+1)]
      dA_previous, dW_current, db_current = self.linear_backward(dA_previous, memory_current, activation_functions[l])
      
      grads["dA" + str(l)] = dA_previous
      grads["dW" + str(l+1)] = dW_current
      grads["db" + str(l+1)] = db_current


    return grads


  def backward_propagation_L2_regularised(self, A_last, Y, memory, activation_functions, lambd):
    
    num_layers = len(memory)
    grads = {}
    Y = Y.reshape(A_last.shape)
    m = Y.shape[1]

    memory_current = memory[-1] #(A_L-1,W_L,b_L), Z_last(output of softmax layer)
    dZ_last = A_last - Y

    A_previous, W, b = memory_current[0]
    
    dW_current = (1/m)*np.dot(dZ_last, A_previous.T) + (lambd/m)*W
    db_current = (1/m)*np.sum(dZ_last, axis=1, keepdims=True)
    dA_previous = np.dot(W.T,dZ_last)

    #dW_current, db_current = linear_backward(dZ_last, memory_current, activation_functions[-1])
      
    grads["dA" + str(num_layers-1)] = dA_previous
    grads["dW" + str(num_layers)] = dW_current
    grads["db" + str(num_layers)] = db_current

    for l in range(num_layers-2,-1,-1):
      memory_current = memory[l]
      #print('flkqenfnejnqn ', activation_functions[l])
      dA_previous = grads["dA" + str(l+1)]
      dA_previous, dW_current, db_current = self.linear_backward(dA_previous, memory_current, activation_functions[l])

      A_previous_, W_, b_ = memory_current[0] #fetching memory cache
      dW_current += (lambd/m)*W_    #adding L2 regularised term
      
      grads["dA" + str(l)] = dA_previous
      grads["dW" + str(l+1)] = dW_current
      grads["db" + str(l+1)] = db_current


    return grads

  def gradient_descent(self,params, grads, learning_rate):
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




  def train(self,X,Y,print_cost=True):

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


      np.random.seed(1)
      costs = []               
      m = X.shape[1]
      params = self.initialize_params(self.layers_list, self.weight_initializer)

      #lopping for number of iterations:

      if self.sgd == True:
        for i in range(0, self.num_iterations):
          total_cost = 0
          for j in range(0,m):
            
            # Forward propagation:
            print(np.reshape(X[:,j],(X.shape[0],1)))
            A_last, memory = self.forward_propagation(np.reshape(X[:,j],(X.shape[0],1)), params, self.activation_functions)
            #print('A Last : ',A_last.shape)
            # Compute cost.

            if self.lambd == 0:
              total_cost += self.calculate_cost(A_last, np.reshape(Y[:,j],(Y.shape[0],1)))
            else:
              total_cost += self.calculate_cost_L2_regularised(A_last, np.reshape(Y[:,j],(Y.shape[0],1)), params, self.lambd)
            #print('Cost : ',cost)
            # Backward propagation.
            if self.lambd==0:
              grads = self.backward_propagation(A_last, np.reshape(Y[:,j],(Y.shape[0],1)), memory, self.activation_functions)
            elif self.lambd !=0:
              grads = self.backward_propagation_L2_regularised(A_last, np.reshape(Y[:,j],(Y.shape[0],1)), memory, self.activation_functions, self.lambd)
         

          cost = total_cost/m
          # Update parameters.
          params = self.gradient_descent(params, grads, self.learning_rate)
        
          # Print the cost every 10 iterations
          if print_cost and i % 10 == 0 or i == self.num_iterations - 1:
              print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
          if i % 10 == 0 or i == self.num_iterations:
              costs.append(cost)

      else:
        for i in range(0, self.num_iterations):

            # Forward propagation:
            
            A_last, memory = self.forward_propagation(X, params, self.activation_functions)
            
            #print('A last shape : ', A_last.shape)
            #print('memory : ', memory.keys())

            # Compute cost.
            if self.lambd == 0:
              cost = self.calculate_cost(A_last, Y)
            else:
              cost = self.calculate_cost_L2_regularised(A_last, Y, params, self.lambd)
            #cost = self.calculate_cost(A_last, self.Y)
                    
            # Backward propagation.
            if self.lambd==0:
              grads = self.backward_propagation(A_last, Y, memory, self.activation_functions)
            elif self.lambd !=0:
              grads = self.backward_propagation_L2_regularised(A_last, Y, memory, self.activation_functions, self.lambd)
          

            
            # Update parameters.
            params = self.gradient_descent(params, grads, self.learning_rate)
          
            # Print the cost every 10 iterations
            if print_cost and i % 10 == 0 or i == self.num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 10 == 0 or i == self.num_iterations:
                costs.append(cost)
        
      return params, costs


  def predict(self,input_x,final_params):
    y_pred_scores, memory_pred = self.forward_propagation(input_x, final_params, self.activation_functions)
    y_pred = np.argmax(y_pred_scores,axis=0)

    return y_pred, y_pred_scores

  def accuracy(self,y_pred, y_actual):
    acc = np.sum(np.equal(y_actual, y_pred)) / len(y_actual)
    return acc

class Data():
  def visualize(self, rows, cols):
    plt.figure(figsize=(12,8))
    for i in range(rows*cols):
      plt.subplot(2,3,i+1)
      plt.imshow(train_x_orig[i])


  def getInfo(self, train_x_orig, train_y, test_x_orig, test_y):
    print('X Train data size : ', train_x_orig.shape)
    print('Y Train data size : ', train_y.shape)
    print('X Test data size : ', test_x_orig.shape)
    print('Y Test data size : ', test_y.shape)
    print('='*50)
    #print('Total training examples : ', train_y.shape[0] )


  def preprocess_data(self,train_x_orig, train_y, test_x_orig, test_y, n_classes):
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
  

    self.getInfo(X_train, y_train, X_test, y_test)
    
    return (X_train, y_train, X_test, y_test)


def main():

    (train_x_orig, train_y), (test_x_orig, test_y) = mnist.load_data()
    data = Data()
    data.visualize(2,3)
    X_train, y_train, X_test, y_test = data.preprocess_data(train_x_orig, train_y, test_x_orig, test_y, 10)
    
    """# Model 1"""
    
    #without regularization
    
    X = X_train
    Y = y_train
    input_shape = X_train.shape[0]
    n_classes = len(np.unique(train_y))
    layers_list = [input_shape, 64,64, n_classes]
    activation_functions = ['relu','relu','relu','softmax']
    weight_initializer = 'he'
    learning_rate = 0.1
    num_iterations = 300
    lambd = 0
    drop_rate=0
    sgd = False
    
    
    
    
    my_model = My_Neural_Network(layers_list,activation_functions, weight_initializer,learning_rate,num_iterations,lambd,drop_rate, sgd)
    
    parameters, costs =  my_model.train(X_train, y_train)
    
    plt.plot(range(1,num_iterations,10),costs)
    plt.title('Cost vs Iteration - without regularisation')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    
    y_pred, y_pred_scores = my_model.predict(X_test,parameters)
    print(y_pred.shape, test_y.shape)
    print('Test Accuracy : ',my_model.accuracy(y_pred, test_y))
    
    y_pred, y_pred_scores = my_model.predict(X_train,parameters)
    print(y_pred.shape, train_y.shape)
    print('Train Accuracy : ',my_model.accuracy(y_pred, train_y))
    
    """### It will go further lower if I increase the number of epochs (iterations)."""
    
    #without regularization
    
    X = X_train
    Y = y_train
    input_shape = X_train.shape[0]
    n_classes = len(np.unique(train_y))
    layers_list = [input_shape, 64,64,64, n_classes]
    activation_functions = ['relu','relu','relu','sigmoid','softmax']
    weight_initializer = 'he'
    learning_rate = 0.1
    num_iterations = 300
    lambd = 0
    drop_rate=0
    sgd = False
    
    
    
    
    my_model2 = My_Neural_Network(layers_list,activation_functions, weight_initializer,learning_rate,num_iterations,lambd,drop_rate, sgd)
    
    parameters, costs =  my_model2.train(X_train, y_train)
    
    
    
    #without regularization
    
    X = X_train
    Y = y_train
    input_shape = X_train.shape[0]
    n_classes = len(np.unique(train_y))
    layers_list = [input_shape, 256,64, n_classes]
    activation_functions = ['relu','relu','relu','softmax']
    weight_initializer = 'he'
    learning_rate = 0.1
    num_iterations = 100
    lambd = 1000
    
    
    
    my_model_reg = My_Neural_Network(X,Y,input_shape,n_classes,layers_list,activation_functions, weight_initializer,learning_rate,num_iterations,lambd)
    
    parameters, costs =  my_model_reg.train()
    
    """### Since the model was already generalizing pretty well, regularization didnt change much. """
    
    #without regularization
    
    X = X_train
    Y = y_train
    input_shape = X_train.shape[0]
    n_classes = len(np.unique(train_y))
    layers_list = [input_shape, 256,64, n_classes]
    activation_functions = ['relu','relu + dropout','relu + dropout','softmax']
    weight_initializer = 'he'
    learning_rate = 0.1
    num_iterations = 100
    lambd = 0
    drop_rate = 0.2
    sgd = True
    
    
    
    my_model_drop = My_Neural_Network(X,Y,input_shape,n_classes,layers_list,activation_functions, weight_initializer,learning_rate,num_iterations,lambd,drop_rate, sgd)
    
    parameters, costs =  my_model_drop.train()
    
    y_pred, y_pred_scores = my_model_drop.predict(X_test,parameters)
    print(y_pred.shape, test_y.shape)
    print('Test Accuracy : ',accuracy(y_pred, test_y))
    
    y_pred, y_pred_scores = my_model_drop.predict(X_train,parameters)
    print(y_pred.shape, train_y.shape)
    print('Train Accuracy : ',accuracy(y_pred, train_y))
    
    y = np.array([1,2,3,4])
    y.shape
    
    z = np.reshape(y,(y.shape[0],1))
    z.shape
    

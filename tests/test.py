#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:50:17 2022

@author: saurabhraj
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import deep_NN as dnn
sys.path.insert(0,"/Users/saurabhraj/Desktop/CITY AI/Programming & Maths/INM702_CW_MNIST/src/lab")



#without regularization

layers_list = [784, 256,64, 10]
activation_functions = ['relu','relu','relu','softmax']
weight_initializer = 'he'
learning_rate = 0.1
num_iterations = 100
lambd = 0
drop_rate=0

sgd = False

my_model = dnn.My_Neural_Network(layers_list,activation_functions, weight_initializer,learning_rate,num_iterations,lambd,drop_rate, sgd)

def test_suit_gradient_descent():

    parameters = {'W1': np.array([[ 1.63535156, -0.62320365, -0.53718766],
                          [-1.07799357,  0.85639907, -2.29470142]]),
                  'b1': np.array([[ 1.74604067],
                          [-0.75184921]]),
                  'W2': np.array([[ 0.32171798, -0.25467393,  1.46902454],
                          [-2.05617317, -0.31554548, -0.3756023 ],
                          [ 1.1404819 , -1.09976462, -0.1612551 ]]),
                  'b2': np.array([[-0.88020257],
                          [ 0.02561572],
                          [ 0.57539477]])}
    grads = {'dW1': np.array([[-1.10061918,  1.14472371,  0.90159072],
                      [ 0.50249434,  0.90085595, -0.68372786]]),
              'db1': np.array([[-0.12289023],
                      [-0.93576943]]),
              'dW2': np.array([[-0.26788808,  0.53035547, -0.69166075],
                      [-0.39675353, -0.6871727 , -0.84520564],
                      [-0.67124613, -0.0126646 , -1.11731035]]),
              'db2': np.array([[0.2344157 ],
                      [1.65980218],
                      [0.74204416]])}
    learning_rate = 0.01

    params = my_model.gradient_descent(parameters, grads, learning_rate)
    W1 = params['W1']
    W2 = params['W2']
    b1 = params['b1']
    b2 = params['b2']
  
    W1_0 = parameters['W1']
    W2_0 = parameters['W2']
    b1_0 = parameters['b1']
    b2_0 = parameters['b2']




    assert W1.shape == W1_0.shape
    assert b1.shape == b1_0.shape
    assert W2.shape == W2_0.shape
    assert b2.shape == b2_0.shape
    
    assert (np.round(np.sum(W1),4) == np.round(np.sum(np.array([[ 1.64635775, -0.63465089, -0.54620357],[-1.08301851,  0.84739051, -2.28786414]])),4))
    assert (np.round(np.sum(b1),4) == np.round(np.sum(np.array([[ 1.74726957],[-0.74249152]])),4))
    assert (np.round(np.sum(W2),4) == np.round(np.sum(np.array([[  0.32439686, -0.25997748,  1.47594115], [-2.05220563, -0.30867375, -0.36715024 ], [ 1.14719436, -1.09963797, -0.150082 ]])),4))
    assert (np.round(np.sum(b2),4) == np.round(np.sum(np.array([[-0.88254673], [ 0.0090177], [ 0.56797433]])),4))



    print("gradient_descent working CORRECTLY.")

  
#test_suit_gradient_descent()

def test_initialize_params():
    layers_list = [64,32,16,10]
    methods = ['trunc_normal','he','xavier','zeros','random']
    W1_shape, b1_shape = (32,64), (32,1)
    W2_shape, b2_shape = (16,32), (16,1)
    W3_shape, b3_shape = (10,16), (10,1)
    for method in methods:
      params = my_model.initialize_params(layers_list, method='random')
      W1,b1 = params['W1'], params['b1']
      W2,b2 = params['W2'], params['b2']
      W3,b3 = params['W3'], params['b3']
      #print(b2.shape)
      assert len(params) == (len(layers_list)-1)*2
      assert W1.shape == (32,64)
      assert b1.shape == (32,1)
      assert W2.shape == (16,32)
      assert b2.shape == (16,1)
      assert W3.shape == (10,16)
      assert b3.shape == (10,1)

      print(f"{method} weight initialzation working CORRECTLY")
#test_initialize_params()   

def test_linear_activation():
    W = np.array([[ 0.50288142, -1.24528809, -1.05795222]])
    b = np.array([[-0.90900761]])
    A = np.array([[-0.41675785, -0.05626683],
        [-2.1361961 ,  1.64027081],
        [-1.79343559, -0.84174737]])
    activation_functions = ['relu','sigmoid']

    (A_current, memory_current) = my_model.linear_activation(W,b,A,'sigmoid')
    assert np.round(np.sum(A_current),4) == np.round(np.sum(np.array([[0.96890023, 0.11013289]])),4)
    (A_current, memory_current) = my_model.linear_activation(W,b,A,'relu')
    assert np.round(np.sum(A_current),4) == np.round(np.sum(np.array([[3.43896131, 0.        ]])),4)

    print("linear_activation working CORRECTLY")

#test_linear_activation()


def test_forward_propagation():
    A0 = np.array([[-0.31178367,  0.72900392,  0.21782079, -0.8990918 ],
        [-2.48678065,  0.91325152,  1.12706373, -1.51409323],
        [ 1.63929108, -0.4298936 ,  2.63128056,  0.60182225],
        [-0.33588161,  1.23773784,  0.11112817,  0.12915125],
        [ 0.07612761, -0.15512816,  0.63422534,  0.810655  ]])
    
    params = {'W1': np.array([[ 0.35480861,  1.81259031, -1.3564758 , -0.46363197,  0.82465384],
                    [-1.17643148,  1.56448966,  0.71270509, -0.1810066 ,  0.53419953],
                    [-0.58661296, -1.48185327,  0.85724762,  0.94309899,  0.11444143],
                    [-0.02195668, -2.12714455, -0.83440747, -0.46550831,  0.23371059]]),
              'b1': np.array([[ 1.38503523],
                    [-0.51962709],
                    [-0.78015214],
                    [ 0.95560959]]),
              'W2': np.array([[-0.12673638, -1.36861282,  1.21848065, -0.85750144],
                    [-0.56147088, -1.0335199 ,  0.35877096,  1.07368134],
                    [-0.37550472,  0.39636757, -0.47144628,  2.33660781]]),
              'b2': np.array([[ 1.50278553],
                    [-0.59545972],
                    [ 0.52834106]]),
              'W3': np.array([[ 0.9398248 ,  0.42628539, -0.75815703]]),
              'b3': np.array([[-0.16236698]])}
             
    activation_functions = ['relu','relu','relu','sigmoid']
    A_last, memory = my_model.forward_propagation(A0, params, activation_functions)
    assert np.round(np.sum(A_last),4) == np.round(np.sum(np.array([[0.03921668, 0.70498921, 0.19734387, 0.04728177]])),4)
    print("forward_propagation working CORRECTLY")


#test_forward_propagation()

def test_linear_backward():
    dA = np.array([[-0.41675785, -0.05626683]])
    #Z,W,b,A
    # memory[0] = A_prev, W,b
    # memory[1] = 
    memory = (
              (np.array([[-2.1361961 ,  1.64027081],
                      [-1.79343559, -0.84174737],
                      [ 0.50288142, -1.24528809]]),
              np.array([[-1.05795222, -0.90900761,  0.55145404]]),
              np.array([[2.29220801]])),
              np.array([[ 0.04153939, -1.11792545]])
              )
    activation_function = ['relu','relu','relu','sigmoid']
    dA_previous, dW, db = my_model.linear_backward(dA, memory, 'sigmoid')
    assert np.round(np.sum(dA_previous),4) == np.round(np.sum(np.array([[ 0.11017994,  0.01105339],[ 0.09466817,  0.00949723],[-0.05743092, -0.00576154]])),4)
    assert np.round(np.sum(dW),4) == np.round(np.sum(np.array([[ 0.10266786,  0.09778551, -0.01968084]])),4)
    assert np.round(np.sum(db),4) == np.round(np.sum(np.array([[-0.05729622]])),4)
    dA_previous, dW, db = my_model.linear_backward(dA, memory, 'relu')
    assert np.round(np.sum(dA_previous),4) == np.round(np.sum(np.array([[ 0.44090989,  0.        ],[ 0.37883606,  0.],[-0.2298228,   0.        ]])),4)
    assert np.round(np.sum(dW),4) == np.round(np.sum(np.array([[ 0.44513824,  0.37371418, -0.10478989]])),4)
    assert np.round(np.sum(db),4) == np.round(np.sum(np.array([[-0.20837892]])),4)
    print('linear_backward working CORRECTLY')


test_initialize_params()
test_linear_activation()
test_suit_gradient_descent()
test_forward_propagation()
test_linear_backward()
test_linear_backward()

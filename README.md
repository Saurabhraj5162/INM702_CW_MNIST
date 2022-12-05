# Deep Neural Network (from SCRATCH)## AIM: The aim of this project is to develop a multi layered neural Network from scratch. In order to achieve this, NUMPY has been extensively used with propervectorization techniques.## Implementaion:There is a .ipynb file and a .py file as well for the solution in the directorysrc/lab/.There are two classes:1. My_Neural_Network2. Data### 1. My_Neural_Network:class My_Neural_Network():  - def sigmoid(self,Z):         return 1/(1+np.exp(-Z))  - def sigmoid_derivative(self,dA,Z):        return dA*dZl  - def relu(self,Z):        return np.maximum(0,Z)  - def relu_derivative(self,dA,Z):        return dA*dZ  - def softmax(self,Z):        return np.divide(e,e_total)  - def initialize_params(self, layers_list, method):        return params  - def linear_activation(self, W,b,A,activation_function):      return (A_current, memory_current)  - def forward_propagation(self, A0, params, activation_functions):        return A_last, memory  - def calculate_cost(self, A_last, Y):        return cost  - def calculate_cost_L2_regularised(self, A_last, Y, parameters, lambd):        return cost  - def linear_backward(self,dA, memory, activation_function):        return dA_previous, dW, db  - def backward_propagation(self,A_last, Y, memory, activation_functions):        return grads  - def backward_propagation_L2_regularised(self, A_last, Y, memory, activation_functions, lambd):        return grads  - def gradient_descent(self,params, grads, learning_rate):       return updated_params  - def initialize_adam_moments(self, parameters):       return v,s  - def adam(self,params, grads, v, s,adam_counter, learning_rate, beta1, beta2,  epsilon):       return updated_params, v, s, v_final, s_final  - def train(self,X,Y,print_cost=True):        return params, costs  - def predict(self,input_x,final_params):        return y_pred, y_pred_scores  - def accuracy(self,y_pred, y_actual):        return acc### 2. DATA:class Data():  - def visualize(self, rows, cols): To plot the input data (MNIST numbers).     - def getInfo(self, train_x_orig, train_y, test_x_orig, test_y): To get the           informations such as shapes of train, test data.  - def preprocess_data(self,train_x_orig, train_y, test_x_orig, test_y, n_classes):          To reshape, scale, transpose, etc the X data and one hot encode the Y data.    return (X_train, y_train, X_test, y_test)    ## How to train a model?To train a model with help of these classes, you first need to import data. Then you can use Data class object to visualize and preprocess the data such thatX data shape :  (dimensions , number of samples)Y data shape :  (number of distinct labels, number of samples)Then we need to define the following parameters:* X = X_train* Y = y_train* input_shape = X_train.shape[0]* n_classes = len(np.unique(train_y))* layers_list = a list of number of neurons for each layer. For eg. if I want to            create 3 layered model with the hidden neurons of size n1 and n2,             then I will declare:            layers_list = [input_shape, n1,n2, n_classes]            * activation_functions = a list of activation functions corresponding to each             layer in layers_list. For the above layer list, we can declare:             activation_functions = ['relu','relu','relu','softmax']            Note: If using Dropout, add '+dropout' as a suffix of activation.            E.g : ['relu','relu+dropout','relu+dropout','softmax']            Note: This code add dropout to all the hidden layers, so please            add '+dropout' to all hidden activations.* weight_initializer = The method to initialize weights. Can be choosen from any            of these - 'he'/'xavier'/'random'/'trunc_normal'/'zeros'.* learning_rate = 0.1* epochs = (iterations for which we want to train)* lambd = the regularization parameter for L2.             Note: If you do not want to use regularization, put lambd = 0.drop_rate = the dropout rate for dropout layer.            Note: If you do not want to use dropout, put drop_rate = 0.sgd = If you want to train with Stochastic gradient descent then put sgd = True            else if you want to train with normal gradient descent, put sgd = False.optimizer = "gd" or "adam"Then we can create our model class object as:* my_model = My_Neural_Network(layers_list,activation_functions,                                 weight_initializer,learning_rate,                                lambd,drop_rate, sgd)                                Then we can train our model and store trained parameters and costs by calling:* parameters, costs =  my_model.train(X_train, y_train,epochs, optimizer)Using the costs, we can plot our cost vs epoch curve.Then you can predict the labels using my_model.predict(X_test,parameters).The accuracy can be found using my_model.accuracy(y_pred, test_y).Note: the "test_y" is the y data without one hot encoding.## Test Suit:There are functions in tests directory to test whether different layers are working correctly or not. To test, you can run the test.py file.                     
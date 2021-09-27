"""
Main method of my study on Neural Networks.
The objective of this study is to develop a network from scratch.
"""

import numpy as np

def sigmoid(x):
    """
    Sigmoid function. Returns the value of x apllied to the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    """
    Derivative of the sigmoid function.
    """
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    """
        Mean Squared Error function.
        y_true and y_pred are numpy arrays of the same length.
    """
    return ((y_true - y_pred) ** 2).mean()

class Neuron:
    """
    Class that generates a single Neuron to the network
    """
    def __init__(self, weights, bias):
        """
        weights:
            Np.array.
            Mathematical weight to the neurons.
        bias:
            Integer.
            Disproportionate weight in favor of or against the ideia.
        """
        self.weights = weights
        self.bias = bias
    
    def feedforward(self, inputs, func=sigmoid):
        """
        inputs:
            Np.array.
            Values to be calculated.
        func:
            Function that will be used.
            By default we are using the sigmoid function.
        """
        total = np.dot(self.weights, inputs) + self.bias
        return func(total)

class OurNeuralNetwork:
    # @TODO Make this generic
    '''
        A neural network with:
            2 inputs
            a hidden layer with 2 neurons (h1, h2)
            an output layer with 1 neuron (o1)
        Each neuron has the same weights and bias:
            w = [0, 1]
            b = 0
    '''
    def __init__(self,):
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x, func=sigmoid):

        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        '''
            data is a (n x 2) numpy array, n = # of samples in the dataset.
            all_y_trues is a numpy array with n elements.
            Elements in all_y_trues correspond to those in data.
        '''
        learn_rate = 0.1
        epochs = 1000 #num of times it will loop

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                #Calculate partial derivatives
                d_L_d_ypred = -2 * (y_true - y_pred)

                #Neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                #Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                #Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                #Update weights and biases
                #Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                #Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                #Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

                #Calculate total loss at the end of each epoch
                if epoch % 10 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, 1, data)
                    loss = mse_loss(all_y_trues, y_preds)
                    print("Epoch %d loss: %.3f" % (epoch, loss))

#Define dataset
data = np.array([
    [-2, -1],
    [25, 6],
    [17,4],
    [-15,-6]
])

all_y_trues = np.array([
    1,
    0,
    0,
    1
])

network = OurNeuralNetwork()
network.train(data, all_y_trues)
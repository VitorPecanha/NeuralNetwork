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
        weights = np.array([0,1])
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x, func=sigmoid):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        out_o1 = self.o1.feedforward([out_h1, out_h2])

        return out_o1

network = OurNeuralNetwork()
x = np.array([2,3])
print(network.feedforward(x))
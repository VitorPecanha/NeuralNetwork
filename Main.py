"""
Método main do estudo de Redes Neurais.
O objetivo desse projeto é desenvolver uma rede neural do zero.
"""

import numpy as np

def sigmoid(x):
    """
    Função sigmoidal. Retorna o valor de x aplicado na função sigmoid.
    x: 
        Inteiro obrigatório.
        Valor que será calculado pela função.
    """
    return 1 / (1 + np.exp(-x))

class Neuron:
    """
    Classe que gera um neurônio para a rede neural
    """
    def __init__(self, weights, bias):
        """
        Classe que inicializa o neurônio.
        weights:
            Np.array.
            Peso dos neurônios.
        bias:
            Inteiro.
            Valor do enviesamento.
        """
        self.weights = weights
        self.bias = bias
    
    def feedforward(self, inputs, func=sigmoid):
        """
        Função de alimentação do neurônio.
        inputs:
            Np.array.
            Valor que será calculado.
        func:
            Função que será usada para os calculos.
            Por padrão é a função sigmoid.
        """
        total = np.dot(self.weights, inputs) + self.bias
        return func(total)

class OurNeuralNetwork:
    '''
        A neural network with:
            - 2 inputs
            - a hidden layer with 2 neurons (h1, h2)
            - an output layer with 1 neuron (o1)
        Each neuron has the same weights and bias:
            - w = [0, 1]
            - b = 0
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
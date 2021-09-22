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

weights = np.array([0, 1]) # w1 = 0, w2 = 1
bias = 4                   # b = 4
n = Neuron(weights, bias)

x = np.array([2, 3])       # x1 = 2, x2 = 3
print(n.feedforward(x, sigmoid))
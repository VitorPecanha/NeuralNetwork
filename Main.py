"""
Método main do estudo de Redes Neurais.
O objetivo desse projeto é desenvolver uma rede neural do zero.
"""

import numpy as np

def sigmoid(x):
    """
    Função sigmoidal. Retorna o valor de x aplicado na função sigmoid.
    x: inteiro obrigatório
    """
    return 1 / (1 + np.exp(-x))

print(sigmoid(7))
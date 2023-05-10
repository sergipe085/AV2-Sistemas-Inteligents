import numpy as np

class neuron: 
    def __init__(self, size):
        self.size = size
        # limiar = -100
        # self.weights = np.concatenate(([limiar], np.random.rand(size)))
        self.weights = np.random.rand(size + 1)

    def get_sum(self, x_treino):
        i = 0
        _sum = 0
        while i < self.size:
            _input = 0
            if (x_treino[i] != "?"):
                _input = float(x_treino[i])
            _sum += _input*self.weights[i]
            i += 1

        return _sum
    
    def get_output(self, x_treino, activation_function):
        _sum = self.get_sum(x_treino);
        _out = activation_function(_sum)
        return _out;

    def fix_weights(self, new_weights):
        self.weights = new_weights
    def get_weights(self):
        return self.weights
        


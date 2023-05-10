from neuron import neuron
import numpy as np 

class perceptron:
    def __init__(self, neuron_size, neuron_count, activation_function) -> None:
        neurons = []
        for i in range(neuron_count):
            neurons.append(neuron(neuron_size))

        self.neurons = neurons
        self.neuron_count = neuron_count
        self.activation_function = activation_function

    def fix_inputs(self, x_treino):
        i = 0
        _inputs = []
        while i < len(x_treino):
            _input = x_treino[i]
            if (_input == "?"):
                _input = "0"
            _input = float(_input)
            _inputs.append(_input)
            i+=1
        _inputs = _inputs / np.linalg.norm(_inputs)
        _inputs = np.concatenate(([-1], _inputs))
        bp = 1
        return _inputs

    def train(self, x_treino, y_treino, max_epochs, learning_rate):

        YTreino = y_treino


        erro = True
        j = 0
        while erro == True and j < max_epochs:
            erro = False
            j += 1
            print(j)
            i = 0
            while i < len(x_treino):
                _index = int(YTreino[i]) - 1

                XTreino = self.fix_inputs(x_treino[i])
                outputs = []
                for _ne in self.neurons:
                    outputs.append(_ne.get_output(XTreino, self.activation_function))
                expected_outputs = [-1, -1, -1, -1, -1]
                expected_outputs[_index] = 1

                errors = []
                for _i in range(self.neuron_count):
                    errors.append(expected_outputs[_i] - outputs[_i])
                    if (expected_outputs[_i] != outputs[_i]):
                        erro = True

                k = 0;
                while k < len(self.neurons):
                    _neuron = self.neurons[k]
                    new_weights = _neuron.get_weights() + learning_rate * errors[k] * XTreino 
                    _neuron.fix_weights(new_weights)
                    k += 1


                i+=1
        bp = 2

    def get_output(self, X):
        _X = self.fix_inputs(X)
        output = []
        for ne in self.neurons:
            output.append(ne.get_output(_X, self.activation_function))

        return output

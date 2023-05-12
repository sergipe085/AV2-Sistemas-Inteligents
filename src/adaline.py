from neuron import neuron
import numpy as np 

class adaline:
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

    def train(self, x_treino, y_treino, max_epochs, learning_rate, precision):
        YTreino = y_treino

        j = 0
        EQMs_list = []
        while  j < max_epochs:
            j += 1
            i = 0
            p = len(x_treino)
            while i < p:
                _index = int(YTreino[i]) - 1

                XTreino = self.fix_inputs(x_treino[i])
                outputs = []
                expected_outputs = [-1, -1, -1, -1, -1]
                expected_outputs[_index] = 1
                EQMs = [0, 0, 0, 0, 0]
                EQMs_anterior = [0, 0, 0, 0, 0]
                k = 0
                for _ne in self.neurons:
                    u = _ne.get_sum(XTreino)
                    d = expected_outputs[k]
                    EQMs[k] += (d - u) ** 2
                    outputs.append(u)
                    k+=1

                k = 0;
                while k < len(self.neurons):
                    _neuron = self.neurons[k]
                    new_weights = _neuron.get_weights() + learning_rate * (expected_outputs[k] - outputs[k]) * XTreino 
                    _neuron.fix_weights(new_weights)
                    k += 1


                i+=1

            finish = True
            for h in range(len(EQMs)):
                EQMs[h] = EQMs[h] / p

                if (EQMs[h] - EQMs_anterior[h] > precision):
                    finish = False

                EQMs_anterior[h] = EQMs[h]
            EQMs_list.append(EQMs)
            
            bp = 1
            if (finish == True): 
                break
        return EQMs_list
        bp = 2

    def get_output(self, X):
        _X = self.fix_inputs(X)
        output = []
        for ne in self.neurons:
            output.append(ne.get_output(_X, self.activation_function))

        return output

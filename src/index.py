import numpy as np 
import matplotlib.pyplot as plt
from data_reader import get_data;
from neuron import neuron
from perceptron import perceptron
from adaline import adaline
import utils as ut

XTreino, YTreino, XTeste, YTeste = get_data()

def binary_activation_function(_i):
    
    if (_i >= 0):
        return 1
    else:
        return -1
def execute_perceptron():
    
    pe = perceptron(36, 5, binary_activation_function)
    pe.train(XTreino, YTreino, 1, 0.001)

    i = 0
    correct_count = 0
    total = len(XTeste)
    while i < total:
        output = pe.get_output(XTeste[i])
        expected_out_index = int(YTeste[i]) - 1
        expected_output = [-1, -1, -1, -1, -1]
        expected_output[expected_out_index] = 1
        
        j = 0
        certo = True
        while j < len(output):
            if (output[j] != expected_output[j]):
                certo = False
            j += 1

        if (certo == True):
            correct_count += 1

        bp = 2
        i+=1


    print(f"Acuracia: {correct_count * 100/total}")
    bp = 3

def execute_adaline():
    print("---Iniciando a rede ADALINE---")

    ada = adaline(36, 5, binary_activation_function)
    _, training_time = ut.execution_time(lambda: ada.train(XTreino, YTreino, 10, 0.01, 0.000000001))
    print(f"Tempo de treinamento: {training_time}")

    i = 0
    correct_count = 0
    total = len(XTeste)
    while i < total:
        output = ada.get_output(XTeste[i])
        expected_out_index = int(YTeste[i]) - 1
        expected_output = [-1, -1, -1, -1, -1]
        expected_output[expected_out_index] = 1
        
        j = 0
        certo = True
        while j < len(output):
            if (output[j] != expected_output[j]):
                certo = False
            j += 1

        if (certo == True):
            correct_count += 1

        bp = 2
        i+=1

    acuracia = correct_count * 100 / total
    print(f"Acuracia: {correct_count * 100/total}")
    return acuracia

execute_adaline()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

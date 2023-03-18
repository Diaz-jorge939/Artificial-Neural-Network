import pickle
import os

class Node():
    def __init__(self):
        self.collector = None
        self.connections = []

class ANN():
    def __init__(self, input_layer, hidden_layer, output_layer):
        
        self.input_layer = [Node() for index in range(input_layer)]         # list comprehension adding a Node() at each index of hidden_layer list 
        self.hidden_layer = [Node() for index in range(hidden_layer)]
        self.output_layer = [Node() for index in range(output_layer)]

        self.network = [self.input_layer, self.hidden_layer, self.output_layer]

    def add_inputs(self, input_vals):
        if len(input_vals) == len(self.input_layer):   
            i = 0
            for val in input_vals:
                self.input_layer[i].collector = val
                i += 1
        else: 
            print("invald size")

    def add_connections(self):
        pass

 
with open('input.txt', 'r') as read_file:
    input_file = read_file.read()

inputs = input_file.split(',')

print(input_file)
print(inputs)

net = ANN(4,2,1)

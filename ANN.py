import pickle
import os

class Node():
    def __init__(self):
        self.collector = 0
        self.connections = []

class ANN():
    def __init__(self, input_layer, hidden_layer, output_layer):
        
        self.input_layer = [Node() for index in range(input_layer)]      # list comprehension adding a Node() at each index of hidden_layer list 
        self.hidden_layer = [Node() for index in range(hidden_layer)]
        self.output_layer = [Node() for index in range(output_layer)]

        self.network = [self.input_layer, self.hidden_layer, self.output_layer]     # network contains 3 lists at each index 

    def add_inputs(self, input_vals):
        if len(input_vals) == len(self.input_layer):   
            i = 0
            for val in input_vals:
                self.input_layer[i].collector = val
                i += 1
        else: 
            print("invald size")

    def add_connections(self,layer1, layer2):
        for i in range(len(layer1)):
            for j in range(len(layer2)):     
                layer1[i].connections.append(layer2[j])      
                
    def sum_collector(self):
        # for layer_index in range(len(self.network)-1):             # iterating through each layer
        #     for node in self.network[layer_index]:              # for each node in layer
        #         for i in range(len(node.connections)):
        #             node.connections[i].collector = node.connections[i].collector + node.collector
        for layer in self.network:
            for node in layer:
                for connected_node in node.connections:
                    connected_node.collector = connected_node.collector + node.collector

if __name__ == '__main__':
    with open('Assignment 2/input.txt', 'r') as read_file:
        input_file = read_file.read()

    inputs = input_file.split(',')
    
    # print(input_file)
    # print(inputs)
    
    new_inputs = [float(val) for val in inputs]

    net = ANN(4,2,1)

    net.add_inputs(new_inputs)

    net.add_connections(net.input_layer, net.hidden_layer)
    net.add_connections(net.hidden_layer,net.output_layer)


    net.sum_collector()

    for layer in net.network:
        for node in layer:
            print(node.collector)

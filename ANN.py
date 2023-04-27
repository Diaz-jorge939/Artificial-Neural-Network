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

    def add_inputs(self, input_vals):
        if len(input_vals) == len(self.input_layer):   
            i = 0
            for val in input_vals:
                self.input_layer[i].collector = val
                i += 1
        else: 
            print("invald size")

    # making the connections from outputlayer -> hiddenlayer-> inputlayer
    def make_connections(self):
        
        or output_node in self.output_layer:
            for hidden_node in self.hidden_layer:
                for input_node in self.input_layer:
                    hidden_node.connections.append(input_node)  #hidden layer connections   
                output_node.connections.append(hidden_node)     #output layer connections
                
    
    def feed_forward(self):
        
        for output_node in self.output_layer:
            for hidden_node in output_node.connections:
                for input_node in hidden_node.connections:
                    hidden_node.collector += input_node.collector       #summation of input layer
                output_node.collector += hidden_node.collector          #summation of hidden_layer 

                    
if __name__ == '__main__':
    
    with open('input.txt', 'r') as read_file:
        input_file = read_file.read()

    inputs = input_file.split(',')

    new_inputs = [float(val) for val in inputs]

    # print(input_file)
    # print(inputs)

    net = ANN(4,2,1)

    net.add_inputs(new_inputs)
    
    net.make_connections()
    
    # net.sum_collector()
    
    net.feed_forward()

    #prints output node collector
    for node in net.output_layer:
        print(node.collector)

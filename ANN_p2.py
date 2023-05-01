import pandas as pd
import pickle
import os
import random
import math

class Node():
    def __init__(self):
        self.collector = 0
        self.connections = []

class ANN():
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        
        self.input_layer = [Node() for index in range(input_nodes)]      # list comprehension adding a Node() at each index of hidden_layer list 
        self.hidden_layer = [Node() for index in range(hidden_nodes)]
        self.output_layer = [Node() for index in range(output_nodes)]


        self.weights ={'hidden_input': [random.random() for i in range(input_nodes * hidden_nodes)],
                'output_hidden': [random.random() for i in range(hidden_nodes)]}
        #randomized weights
        # self.weights ={
        #         'hidden_input': [.1,.2,.2,.1,.1,.1,.1,.1],
        #         'output_hidden': [.3,.5]
        # }
        # print(f'weights initialized: {self.weights}')
        self.bias = 1.0
        
        self.lr = learning_rate

        
    """
    Args:
        df: pandas dataframe
        row: specified row used to set input node collector values
    """
    def set_inputs(self, df,row):
        
        try:   
            # Select a row up to the last column
            # last column represents expected output

            i=0
            for val in df.iloc[row, :-1]:
                self.input_layer[i].collector = val
                i += 1
        except: 
            print("invalid size")

    # use list indexing to feed forward and connections to back propagation 
    def make_connections(self):

        for output_node in self.output_layer:
            for hidden_node in self.hidden_layer:
                for input_node in self.input_layer:
                    hidden_node.connections.append(input_node)
                output_node.connections.append(hidden_node)
                
    def forward_propagation(self):
        outputs = []
        for output_node in self.output_layer:
            j=0
            i = 0
            for hidden_node in output_node.connections:
                for input_node in hidden_node.connections:
                    # X = Summation(input_node i  * hidden_node j) + bias
                    hidden_node.collector += input_node.collector * self.weights['hidden_input'][i]
                    i+=1
                #hidden_node.collector + bias
                # print(f'hidden_node {j}: {hidden_node.collector}')
                hidden_node.collector = self.sigmoid(hidden_node.collector)
        
                # print(f'hidden_output {j}: {hidden_node.collector}')

                # print(f'output_node: {hidden_node.collector} x {self.weights["output_hidden"][j]}')

                output_node.collector += hidden_node.collector * self.weights['output_hidden'][j] # sigmoid(hidden_node) * output_node Weight
                # print(output_node.collector)
                
                j +=1

            # print(f'output layer node collector send this value to sigmoidal: {output_node.collector}')
            output_node.collector = self.sigmoid(output_node.collector)

            # print(f'final output {output_node.collector}')
            outputs.append(output_node.collector)
        
        #network currently designed for network with 1 output node
        # more output nodes -> more outputs in list -> return outputs
        return outputs[0]

    def sigmoid(self, neuron):
        return 1.0 / (1.0 + math.exp(-neuron))
    
    # Calculate the derivative of an neuron output
    # sigmoid(x)(1-sigmoid(x))
    def transfer_derivative(self, output):
        return output * (1.0 - output)

    def back_propogate(self, error):
        for output_node in self.output_layer:
            k=0
            i = 0
            output_node_error = error * self.transfer_derivative(output_node.collector)
            for hidden_node in output_node.connections:
                self.adjust_weights('output_hidden',k, output_node_error, hidden_node.collector)
                hidden_node_error = (self.weights['output_hidden'][k] * output_node_error) * self.transfer_derivative(hidden_node.collector)
                for input_node in hidden_node.connections:
                    self.adjust_weights('hidden_input', i, hidden_node_error, input_node.collector)
                    # self.adjust_weights('hidden_input', i, hidden_error)
                    i+=1
                k+=1
        # print(self.weights)
        
    def adjust_weights(self, layer, index, gradient_error, collector):
        # print(f'adjusting weights {layer}, {index}, {gradient_error} : {self.weights[layer][index]}')
        self.weights[layer][index] = self.weights[layer][index] - self.lr * gradient_error * collector
        # print(f'new weight: {self.weights[layer][index]}')
        
    def train_network(self, df, n_epochs, n_outputs):
        for epoch in range(n_epochs):
            error_sum = 0.0
            for i in range(n_outputs):
                #set inputs 
                row = i
                self.set_inputs(df, row)
                output = self.forward_propagation()

                error = (df['expected'][row] - output)**2
                error_sum+= error

                self.back_propogate(error)

            if error_sum/(epoch+1) <= 0.05:
                print("Target Error Reached error=%.3f" % ( error_sum/(epoch+1)))
                return

            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.lr, error_sum/(epoch+1)))


if __name__ == '__main__':
    
    lr = 0.01

    net = ANN(4,2,1,lr)
    
    
    df = pd.read_csv('dataset.csv', header=None)
    
    #converting all df to float value
    df = df.astype(float)

    #creating column names for dataset
    df.columns = ['input1', 'input2', 'input3', 'input4','expected']
    n_outputs = len(df)-1
    n_epochs = 200

    # net.set_inputs(df,15)
    
    net.make_connections()

    net.train_network(df, n_epochs, n_outputs)

    # output = net.forward_propagation()
    # expected_output = df['expected'][15] #expected output of current row
   
    # error = (output - expected_output)
    

    # net.back_propogate(error)

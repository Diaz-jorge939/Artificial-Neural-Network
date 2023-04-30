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


        # 'hidden_output': [random.random() for i in range(input_nodes * hidden_nodes)],
        #         'output_hidden': [random.random() for i in range(hidden_nodes)]
        #randomized weights
        self.weights ={
                'hidden_output': [.1,.2,.2,.1,.1,.1,.1,.1],
                'output_hidden': [.3,.5]
        }
        print(f'weights initialized: {self.weights}')
        self.bias = 1.0
        
        self.lr = learning_rate

        
    """
    Args:
        df: pandas dataframe
        row: specified row used to set input node collector values
    """
    def set_inputs(self, row):
        
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
                    hidden_node.collector += input_node.collector * self.weights['hidden_output'][i]
                    i+=1
                #hidden_node.collector + bias
                output = self.sigmoid(hidden_node.collector)
                # print(f'hidden_node {j}: {hidden_node.collector}')
                # print(f'hidden_output {j}: {output}')

                # print(f'output_node: {output} x {self.weights["output_hidden"][j]}')

                output_node.collector += output * self.weights['output_hidden'][j]
                # print(output_node.collector)
                
                j +=1

            output = self.sigmoid(output_node.collector)

            # print(f'output layer node collector send this value to sigmoidal: {output_node.collector}')
            # print(f'final output {output}')
            outputs.append(output)
        
        return outputs[0]

    def sigmoid(self, activation):
        return 1.0 / (1.0 + math.exp(-activation))
    
    # Calculate the derivative of an neuron output
    def transfer_derivative(self, output):
        return output * (1.0 - output)
    def back_propogate(self,error):
        for output_node in self.output_layer:
            pass


    def train_network(self, df, n_epochs, n_outputs):
        for epoch in range(n_epochs):
            error_sum = 0
            for i in range(n_outputs):
                #set inputs 
                row = i
                self.set_inputs(df.iloc[row, :-1])
                output = self.forward_propagation()
                expected_output = df['expected'][row] #expected output of current row
                #print(output)
                #calculate error:
                print(f'expected output at row {i}: {expected_output}')
                error = expected_output - output

                # if error < 0.05:
                #     print('small ahh error')
                #     return
                error_sum += error
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.lr, error_sum))


if __name__ == '__main__':
    
    lr = 0.01

    net = ANN(4,2,1,lr)
    
    
    df = pd.read_csv('dataset.csv', header=None)
    
    #converting all df to float value
    df = df.astype(float)

    #creating column names for dataset
    df.columns = ['input1', 'input2', 'input3', 'input4','expected']
    
    # print(weights)
    # print(weights['hidden_output'][0])
    # print(df)

    # for i in df.iloc[0,:-1]:
    #     print(i)
    
    net.set_inputs(df,15)

    # for i in net.input_layer:
    #     print(i.collector)
    
    net.make_connections()

    net.feed_forward(df,15)

import pandas as pd
import pickle
import os

class Node():
    def __init__(self):
        self.collector = 0
        self.connections = []

class ANN():
    def __init__(self, input_layer, hidden_layer, output_layer, learning_rate):
        
        self.input_layer = [Node() for index in range(input_layer)]      # list comprehension adding a Node() at each index of hidden_layer list 
        self.hidden_layer = [Node() for index in range(hidden_layer)]
        self.output_layer = [Node() for index in range(output_layer)]

        self.weights = []
        
        self.lr = learning_rate
    """
    Args:
        df: pandas dataframe
        row: specified row used to set input node values
    """
    def set_inputs(self, df, row):
        
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
                
    def feed_forward(self):
        
        for output_node in self.output_layer:
            for hidden_node in output_node.connections:
                for input_node in hidden_node.connections:
                    hidden_node.collector += input_node.collector       #summation of input layer
                output_node.collector += hidden_node.collector          #summation of hidden_layer

    def back_propogate():
        pass

    def train_network():
        pass

if __name__ == '__main__':
    df = pd.read_csv('dataset.csv', header=None)
    
    #converting all df to float value
    df = df.astype(float)

    #creating column names for dataset
    df.columns = ['input1', 'input2', 'input3', 'input4','expected']
    

    # print(df)

    # for i in df.iloc[0,:-1]:
    #     print(i)
    
    lr = 0.01

    net = ANN(4,2,1,lr)

    net.set_inputs(df,15)

    for i in net.input_layer:
        print(i.collector)
    
    # net.make_connections()

    # net.feed_forward()


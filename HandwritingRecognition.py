import pandas as pd
import pickle
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

        #randomized weights
        self.weights ={'hidden_input': [random.random() for i in range(input_nodes * hidden_nodes)],
                'output_hidden': [random.random() for i in range(hidden_nodes)]}
        
        self.bias = 1.0
        
        self.lr = learning_rate

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
                    hidden_node.collector += input_node.collector * self.weights['hidden_input'][i] + self.bias
                    i+=1

                hidden_node.collector = self.sigmoid(hidden_node.collector)

                output_node.collector += hidden_node.collector * self.weights['output_hidden'][j] # sigmoid(hidden_node) * output_node Weight
                
                j +=1

            output_node.collector = self.sigmoid(output_node.collector)

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

                    i+=1
                k+=1
        
    def adjust_weights(self, layer, index, gradient_error, collector):
        self.weights[layer][index] = self.weights[layer][index] - self.lr * gradient_error * collector
        
    def train_network(self, df, n_epochs, n_outputs, target_error):
        for epoch in range(n_epochs):
            error_sum = 0.0

            for i in range(n_outputs):
                 
                row = i
                #set input layer 
                j=0
                for val in df.iloc[j, 1:]:
                    self.input_layer[j].collector = val
                    j += 1
                    
                output = self.forward_propagation()

                error = (df[0][row] - output)**2
                error_sum+= error

                self.back_propogate(error)

            if error_sum/(epoch+1) <= target_error:
                print("Target Error Reached error=%.3f" % ( error_sum/(epoch+1)))
                return

            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.lr, error_sum/(epoch+1)))

        # Make a prediction with a network
    # def predict(self, row, threshold):
    #     #set input layer
    #     j=0
    #     for val in df.iloc[row, 1:]:
    #         self.input_layer[j].collector = val
    #         j += 1

    #     output = self.forward_propagate()

    #     if output > threshold:
    #         #output is greater than threshold then it guessed correctly
    #         return 1
    #     #else it guessed wrong
    #     return 0

if __name__ == '__main__':
    
    df = pd.read_csv('A_Z Handwritten Data.csv', header=None)

    print
    #filtering table by column value. 
    df = df[df[0] == 25]

    #normalizes data
    df = df.div(255.0)

    # 1000 random rows selected for training 
    training_df = df.sample(n=1000)

    #print(training_df.to_string(max_rows=10,max_cols=10))

    training_df = training_df.reset_index(drop=True)

    testing_df = df.sample(n=200)

    testing_df = testing_df.reset_index(drop=True)

    lr = 0.01
    net = ANN(784,98,1,lr)

    net.make_connections()

    net.train_network(df=training_df, n_epochs=1000, n_outputs=len(training_df), target_error=.05)

    # for row in range(len(testing_df)):
    #     prediction = net.predict(testing_df.iloc[row, 1:], 0.5)
    #     print('Expected=%d, Got=%d' % (testing_df[row][0], prediction))

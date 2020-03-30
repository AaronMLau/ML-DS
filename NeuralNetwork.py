"""
    Author: Aaron Lau
    University of Illinois at Chicago, 
    CS 412: Machine Learning, Spring '20
    
"""
#The commented variables are suggestions so change them as appropriate,
#However, do not change the __init__(), train(), or predict(x=[]) function headers
#You may create additional functions as you see fit

import numpy as np
np.random.seed(100)

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def sigmoid_derivative(p):
    return p * (1 - p)

'''
Base code from: 
    video lecture for CS 412
'''
class NeuralNetwork:
    
    #Do not change this function header
    def __init__(self,x=[[]],y=[],numLayers=2,numNodes=2,eta=0.001,maxIter=10000):
        self.data = np.append(x,np.ones([len(x),1]),1)
        self.labels = np.array(y)
        
        #for the 1st hidden lyaer with two nodes;
        #randomize starting values before trainig model with back-prop
        self.weights1 = np.random.rand(len(x)+1,2)
        #this is the first layer of the neural network and it has weights from all inputs to both internal nodes
        self.weights2 = np.random.rand(3,1)
        #these are the weights from the last two nodes, and the bias to the final neuron

        self.output1 = [] #all of the outputs from the 2 neurons in the hidden layer
        self.output2 = [] #the final output 


    def train(self):
        #one loop through the data (for now), feedforward then backprop
        #for each datapoint, feedforward and figure out the value, 
        #then give it to backprop which will calculate the sesitivities
        #to determine how it will modify the weights
        for index in range(len(x)):
            self.feedforward(self.data[index])
            self.backprop(self.data[index],self.labels[index])

    def predict(self,x=[]):
        #predict by feeding forward then backprop
        self.feedforward(np.append(x,1))
        #add the bias for feedforward
        return self.output2[0]

    def feedforward(self,point):
        #calcluate the output1 and output2
        #feedforward from front to back (sigmoid)
        self.output1 = sigmoid(np.dot(point,self.weights1))
        #for each point, returns two seperate values from the dot product
        #add the bias
        self.output1 = np.append(self.output1,1)
        self.output2 = sigmoid(np.dot(self.output1,self.weights2))


    def backprop(self,point,label):
        #calculate the sensitivities and the weights
        sensitivity2 = (label-self.output2) * sigmoid_derivative(self.output2)
        sensitivity1 = np.multiply(np.dot(sensitivity2,self.weights2.T),sigmoid_derivative(self.output1))

        change_w2 = np.array([np.multiply(self.output1,sensitivity2)]).T
        change_w1 = np.outer(point,sensitivity1)

        self.weights += change_w1
        self.weights += change_w2


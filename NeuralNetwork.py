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


'''
Base code from:
    neuralnetwork.py code provided as a base for CS 412 at UIC
class NeuralNetwork:

    #Do not change this function header
    def __init__(self,x=[[]],y=[],numLayers=2,numNodes=2,eta=0.001,maxIter=10000):
        self.data = x
        self.labels = y
        self.nLayers = numLayers
        self.nNodes = numNodes
        self.eta = eta
        self.maxIt = maxIter
        #self.weights = [np.random.rand(len(x[0]),numNodes)] #create the weights from the inputs to the first layer
        #for each of the layers
            #self.weights.append(np.random.rand(numNodes,numNodes) #create the random weights between internal layers
            #self.weights.append(np.random.rand(numNodes,1)) #create weights from final layer to output node
            #self.outputs = np.zeros(y.shape)
            #self.train()
        self.a = 1 #this is how you define a non-static variable

    def train(self):
        #Do not change this function header
        return 0.0

    def predict(self,x=[]):
        #Do not change this function header
        return 0.0

    def feedforward(self):
        #This function is likely to be very helpful, but is not necessary
        return 0.0

    def backprop(self):
        #This function is likely to be very helpful, but is not necessary
        return 0.0
'''

'''
Base code from: https://towardsdatascience.com
    /how-to-build-your-own-neural-network
    -from-scratch-in-python-68998a08e4f6
class NeuralNetwork:
    def __init__(self,x,y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(y.shape)
    
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.imput, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
'''

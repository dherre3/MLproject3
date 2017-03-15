#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 17:10:08 2017

@author: davidherrera
"""

import numpy as np
from  sklearn.metrics import log_loss
class ActivationFunctions:
    def sigmoid(self, z, Derivative=False):
        if not Derivative:
            return 1/(1+np.exp(-z))
        else:
            out = self.sigmoid(z)
            return out*(1-out)
    def softsign(self, z, Derivative=False):
        if not Derivative:
            return z/(1+np.abs(z))
        else:
            return z/(np.abs(z)*(1+np.abs(z))**2)
    def tanh(self, z, Derivative=False):
        if not Derivative:
            return np.tanh(z)
        else:
            return (4*(np.cosh(z))**2)/(np.cosh(2*z)+1)**2
        
class BackPropagationNetwork:
        
    layerCount = 0
    shape = None
    weights = []
    
    def __init__(self, layerSize):
        
        # Layer Info
        self.layerCount = len(layerSize)-1
        self.shape = layerSize
        
        #Input data from last run
        self._layerInput = []
        self._layerOutput = []
        self._previousWeightDelta = []
        
        #create the weight matrices
        for (l1,l2) in zip(layerSize[:-1],layerSize[1:]):
            high = 1/np.sqrt(l1)
            self.weights.append(np.random.uniform(low = -high,high=high, size=(l2,l1+1)))
            self._previousWeightDelta.append(np.zeros((l2,l1+1)))
    def softsign(self, z, Derivative=False):
        if not Derivative:
            return z/(1+np.abs(z))
        else:
            return 1/(1+np.abs(z))**2
        
    def sigmoid(self, z, Derivative=False):
        if not Derivative:
            return 1/(1+np.exp(-z))
        else:
            out = self.sigmoid(z)
            return out*(1-out)
    def tanh(self, z, Derivative=False):
        if not Derivative:
            return np.tanh(z)
        else:
            return (4*(np.cosh(z))**2)/(np.cosh(2*z)+1)**2
    def relu(self, z, Derivative=False):
        if not Derivative:
            return np.log(1+np.exp(z))
        else:
            return self.sigmoid(z, Derivative)
    def error(self, output, target):
        outputL = output-target
        return np.sum(outputL**2)
    def stochasticGradientDescent(self,input,targe, trainingRate = 0.2, momentum = 0.5):
           #Gradient decent tolerance and errTol.
        maxIter = 100
        errTol = 1e-5
        errorTrain = []
        errorVal = []
        for i in range(maxIter+1):
            for sample in range(newTrainX.shape[0]):
                err = bbn.trainEpoch(newTrainX[sample],trainYBi[sample],trainingRate,momentum)
                #err = bbn.batchTraining(train,target,trainingRate = 0.1,momentum = 0.5)
                print("Iteration: {0}\tError: {1:0.6f}".format(i+1,err))
                if err <= errTol:
                    print("Tolerance error reached at {0}".format(i+1))
                    break;
            OutValidation = bbn.Run(newTrainX)
            
            errorTrain.append(err);
            OutValidation = bbn.Run(newValX)
            errorVal.append(bbn.error(OutValidation,testBi))   
            
    def trainEpoch(self, input, target, trainingRate = 0.2, momentum = 0.5):
        delta=[]
        training_size = input.shape[0]
        #Run Network
        self.Run(input)
        
        #Calculate deltas
        for index in reversed(range(self.layerCount)):
            if index == self.layerCount - 1:
                #Compared to target values
                output_delta = self._layerOutput[index] - target.T
                error = np.sum(output_delta**2)
#               error =  -np.ndarray.sum((target.T*np.log(self._layerOutput[index])+(1-target.T)*np.log(1-self._layerOutput[index])))
#                print(error)
                delta.append(output_delta*self.relu(self._layerInput[index],True))
            else: 
                delta_pullback = self.weights[index+1].T.dot(delta[-1])
                delta.append(delta_pullback[:-1,:]*self.relu(self._layerInput[index], True))
        #Computer delta weights
        for index in range(self.layerCount):
            delta_index = self.layerCount - 1 - index
            
            if index == 0:
                layerOutput = np.vstack([input.T, np.ones([1,training_size])])
            else:
                layerOutput = np.vstack([self._layerOutput[index-1],np.ones([1,self._layerOutput[index - 1].shape[1]])])
            
            curWeightDelta = np.sum(layerOutput[None,:,:].transpose(2,0,1)*delta[delta_index][None,:,:].transpose(2,1,0),axis = 0)
            
            weightDelta = trainingRate * curWeightDelta + momentum * self._previousWeightDelta[index]
            
            self.weights[index] -=weightDelta
            
            self._previousWeightDelta[index] = weightDelta 
        return error
    
    def batchTraining(self,input, target,split_number = 4,trainingRate = 0.2, momentum = 0.5):
        err = 0
        for i in range(len(input)):
            err = self.trainEpoch(input[i],target[i], trainingRate, momentum)
            print("Batch Iteration: {0}\tError: {1:0.6f}".format(i+1,err))
        return err
    
    def Run(self,input):
        training_size = input.shape[0];
        #Input data from last run
        self._layerInput = []
        self._layerOutput = []
        
        for index in range(self.layerCount):
            if index==0:
                layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1,training_size])]))
            else:
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1,training_size])]))
            self._layerInput.append(layerInput)
            self._layerOutput.append(self.relu(layerInput))
        return self._layerOutput[-1].T
        
if __name__ == "__main__":
    bbn = BackPropagationNetwork((2,3,2))
    X = np.array([[0, 0], [1, 1],[0,1],[1,0]])
    Y = np.array([[1,1],[0,0],[1,0],[0,1]])
    maxIter = 100000
    errTol = 1e-5
    for i in range(maxIter+1):
        err = bbn.trainEpoch(X,Y,0.2,0.5)
        if i%10000 == 0:
            print("Iteration: {0}\tError: {1:0.6f}".format(i,err))
        if err <= errTol:
            print("Tolerance error reached at {0}".format(i))
            break;
    Out = bbn.Run(X)
    print("Input:{0}, \nOutput:{1}".format(X, Out))
    
    
    
    
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 17:10:08 2017

@author: davidherrera
"""

import numpy as np

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
            self.weights.append(np.random.normal(scale=0.1, size=(l2,l1+1)))
            self._previousWeightDelta.append(np.zeros((l2,l1+1)))
    def sigmoid(self, z, Derivative=False):
        if not Derivative:
            return 1/(1+np.exp(-z))
        else:
            out = self.sigmoid(z)
            return out*(1-out)
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
                delta.append(output_delta*self.sigmoid(self._layerInput[index],True))
            else: 
                delta_pullback = self.weights[index-1].T.dot(delta[-1])
                delta.append(delta_pullback[:-1,:]*self.sigmoid(self._layerInput[index], True))
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
            self._layerOutput.append(self.sigmoid(layerInput))
        return self._layerOutput[-1].T
        
if __name__ == "__main__":
    bbn = BackPropagationNetwork((2,2,1))
    print(bbn.shape)
    print(bbn.weights)
    X = np.array([[0, 0], [1, 1],[0,1],[1,0]])
    Y = np.array([[1],[1],[0],[0]])
    maxIter = 100000
    errTol = 1e-5
    for i in range(maxIter+1):
        err = bbn.trainEpoch(X,Y)
        if i%10000 == 0:
            print("Iteration: {0}\tError: {1:0.6f}".format(i,err))
        if err <= errTol:
            print("Tolerance error reached at {0}".format(i))
            break;
    Out = bbn.Run(X)
    print("Input:{0}, \nOutput:{1}".format(X, Out))

    
    
    
    
    
    
    
    
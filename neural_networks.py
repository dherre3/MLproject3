#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 11:06:43 2017

@author: davidherrera
"""
import numpy as np

class Neural_Network(object):
    def __init__(self, X, Y, hidden_layers = 3):
        self.input = X
        self.output = Y
        self.inputLayerSize = X.shape[1];
        self.outputLayerSize = Y.shape[0];
        self.hiddenLayerSize = hidden_layers;
    def forward(self, x,num_layers, totalTheta):
        aTot = []
        a = x
        a = self.addBias(a);
        aTot.append(a)
        #Calculate feedforward feedback
        for layer in range(num_layers):
            a = self.sigmoid(np.dot(totalTheta[layer],a))
            a = self.addBias(a)
            aTot.append(a)
        aTot.append(self.sigmoid(np.dot(totalTheta[len(totalTheta)-1],a)))
        return np.array(aTot)
    
    def getDeltaGrad(deltaGrad, sigma, aTot):
        for layer in range(1,len(deltaGrad)-1):
            deltaGrad[layer] = deltaGrad[layer] + np.dot(sigma[layer+1], aTot[layer].T)
        return deltaGrad
    
    def backpropagation(self, reg_param = 0.1):
        x = self.input
        y = self.output
        num_layers = self.hiddenLayerSize
        size_output = y.shape[1]
        size_feat = x.shape[1]
        print(size_feat)
        totalTheta = self.initWeights(size_feat,size_output,num_layers)
        deltaGrad = self.initWeights(size_feat,size_output,num_layers,1)
        size_sample = x.shape[0]
        for training_sample in range(size_sample):  
            aTot = self.forward(x[training_sample],totalTheta, num_layers)
            sigma = self.getSigma(aTot,totalTheta,y[training_sample])
            deltaGrad = self.getDeltaGrad(deltaGrad, sigma, aTot);
        D = self.getDerivTheta(size_sample, deltaGrad, totalTheta,reg_param)
        return D
    
    def getDerivTheta(m, deltaGrad, totalTheta,reg_param):
        for layer in range(len(deltaGrad)):
            deltaGrad[layer] = (1/m)*deltaGrad[layer] + reg_param*totalTheta[layer]
            deltaGrad[layer][0] = (1/m)*deltaGrad[layer][0] 
        return deltaGrad
        
    def getSigma(a, theta, y_sample):
        totalSigma = []
        sigmaL = a[a.size-1]-y_sample;
        totalSigma.append(sigmaL);
        for i in range(1,len(theta)):
            temp1 = np.dot(np.array(theta[len(theta)-i]).T,sigmaL)
            temp = temp1*(a[a.size-i-1]*(1-a[a.size-i-1]))
            totalSigma.append(temp)
        return np.array(totalSigma)
    
    def initWeights(x_size, size_output, num_layers, to_zero = 0):
        totalTheta = []
        print(x_size, size_output)
        if to_zero == 1:
            totalTheta.append(np.zeros((num_layers,x_size+1)))
            for layer in range(num_layers-1):
                theta = np.zeros((num_layers,num_layers+1))
                totalTheta.append(theta)
            totalTheta.append(np.zeros((size_output, num_layers+1)))
        else:
            totalTheta.append(np.random.rand(num_layers,x_size+1))
            for layer in range(num_layers-1):
                theta = np.random.rand(num_layers,num_layers+1)
                totalTheta.append(theta)
            totalTheta.append(np.random.rand(size_output, num_layers+1))
        return totalTheta
    def sigmoid(z):
        return 1/(1+np.exp(-z))
    
    def addBias(x):
        x = np.append([1],x)
        return x
X = np.array([[0., 0.], [1., 1.],[0.,1.], [0.,1.]])
Y =  np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,1]]) 
clf = Neural_Network(X,Y,hidden_layers = 3)
pred = clf.backpropagation(reg_param = 0.1)
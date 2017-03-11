#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 10:14:48 2017

@author: davidherrera
"""

#==============================================================================
# Feedforwad Neural Network
#==============================================================================
import numpy as np
trainX = np.load('./data/tinyX.npy') # this should have shape (26344, 3, 64, 64)
trainY = np.load('./data/tinyY.npy') 
testX = np.load('./data/tinyX_test.npy') # (6600, 3, 64, 64)

print(trainX.shape)
import scipy.misc
scipy.misc.imshow(trainX[0].transpose(2,1,0)) # put RGB channels last


def sigmoid(z):
    return 1/(1+np.exp(-z))
trainX[0]
np.unique(trainY).size
def costFunction(theta, x, y):
    K = np.unique(trainY).size
    trainingSize = x.size
    for i in range(trainingSize):
        for k in range(40):
            print(k)
            (y[i]==k)*np.log(sigmoid(np.dot(theta.T,x)))
        
def addBias(x):
    x = np.append([1],x)
    return x

def initWeights(x_size, size_output, num_layers, to_zero = 0):
    totalTheta = []
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
    

#==============================================================================
# Feed Forward
#==============================================================================
def feedForward(x,totalTheta, num_layers = 3 ):
    aTot = []
    a = x
    a = addBias(a);
    aTot.append(a)
    print(a)
    #Calculate feedforward feedback
    for layer in range(num_layers):
        a = sigmoid(np.dot(totalTheta[layer],a))
        a = addBias(a)
        aTot.append(a)
    aTot.append(sigmoid(np.dot(totalTheta[len(totalTheta)-1],a)))
    return np.array(aTot)
#==============================================================================
# Getting the delta function for the back propagation change
#==============================================================================
def getSigma(a, theta, y_sample):
    totalSigma = []
    sigmaL = a[a.size-1]-y_sample;
    totalSigma.append(sigmaL);
    for i in range(1,len(theta)):
        temp1 = np.dot(np.array(theta[len(theta)-i]).T,sigmaL)
        temp = temp1*(a[a.size-i-1]*(1-a[a.size-i-1]))
        totalSigma.append(temp)
    return np.array(totalSigma)
#==============================================================================
# Getting the derivative
#==============================================================================
def getDerivTheta(deltaGrad, totalTheta,reg_param):
    for layer in range(1,len(deltaGrad)-1):
        np.dot(sigma[layer+1], aTot[layer].T)
        deltaGrad[layer] = deltaGrad[layer] + np.dot(sigma[layer+1], aTot[layer].T)
    return deltaGrad
def getDeltaGrad(deltaGrad, sigma, aTot):
    for layer in range(1,len(deltaGrad)-1):
        deltaGrad[layer] = deltaGrad[layer] + np.dot(sigma[layer+1], aTot[layer].T)
    return deltaGrad
def ffnn(x,y,num_layers = 3, reg_param = 0.001):
    size_output = y.shape[1]
    size_feat = x.shape[1]
    totalTheta = initWeights(size_feat,size_output,num_layers)
    derivThetaTotal = []
    deltaGradTotal = []
    for training_sample in range(x.shape[0]):  
        deltaGrad = initWeights(size_feat,size_output,num_layers,1)
        aTot = feedForward(x[training_sample],totalTheta, num_layers)
        sigma = getSigma(aTot,totalTheta,y[training_sample])
        deltaGrad = getDeltaGrad(deltaGrad, sigma, aTot);
        deltaGradTotal.append(deltaGrad)
        derivTheta = getDerivTheta(deltaGrad, totalTheta,reg_param)
        derivThetaTotal.append(derivTheta)
    
X = np.array([[0., 0.], [1., 1.],[0.,1.], [0.,1.]])
Y =  np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,1]])
print(Y.shape)
sigma =  ffnn(X,Y,3)
print(sigma.shape)



from sklearn.neural_network import MLPClassifier
y = [0, 1,1,1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(trainX, trainY)  
clf.predict([[2., 2.], [-1., -2.]]) 
print(pred)

 
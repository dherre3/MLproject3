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


#J =J + (lambda/(2*m)) * (sum(sum(theta_1(:,2:end).^2,2)) + sum(sum(theta_2(:,2:end).^2,2)));

def sigmoid(z):
    return 1/(1+np.exp(-z))



        
def addBias(x):
    x = np.append([1],x)
    return x

def initWeights(x_size, size_output, hidden_layers, to_zero = 0):
    totalTheta = []
    num_layers = hidden_layers[0]
    unit_layers = hidden_layers[1]
    if to_zero == 1:
        totalTheta.append(np.zeros((unit_layers,x_size+1)))
        for layer in range(num_layers-1):
            theta = np.zeros((unit_layers,unit_layers+1))
            totalTheta.append(theta)
        totalTheta.append(np.zeros((size_output, unit_layers+1)))
    else:
        totalTheta.append(np.random.rand(unit_layers,x_size+1))
        for layer in range(num_layers-1):
            theta = np.random.rand(unit_layers,unit_layers+1)
            totalTheta.append(theta)
        totalTheta.append(np.random.rand(size_output, unit_layers+1))
    return totalTheta
#==============================================================================
# Feed Forward
#==============================================================================
def feedForward(x,totalTheta, num_layers = 3 ):
    aTot = []
    a = x
    a = addBias(a);
    aTot.append(a)
    #Calculate feedforward feedback
    for layer in range(num_layers):
        a = sigmoid(np.dot(totalTheta[layer],a))
        a = addBias(a)
        aTot.append(a)
    aTot.append(sigmoid(np.dot(totalTheta[len(totalTheta)-1],a)))
    print(aTot)
    return np.array(aTot)

#==============================================================================
# Getting the derivative
#==============================================================================
def getDerivTheta(m, deltaGrad, totalTheta,reg_param):
    for layer in range(len(deltaGrad)):
        deltaGrad[layer] = (1/m)*deltaGrad[layer] + reg_param*totalTheta[layer]
        deltaGrad[layer][0] = (1/m)*deltaGrad[layer][0] 
    return deltaGrad
def getDeltaGrad(deltaGrad, sigma, aTot):
    for layer in range(1,len(deltaGrad)-1):
        deltaGrad[layer] = deltaGrad[layer] + np.dot(sigma[layer+1], aTot[layer].T)
    return deltaGrad

def costFunction(thetaLast,theta, x, y, reg_lambda):
    np.dot(thetaLast,x)
    print((y*(np.log(sigmoid(np.dot(thetaLast,x))))).size)
    print(((1-y)*(1-np.log(sigmoid(np.dot(thetaLast,x))))).size)
    J = (-1/x.shape[0])*np.sum(y*(np.log(sigmoid(np.dot(thetaLast,x))))+(1-y)*(1-np.log(sigmoid(np.dot(thetaLast,x)))),axis=1)
    regcost = 0
    for layer in range(len(theta)):
        theta[layer] = np.array(theta[layer])
        regcost += np.sum(theta[layer][:,2:]**2,2) 
    J = J + (reg_lambda/(2*x.shape[0]))*regcost
    return J
#==============================================================================
# Getting the delta function for the back propagation change
#==============================================================================
def getSigma(a, theta, y_sample):
    totalSigma = []
    asize = a.shape[0]
#    sigmaL = a[a.size-1]-y_sample;
#    totalSigma.append(sigmaL);
    totalSigma = [None]*asize
    totalSigma[asize-1] =  a[asize-1]-y_sample;
    #First layer not done and last layer already done, size(a)-3 iter
    for i in range(asize-2,0,-1):
        print("LENGTH:",i, np.array(theta[i]).T.shape,len(totalSigma[i+1]))
        if i == 0:
            temp1 = np.array(theta[i]).T*totalSigma[i+1]
            temp = temp1*(a[i]*(1-a[i]))
        else:
            temp1 = np.array(theta[i]).T*totalSigma[i+1][1:]
            temp = temp1*(a[i]*(1-a[i]))
        totalSigma[i] = temp;
    return totalSigma
def ffnn(x,y,hidden_layers =(1, 3), reg_param = 0.1):
    num_layers = hidden_layers[0]
    unit_layers = hidden_layers[1]
    size_output = y.shape[1]
    size_feat = x.shape[1]
    totalTheta = initWeights(size_feat,size_output,hidden_layers)
    deltaGrad = initWeights(size_feat,size_output,hidden_layers,1)
    print(totalTheta)
    size_sample = x.shape[0]
    for training_sample in range(size_sample):  
        aTot = feedForward(x[training_sample],totalTheta, num_layers)
        sigma = getSigma(aTot,totalTheta,y[training_sample])
        deltaGrad = getDeltaGrad(deltaGrad, sigma, aTot);
    D = getDerivTheta(size_sample, deltaGrad, totalTheta,reg_param)
    #cost = costFunction(totalTheta[len(totalTheta)-1],totalTheta, x,y,reg_param)
    #print(cost)
    return D

X = np.array([[0., 0.], [1., 1.],[0.,1.], [0.,1.]])
Y =  np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,1]])
totalSigma = np.zeros((3,))
a = np.array([-0.05653511,  0.69270575,  0.80896514])
X1 = np.array([[0., 0.]])
Y1 =  np.array([[1,0,0]])
print(Y.shape)
sigma =  ffnn(X1,Y1,(2,3))
print(sigma)
for i in range(3,1, -1):
    print(i)
from sklearn.neural_network import MLPClassifier
y = [0, 1,1,1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
a = [None]*4
a[3] = [1,2,3]
print(a)

    

 
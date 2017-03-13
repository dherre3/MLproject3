#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 18:27:10 2017

@author: davidherrera
"""

import nn 
import numpy as np



if __name__ == "__main__":
    trainX = np.load('./data/tinyX.npy') # this should have shape (26344, 3, 64, 64)
    trainY = np.load('./data/tinyY.npy') 
    testX = np.load('./data/tinyX_test.npy') # (6600, 3, 64, 64)
    newTrain = np.empty((trainX[:10000].shape[0],trainX[0].T.ravel().shape[0]))
    for i in range(trainX[:10000].shape[0]):
        newTrain[i] = trainX[i].T.ravel(order='A')
    print(newTrain.shape)
    classes = np.unique(trainY[:10000]).size
    bbn = nn.BackPropagationNetwork((newTrain.shape[1],2*classes,classes))
    maxIter = 100000
    errTol = 1e-5
    for i in range(maxIter+1):
        err = bbn.trainEpoch(newTrain,trainY[:10000],100)
        print("Iteration: {0}\tError: {1:0.6f}".format(i,err))
        if err <= errTol:
            print("Tolerance error reached at {0}".format(i))
            break;
    Out = bbn.Run(newTrain)
    print("Input:{0}, \nOutput:{1}".format(newTrain, Out))
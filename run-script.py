#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 18:27:10 2017

@author: davidherrera
"""

import nn 
import numpy as np
import sklearn.metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import preprocessing_images as preIm
import math
if __name__ == "__main__":
    #Loading Data
    trainX = np.load('./data/tinyX.npy') # this should have shape (26344, 3, 64, 64)
    trainY = np.load('./data/tinyY.npy') 
    testX = np.load('./data/tinyX_test.npy') # (6600, 3, 64, 64)
    bottleNeck = np.load('./data/bottleneck_features_train.npy') # this should have shape (26344, 3, 64, 64)
    testX = np.load('./data/tinyX_test.npy') # (6600, 3, 64, 64)
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.2, random_state=42)
    #sizeTraining = 15000
    #Translating Y into binary arrays for output
    sizeTraining = y_train.shape[0]
    lb = preprocessing.LabelBinarizer()
    classes = np.unique(y_train[:sizeTraining]).size
    trainYBi = lb.fit_transform(y_train[:sizeTraining])
    testBi = lb.fit_transform(y_test)
    #trainYBitot = lb.fit_transform(trainY)
    #Training Ravelling
    preI = preIm.ImagePreprocessing()
    newTrainX = preI.unravelImages(X_train[:sizeTraining])
    newTrainX = preIm.normalize(newTrainX)
    newValX = preI.unravelImages(X_test)
    #newBottle = preI.unravelImages(bottleNeck)
    baseline = np.zeros((y_train[:sizeTraining].shape))
    #Splitting Into Batches
    (train, target) = preI.splitIntoBatches(newTrainX,trainYBi,5)
    #print(newBottle.shape[0], newTrainX.shape[0])
    #(trainBottle,targetBottle) = preI.splitIntoBatches(newBottle,trainYBitot,5)
    #train = trainBottle
    #target = targetBottle
    print(classes)
    #Initializating Network and Weights
    bbn = nn.BackPropagationNetwork((newTrainX.shape[1],math.sqrt(newTrainX.shape[1]),  classes))
    
    #Gradient decent tolerance and errTol.
    maxIter = 100
    errTol = 1e-5
    errorTrain = []
    errorVal = []
    for i in range(maxIter+1):
       # err = bbn.trainEpoch(newTrainX,trainYBi,0.1,0.7)
        err = bbn.batchTraining(train,target,trainingRate = 0.7,momentum = 0.5)
        errorTrain.append(err);
        OutValidation = bbn.Run(newValX)
        errorVal.append(bbn.error(OutValidation,testBi))                
        print("Iteration: {0}\tError: {1:0.6f}".format(i+1,err))
        if err <= errTol:
            print("Tolerance error reached at {0}".format(i+1))
            break;
    #Obtaining estimates for training
    #OutTraining = bbn.Run(newBottle)
    OutTraining = bbn.Run(newTrainX)
    #Obtaining estimates for validation
    OutValidation = bbn.Run(newValX)
    
    #Translating back into index
    predictionTrain = np.argmax(OutTraining, axis = 1)
    predictionVal = np.argmax(OutValidation, axis = 1)
    
    #Accuracy and Validation Calculation
    #accuracyBottle = sklearn.metrics.accuracy_score(predictionTrain,trainY)

    accuracyTrain = sklearn.metrics.accuracy_score(predictionTrain,y_train[:sizeTraining])
    accuracyVal = sklearn.metrics.accuracy_score(predictionVal,y_test)
    accuracyBaseline = sklearn.metrics.accuracy_score(baseline,y_train[:sizeTraining])
    
    print(" Accuracy Train: {0}\tAccuracy Val: {1}\tBaseline Acc: {2}".format(accuracyTrain, accuracyVal,accuracyBaseline))


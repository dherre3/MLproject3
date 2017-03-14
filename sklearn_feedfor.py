#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 20:08:01 2017

@author: davidherrera
"""
import numpy as np
import sklearn.metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import preprocessing_images as preIm
#Loading Data
trainX = np.load('./data/tinyX.npy') # this should have shape (26344, 3, 64, 64)
trainY = np.load('./data/tinyY.npy') 
testX = np.load('./data/tinyX_test.npy') # (6600, 3, 64, 64)
bottleNeck = np.load('./data/bottleneck_features_train.npy') # this should have shape (26344, 3, 64, 64)
testX = np.load('./data/tinyX_test.npy') # (6600, 3, 64, 64)
X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.2,random_state=42)
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

from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
sizeHiddenLayers =int(np.floor(np.log(newTrainX.shape[0])))
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(sizeHiddenLayers,sizeHiddenLayers), random_state=1,max_iter=1000,verbose = True, learning_rate_init=0.01,momentum= 0.9)
clf.fit(newTrainX, trainYBi) 
predictionTraining = clf.predict(newTrainX)
predictionTrain = np.argmax(predictionTraining, axis = 1)
accuracyTrain = sklearn.metrics.accuracy_score(predictionTrain,y_train[:sizeTraining])
print(accuracyTrain)
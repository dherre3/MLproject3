#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:50:09 2017

@author: davidherrera
"""
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder
class ImagePreprocessing:
    def unravelImages(self,X_train):
        newTrainX = np.empty((X_train.shape[0],X_train[0].T.ravel().shape[0]))
        for i in range(X_train.shape[0]):
            newTrainX[i] = X_train[i].T.ravel(order='A')
        return newTrainX
    def splitIntoBatches(self,input,target, split_number = 4):
         #Splitting Into Batches
         input = np.array_split(input, split_number)
         target = np.array_split(target, split_number)
         return (input,target)
    def normalize(X):
        return normalize(X,axis = 1)

if __name__ == "__main__":
    one = OneHotEncoder()
    da = one.fit_transform(np.array([0,1,2,3,4])).toarray()
    print(da)
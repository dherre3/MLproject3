#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 13:31:08 2017

@author: davidherrera
"""
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
trainY = np.load('./data/tinyY.npy') 
one = OneHotEncoder(sparse=False)
from itertools import groupby
np.mean([len(list(group)) for key, group in groupby(trainY)])
print(lb.fit_transform(trainY))
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                    samples_per_epoch=len(X_train), epochs=epochs)
np.random.uniform(low=-2, high=2)



# here's a more "manual" example
for e in range(epochs):
    print 'Epoch', e
    batches = 0
    for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=32):
        loss = model.train(X_batch, Y_batch)
        batches += 1
        if batches >= len(X_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
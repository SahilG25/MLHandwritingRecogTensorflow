# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:22:11 2022

@author: Sahil
"""

import pandas as pd
import numpy as np
training_data = pd.read_csv("emnist-balanced-train.csv") # reading training data from csv file using pandas module
testing_data = pd.read_csv("emnist-balanced-test.csv") # reading testing data from csv file using pandas module
#size of training_data is (88799, 785)
#size of testing_data is (14799, 785)
train_data = np.array(training_data, dtype = np.float64) # converts training_data to numpy arrays so elements can be called using its indexes
test_data = np.array(testing_data, dtype = np.float64) # converts testing_data to numpy arrays so elements can be called using its indexes 
train_labels = []
for x in range(0, len(train_data)):
    train_labels.append(train_data[x][0]) # I add the first element of train_data[x] (the label) to train_labels
test_labels = []
for x in range(0, len(test_data)):
    test_labels.append(test_data[x][0]) # I add the first element of test_data[x] (the label) to test_labels
train_data = train_data/255.0
test_data = test_data/255.0
new_train_data = []
for x in range(0, len(train_data)):
    img = np.resize(train_data[x][1:], (28,28)) # I have to reshape the EMNIST data from (784) to (28, 28) for np.fliplr() and np.rot90() to work
    img_flip = np.fliplr(img) # flips image
    img_rotate = np.rot90(img_flip) # rotates image by 90 degrees
    new_train_data.append(img_rotate) # save it back as an element of train_data
new_test_data = []
for x in range(0, len(test_data)):
    img = np.resize(test_data[x][1:], (28,28)) # I have to reshape the EMNIST data from (784) to (28, 28) for np.fliplr() and np.rot90() to work
    img_flip = np.fliplr(img) # flips image
    img_rotate = np.rot90(img_flip) # rotates image by 90 degrees
    new_test_data.append(img_rotate) # save it back as an element of test_data
import tensorflow as tf
model = tf.keras.Sequential([ 
    tf.keras.layers.Conv2D(32,3, input_shape=(28,28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(input_shape=(28,28, 1)),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(47,activation='softmax')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
new_train_data = np.array(new_train_data, dtype = np.float64) # I will convert new_train_data to a NumPy array as tensorflow CNN can only accept NumPy arrays
new_train_data = np.resize(new_train_data, (len(new_train_data), 28, 28, 1)) # since the input for the sequential CNN is (28, 28, 1) we need to resize the new_train_data to facilitate that
train_labels = np.resize(train_labels, (len(train_labels), 1)) # the training label array is being formatted from list of number labels to list of arrays containging number labels
new_test_data = np.array(new_test_data, dtype = np.float64) # I will convert new_train_data to a NumPy array as tensorflow CNN can only accept NumPy arrays
new_test_data = np.resize(new_test_data, (len(new_test_data), 28, 28, 1)) # since the input for the sequential CNN is (28, 28, 1) we need to resize the new_test_data to facilitate that
test_labels = np.resize(test_labels, (len(test_labels), 1)) # the testing label array is being formatted from list of number labels to list of arrays containging number labels
train_labels = np.array(train_labels, np.uint8)
test_labels = np.array(test_labels, np.uint8)
history = model.fit(new_train_data, train_labels,  epochs = 50, validation_data=(new_test_data, test_labels))
model.save("final_CNN")






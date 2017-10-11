# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 20:09:13 2017

@author: Mohitosh
"""
import os

import re
import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Activation, Dropout
from tensorflow.contrib.keras.python.keras.preprocessing import text, sequence
from tensorflow.contrib.keras.python.keras import utils




files = []
emails = []
b=[]
for file in os.listdir("./spam"):
    if file.endswith(".txt"):
        if(file!="labels.txt"):
            files.append(file)
            f = open(os.path.join("./spam", file))
            a= f.read()
            f.close()
        emails.append(a)
print(files)       
del emails[50]

la = []
labels = []
count = 0
f = open(os.path.join("C:/Users/Mohitosh/Desktop/C/spamEmail/misc/spam_data", "labels.txt"))
b=f.read()
f.close()
la =b.split()
labels = la[::2]
print(labels)
print(len(labels))

data = pd.DataFrame({"emails":emails,"spam":labels})

print(data.head)

train_size =int(len(data)/2)
train_email = data['emails'][:train_size]
train_spam = data['spam'][:train_size]
print(train_email)
print(len(train_spam))

test_email = data['emails'][train_size:]
test_spam = data['spam'][train_size:]
#test_words = bow[word_split:]

print(test_email)
max_words = 100
tokenize = text.Tokenizer(num_words = max_words, char_level = False)
tokenize.fit_on_texts(train_email)
x_train = tokenize.texts_to_matrix(train_email)
x_test = tokenize.texts_to_matrix(test_email)
x_train
tokenize.word_index['xmodmap']
encoder = LabelEncoder()
encoder.fit(train_spam)
y_train = encoder.transform(train_spam)
y_test = encoder.transform(test_spam)
y_train[0]
# Inspect the dimenstions of our training and test data (this is helpful to debug)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

import collections
class myKnn:
    n_neigh =0
    trainX = []
    trainY = []
    def __init__(self, n_neigh):
        self.n_neigh = n_neigh
        
    def fit(self, X_train, y_train):
        self.trainX = X_train
        self.trainY = y_train
        return
    
    def calc(self,X_test):
        distances=[]
        targets = []
        
        for i in range(len(self.trainX)):
            distance = np.sqrt(np.sum(np.square(X_test - self.trainX[i,:])))
            distances.append([distance,i])
            
       
        distances =sorted(distances)
        
        for i in range(self.n_neigh):
            index = distances[i][1]
            targets.append(self.trainY[index])
            
        return collections.Counter(targets).most_common(1)[0][0]
    
    
    def predict(self,X_test,y_test):
        predictions =[]
        for i in range(len(X_test)):
            predictions.append(self.calc(X_test[i,:]))
        return predictions    
myCLF = myKnn(n_neigh =3)
myCLF.fit(x_train,y_train)
y_pred =myCLF.predict(x_test,y_test)
print("Accuracy is",accuracy_score(y_test,y_pred))

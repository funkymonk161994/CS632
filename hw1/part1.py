# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 20:26:28 2017

@author: Mohitosh
"""


#Default KNN mock-up

import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
np.unique(iris_y)
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test  = iris_X[indices[-10:]]
iris_y_test  = iris_y[indices[-10:]]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(iris_X_train, iris_y_train) 
defknnPred = knn.predict(iris_X_test)
iris_y_test

#Custom KNN Implementation

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
myCLF.fit(iris_X_train, iris_y_train)
y_pred =myCLF.predict(iris_X_test,iris_y_test)
print('MyAccuracy:',accuracy_score(iris_y_test,y_pred))
print("KnnAccuracy:",accuracy_score(iris_y_test,defknnPred))
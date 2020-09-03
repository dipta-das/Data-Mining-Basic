# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 21:45:24 2020

@author: Dipta Das
"""
###Load The Data Set
from sklearn import datasets
iris = datasets.load_iris()

x = iris.data

y = iris.target

#Spliting the Data Set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=.25)

###Build The Model
### We are using Decision Tree Classifier
from sklearn import tree
classifier=tree.DecisionTreeClassifier()

###Train The model
classifier.fit(x_train,y_train)

###Make Predictions
predictions=classifier.predict(x_test)
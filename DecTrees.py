#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pylab as pltX_
from time import sleep
from IPython import display
from sklearn.tree import DecisionTreeClassifier


def DecisionTrees(trainSet, trainLabels, testSet,D,T):
    
    #%%
    X_train=trainSet
    y_train=trainLabels
    y_train=2*y_train-1
    X_test=testSet
    #%%
    ### Train AdaBoost
    
    w = np.ones(X_train.shape[0]) / X_train.shape[0]
    testpreds=[]
    training_scores=[]

    
    #===============================
    for t in range(T):
        
        clf=DecisionTreeClassifier(max_depth=D)      
        clf.fit(X_train,y_train,sample_weight=w)                                    #Create a decision tree taking into account the weights w
        y_predts=(clf.predict(X_test))                                              #Use the tree to make predictions on the training and test sets
        y_predtr=(clf.predict(X_train))
        gamma=sum([w[i]*(y_predtr!=y_train)[i] for i in range(len(w))])/sum(w)
        alpha_t=np.log((1-gamma)/gamma)                                             #Calculate the weight of each node
        training_scores.append(alpha_t)                      
    
        w=list(w*np.exp(alpha_t*(y_train!=y_predtr)))        #Update the weight of each node
        testpreds.append(y_predts)                           #Save the predictions made by the current tree
        
        

    
    #===============================
    Y_predts=[np.sign(sum([training_scores[r]*testpreds[r][i] for r in range(T)])) for i in range(len(testpreds[0]))]
    Y_predts=[int(elem==1) for elem in Y_predts]

    
    return (Y_predts)






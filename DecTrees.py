#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pylab as pltX_
from time import sleep
from IPython import display

def DecisionTrees(trainSet, trainLabels, testSet):
    ### Fetch the data and load it in pandas
    X_train=trainSet
    #print(np.shape(X_train))
    #print ("Size of the data: ", data.shape)
    
    #%%
    # See data (five rows) using pandas tools
    #print (data.head())
    
    
    ### Prepare input to scikit and train and test cut
    
    #print 
    y_train=trainLabels
    y_train=[int(elem==1)-int(elem==0) for elem in y_train]

    #print(y_train)
    #print(np.shape(y_train))
    X_test=testSet
    
    #%%
    # Import cross validation tools from scikit
            
    #%%
    ### Train a single decision tree
    
    from sklearn.tree import DecisionTreeClassifier
    
    clf = DecisionTreeClassifier(max_depth=8)
    
    # Train the classifier and print training time
    clf.fit(X_train, y_train)
    
    #%%
    # Do classification on the test dataset and print classification results
    #from sklearn.metrics import classification_report
    #target_names = data['Cover_Type'].unique().astype(str).sort()
    y_pred = clf.predict(X_test)
    #print(classification_report(y_test, y_pred, target_names=target_names))
    
    #%%
    # Compute accuracy of the classifier (correctly classified instances)
    from sklearn.metrics import accuracy_score
    #print(accuracy_score(y_test, y_pred))
    #y_pred=2*(y_pred-0.5)
    #print(y_pred)
    
    
    
    #===================================================================
    #%%
    ### Train AdaBoost
    
    # Your first exercise is to program AdaBoost.
    # You can call *DecisionTreeClassifier* as above, 
    # but you have to figure out how to pass the weight vector (for weighted classification) 
    # to the *fit* function using the help pages of scikit-learn. At the end of 
    # the loop, compute the training and test errors so the last section of the code can 
    # plot the lerning curves. 
    # 
    # Once the code is finished, play around with the hyperparameters (D and T), 
    # and try to understand what is happening.
    
    D = 3 # tree depth
    T = 100 # number of trees
    w = np.ones(X_train.shape[0]) / X_train.shape[0]
    #print(w)
    #training_scores = np.zeros(X_train.shape[0])
    training_scores=[]
    test_scores = np.zeros(X_test.shape[0])
    
    #ts = plt.arange(len(training_scores))
    training_errors = []
    test_errors = []
    testpreds=[]
    lypdts=[]
    lprec=[]
    
    #===============================
    for t in range(T):
        
        clf=DecisionTreeClassifier(max_depth=D)
        clf.fit(X_train,y_train,sample_weight=w)
        y_predtr=(clf.predict(X_train))
        y_predts=(clf.predict(X_test))
        #y_predtr=[int(elem==1)-int(elem==0) for elem in y_predtr]
        #y_predts=[int(elem==1)-int(elem==0) for elem in y_predts]
        #print(y_predts)
        training_errors.append(len(np.where(y_predtr!=y_train)[0])/X_train.shape[0])
        #test_errors.append(len(np.where(y_predts!=y_test)[0])/X_test.shape[0])
        gamma=sum([w[i]*(y_predtr!=y_train)[i] for i in range(len(w))])/sum(w)
        alpha_t=np.log((1-gamma)/gamma)
        #print(gamma)
        training_scores.append(alpha_t)
    
        w=list(w*np.exp(alpha_t*(y_train!=y_predtr)))
        testpreds.append(y_predts)
        
    #print(training_scores[0]*testpreds[0][0])
        Y_predts=[np.sign(sum([training_scores[r]*testpreds[r][i] for r in range(t+1)])) for i in range(len(testpreds[0]))]
        #lypdts.append(Y_predts)
        #y2=np.array(Y_predts)
        #lprec.append(len(np.where(y2!=y_test)[0])/X_test.shape[0])
    
    #===============================
    
      #Plot training and test error    
    #plt.plot(training_errors, label="training error")
    #plt.plot(test_errors, label="test error")
    #plt.plot(lprec,label="overall precision")
    #plt.legend()
    #print(y_predts)
    Y_predts=[int(elem==1) for elem in Y_predts]

    
    return (Y_predts)






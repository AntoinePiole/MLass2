# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

def neuralNetworkGetModel(Xtrain,ytrain):
    fashion_mnist = keras.datasets.fashion_mnist
    
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.tanh)
    ])
    
    model.compile(optimizer=tf.train.AdamOptimizer(), 
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    
    model.fit(Xtrain, ytrain, epochs=7, verbose=0)
    theta = optimizeTheta(model, Xtrain, ytrain, Ncoefs=40)
    return model, theta

def optimizeTheta(model, Xtrain, ytrain, Ncoefs=40):
    coefs= np.linspace(-1,1,Ncoefs)
    precisions = []
    
    pred = model.predict(Xtrain)
    #print(pred)
    
    for coef in coefs :
        predictions = [(x>=coef) for x in pred]
        precision = sum(([(predictions[k] == ytrain[k]) for k in range(len(predictions))]))/len(predictions)
        precisions.append(precision)
        
    #Get all the indices of the maximal value. Sometimes they are all the same, so argmax isn't enough
    bestTheta=coefs[np.argmax(precisions)]
    return bestTheta
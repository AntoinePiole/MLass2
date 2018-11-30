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
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.tanh)
    ])
    
    model.compile(optimizer=tf.train.AdamOptimizer(), 
                  loss='mean_squared_logarithmic_error',
                  metrics=['accuracy'])
    
    model.fit(Xtrain, ytrain, epochs=7, verbose=0)
    return model
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

def neuralNetworkPredict(Xtest, model, theta=0.5):
    y_pred = model.predict(Xtest)
    plt.figure(1)
    plt.scatter(y_pred, list(range(len(y_pred))))
    plt.show()
    y_pred = [int(y>theta) for y in y_pred]
    return y_pred
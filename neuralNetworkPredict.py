# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

def neuralNetworkPredict(Xtest, model):
    y_pred = model.predict(Xtest)
    y_pred = [int(y>0.5) for y in y_pred]
    return y_pred
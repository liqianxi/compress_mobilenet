from keras import backend as K
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt


def report_confusion_matrix(model, test_dataset):
    allY = np.array([])
    trueY = np.array([])

    for images, labels in test_dataset.as_numpy_iterator():  # only take first element of dataset
        y = model.predict(images).flatten()

        # Apply a sigmoid since our model returns logits
        predictions = tf.nn.sigmoid(y)
        predictions = tf.where(predictions < 0.5, 0, 1)
        allY = np.concatenate([allY, predictions])
        trueY = np.concatenate([trueY, labels])
    
    report = classification_report(trueY, allY)
    
    return report

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
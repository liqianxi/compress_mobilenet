import tensorflow as tf
import keras
import shutil
import json

from datetime import datetime
from helpers import report_confusion_matrix, recall_m, precision_m, f1_m
import ssl
from keras import backend as K
import numpy as np
from mobilenet_trimmed import MobileNetV2 
import pandas as pd
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
#import tensorflow_model_optimization as tfmot
from tensorflow.keras import datasets, layers, models


ssl._create_default_https_context = ssl._create_unverified_context
trainset_identifier = 20000
testval_set_identifier = 4000
initial_epochs = 100

base_lr = 1e-4
BATCH_SIZE = 32
IMG_SIZE = (96, 96)
IMG_SHAPE = IMG_SIZE + (3,)
train_dir = "./split_data/train/split_train"
validation_dir = "./split_data/train/split_val"
#cur_path = "/home/qianxi/scratch/code/"
cur_path = "/Users/qianxi/Desktop/Leon/2022-2024/2022fall/644/project/code/"
timestamp = datetime.now()
dt_string = timestamp.strftime("%Y%m%d")
full_store_path = f'644model/{dt_string}_{trainset_identifier}train_{testval_set_identifier}valtest_pretrained_{initial_epochs}train/'
full_path = cur_path + full_store_path

# Load train, test, validation set.
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            label_mode="int",
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

train_batches = tf.data.experimental.cardinality(train_dataset)
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 label_mode="int",
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)

validation_dataset = validation_dataset.skip(val_batches // 5)

AUTOTUNE = tf.data.AUTOTUNE

# Use buffered prefetching to load images from disk without having I/O become blocking.
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# This model expects pixel values in [-1, 1].
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

def compose_base_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SHAPE))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    return model

def compose_model(base_model):
    # Create the base model from the pre-trained model MobileNet V2.
    
    #base_model.trainable = False
    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)
    # Add a classification head.
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    
    x = preprocess_input(inputs)
    x = base_model(x)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    
    
    return model

def train_model(model, base_lr,initial_epochs):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_lr),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy',f1_m,precision_m, recall_m])
    print(model.summary())
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', verbose=1, mode='min',patience=5)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=full_path+"cp.ckpt",
                                                 save_weights_only=True,
                                                 verbose=1)

    history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset,callbacks=[earlystop,cp_callback] )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    return acc, val_acc, loss, val_loss, model, history


base_model = compose_base_model()
'''
base_model = MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               alpha=0.35,weights=None)'''
composed_full_model = compose_model(base_model)

acc, val_acc, loss, val_loss, model, history = train_model( composed_full_model, base_lr, initial_epochs)

# Evaluate the model using testset.
loss, accuracy, f1, precision, recall = composed_full_model.evaluate(test_dataset)
print('Test accuracy :', accuracy)

report = report_confusion_matrix(composed_full_model, test_dataset)
print(report)
print(full_path)


if not os.path.exists(full_path):
    os.makedirs(full_path)

composed_full_model.save(full_path)


with open(full_path+"hist.json","w") as obj:
    obj.write(json.dumps(history.history))


with open(full_path+"report.txt","w") as obj:
    obj.write(str(report))

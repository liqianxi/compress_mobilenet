import tensorflow as tf
import keras
import shutil
import json

from datetime import datetime
from helpers import report_confusion_matrix, recall_m, precision_m, f1_m
import ssl
from keras import backend as K
import numpy as np

import pandas as pd
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot

ssl._create_default_https_context = ssl._create_unverified_context
trainset_identifier = 20000
testval_set_identifier = 4000
initial_epochs = 1
fine_tune_epochs = 1
base_lr = 1e-4
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)
train_dir = "./nov7data/train"
validation_dir = "./nov7data/val"
#cur_path = "/home/qianxi/scratch/code/"
cur_path = "/Users/qianxi/Desktop/Leon/2022-2024/2022fall/644/project/code/"
timestamp = datetime.now()
dt_string = timestamp.strftime("%Y%m%d")
full_store_path = f'644model/{dt_string}_{trainset_identifier}train_{testval_set_identifier}valtest_pretrained_{initial_epochs}train_{fine_tune_epochs}finetune/'
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

# Introduce sample diversity by applying random, yet realistic, transformations 
# to the training images, such as rotation and horizontal flipping.
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

# This model expects pixel values in [-1, 1].
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)


def compose_model(base_model):
    # Create the base model from the pre-trained model MobileNet V2.
    
    base_model.trainable = False
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

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    
    
    return model

def load_model(base_model):
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

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    
    
    return model


def prune_model(base_model,loaded_path):




    # Create the base model from the pre-trained model MobileNet V2.
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=train_batches)
}
    base_model.trainable = False
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

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    model.load_weights(loaded_path)
    
    return model

def train_model_with_finetune(base_model, model,base_lr,initial_epochs, fine_tune_epochs):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_lr),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy',f1_m,precision_m, recall_m])
    print(model.summary())
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, mode='min',patience=3)
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

    base_model.trainable = True
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_lr/10),
              metrics=['accuracy',f1_m,precision_m, recall_m])

    print(model.summary())


    total_epochs =  initial_epochs + fine_tune_epochs

    history_fine = model.fit(train_dataset,
                            epochs=total_epochs,
                            initial_epoch=history.epoch[-1],
                            validation_data=validation_dataset,callbacks=[earlystop])
    
    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    return acc, val_acc, loss, val_loss, model, history_fine


base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
                                  
model = compose_model(base_model)

acc, val_acc, loss, val_loss, model, history = train_model_with_finetune(base_model, model, base_lr, initial_epochs,fine_tune_epochs)

# Evaluate the model using testset.
loss, accuracy, f1, precision, recall = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)

report = report_confusion_matrix(model, test_dataset)
print(report)



print(full_path)


if not os.path.exists(full_path):
    os.makedirs(full_path)

model.save(full_path)
model2 = tf.keras.models.load_model(full_path)
print(model2)

with open(full_path+"hist.json","w") as obj:
    obj.write(json.dumps(history.history))


with open(full_path+"report.txt","w") as obj:
    obj.write(str(report))



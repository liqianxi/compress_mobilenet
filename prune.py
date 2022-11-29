import tensorflow as tf
import keras
import shutil
import json

from datetime import datetime
from helpers import report_confusion_matrix, recall_m, precision_m, f1_m
import ssl
from keras import backend as K
import numpy as np
import tempfile
import pandas as pd
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import prunable_layer
from tensorflow_model_optimization.python.core.sparsity.keras import prune_registry


ssl._create_default_https_context = ssl._create_unverified_context
trainset_identifier = 20000
testval_set_identifier = 4000
initial_epochs = 1
fine_tune_epochs = 1
epochs = 5
base_lr = 1e-4
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)
train_dir = "./real_nov7/split_train"
validation_dir = "./real_nov7/split_val"
#cur_path = "/home/qianxi/scratch/code/"
cur_path = "/Users/qianxi/Desktop/Leon/2022-2024/2022fall/644/project/code/"
timestamp = datetime.now()
dt_string = timestamp.strftime("%Y%m%d")
full_store_path = f'644model/{dt_string}_{trainset_identifier}train_{testval_set_identifier}valtest_pretrained_{initial_epochs}train_{fine_tune_epochs}finetune/'
full_path = cur_path + full_store_path
#recall_m, precision_m, f1_m


base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                alpha=0.35,
                                                include_top=False,
                                                weights='imagenet')
base_model.summary()

assert 1==2
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

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                            final_sparsity=0.80,
                                                            begin_step=0,
                                                            end_step=train_batches*initial_epochs)
}

def apply_pruning_to_prunable_layers(layer):
    if isinstance(layer, prunable_layer.PrunableLayer) or hasattr(layer, 'get_prunable_weights') or prune_registry.PruneRegistry.supports(layer):
        return tfmot.sparsity.keras.prune_low_magnitude(layer,**pruning_params)
    print("Not Prunable: ", layer)
    return layer


def prune_model(model):

    
    model_for_pruning = tf.keras.models.clone_model(
        model,
        clone_function=apply_pruning_to_prunable_layers,
    )
    model_for_pruning.summary()




    
    return model


def train(model,base_lr, epochs):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_lr),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy',f1_m,precision_m, recall_m])
    print(model.summary())
    #earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, mode='min',patience=3)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=full_path+"cp.ckpt",
                                                 save_weights_only=True,
                                                 verbose=1)
    log_dir = tempfile.mkdtemp()
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        # Log sparsity and other metrics in Tensorboard.
        tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir),
        cp_callback
        #earlystop
    ]


    history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset,callbacks=callbacks )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    return acc, val_acc, loss, val_loss, model, history

def make_prediction(model):
    loss, accuracy, f1, precision, recall = model.evaluate(test_dataset)
    return loss, accuracy, f1, precision, recall


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



from keras.utils.vis_utils import plot_model
plot_model(base_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

assert 1==2
prune = 1
if prune:
    #model = load_model(base_model,"./baseline/saved_model.pb")
    #acc, val_acc, loss, val_loss, model, history = train(base_model, model, base_lr, initial_epochs)
    model2 = tf.keras.models.load_model("./baseline2/",custom_objects={"f1_m":f1_m,"precision_m":precision_m,"recall_m":recall_m}, compile=True)
    model2.summary()
    #loss, accuracy, f1, precision, recall = make_prediction(model2)
    #print(loss, accuracy, f1, precision, recall)
    model = prune_model(model2)
    acc, val_acc, loss, val_loss, model, history = train(model,base_lr, epochs)



else:                                          
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


    with open(full_path+"hist.json","w") as obj:
        obj.write(json.dumps(history.history))


    with open(full_path+"report.txt","w") as obj:
        obj.write(str(report))



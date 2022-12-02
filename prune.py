import tensorflow as tf
import keras
import shutil
import json
from keras.utils.vis_utils import plot_model
from datetime import datetime
from helpers import report_confusion_matrix, recall_m, precision_m, f1_m
import ssl
import os
import zipfile
from keras import backend as K
import numpy as np
import tempfile
import pandas as pd
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
#import tensorflow_model_optimization as tfmot
#from tensorflow_model_optimization.python.core.sparsity.keras import prunable_layer
#from tensorflow_model_optimization.python.core.sparsity.keras import prune_registry
import tempfile


def get_gzipped_model_size(model):
    # Returns size of gzipped model, in bytes.


    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(model)

    return os.path.getsize(zipped_file)

ssl._create_default_https_context = ssl._create_unverified_context
trainset_identifier = 20000
testval_set_identifier = 4000
initial_epochs = 1
fine_tune_epochs = 1
epochs = 5
base_lr = 1e-4
BATCH_SIZE = 32

train_dir = "./real_nov7/split_train"

cur_path = "/Users/qianxi/Desktop/Leon/2022-2024/2022fall/644/project/code/"










"""pruning_params = {
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
"""


def representative_dataset():
    for image, _ in train_dataset.take(50):

        yield([image])

root_path = "/Users/qianxi/Desktop/Leon/2022-2024/2022fall/644/project/code/644model/mobilenet"
for model_folder in os.listdir(root_path):
    if model_folder != ".DS_Store":
        full_path = root_path +'/'+model_folder
        if "96img" in model_folder:
            img_size = 96
        elif "224img" in model_folder:
            img_size = 224
        elif "160img" in model_folder:
            img_size = 160

        IMG_SIZE = (img_size,img_size)
        IMG_SHAPE = IMG_SIZE + (3,)
        # Load train, test, validation set.
        train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                    shuffle=True,
                                                                    label_mode="int",
                                                                    batch_size=BATCH_SIZE,
                                                                    image_size=IMG_SIZE)


        AUTOTUNE = tf.data.AUTOTUNE

        # Use buffered prefetching to load images from disk without having I/O become blocking.
        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

        print("current processing",model_folder)
        model2 = tf.keras.models.load_model(full_path,custom_objects={"f1_m":f1_m,"precision_m":precision_m,"recall_m":recall_m}, compile=True)

        converter = tf.lite.TFLiteConverter.from_keras_model(model2) 
        #converter.optimizations = [tf.lite.Optimize.DEFAULT]
        model_tflite_no_quant = converter.convert()
        
        lite_path = full_path+'/mobilenet_no_quant.tflite'
        with open(lite_path, 'wb') as f:
            f.write(model_tflite_no_quant)
        #print("Size of gzipped quantization model: %.2f bytes" % (get_gzipped_model_size(model_tflite_no_quant)))
        os.system(f"xxd -i {lite_path} > {full_path}/mobilenet_no_quant_model.cc")

        converter = tf.lite.TFLiteConverter.from_keras_model(model2) 
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Enforce integer only quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.representative_dataset = representative_dataset
        model_tflite = converter.convert()
        #print("Size of gzipped quantization model: %.2f bytes" % (get_gzipped_model_size(model_tflite)))
        
        lite_path = full_path+'/mobilenet_quantization.tflite'
        with open(lite_path, 'wb') as f:
            f.write(model_tflite)
        #assert 1==2
        os.system(f"xxd -i {lite_path} > {full_path}/quantization_model.cc")
            




import tensorflow as tf
import keras
import shutil
import json

from datetime import datetime
from helpers import report_confusion_matrix, recall_m, precision_m, f1_m
import sys
import time


test_dir = sys.argv[1]
#test_dir = '/Users/qianxi/Desktop/Leon/2022-2024/2022fall/644/project/split_data/test_new'
BATCH_SIZE = 32
IMG_SIZE = (96,96)
IMG_SHAPE = IMG_SIZE + (3,)
test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                                 shuffle=True,
                                                                 label_mode="int",
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)

test_dataset_size=1000
model = tf.keras.models.load_model("/Users/qianxi/Desktop/Leon/2022-2024/2022fall/644/project/code/644model/20221130_20000train_4000valtest_pretrained_80train_15finetune",custom_objects={"f1_m":f1_m,"precision_m":precision_m,"recall_m":recall_m}, compile=True)
time1 = time.time()
loss, accuracy, f1, precision, recall = model.evaluate(test_dataset)
time2 = time.time()

diff_mill = (time2-time1)*1000 / test_dataset_size
print("difference",)
print('Test accuracy :', accuracy)


report = report_confusion_matrix(model, test_dataset)
print(report)
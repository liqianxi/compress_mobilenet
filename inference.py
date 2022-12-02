import tensorflow as tf
import keras
import shutil
import json

from datetime import datetime
from helpers import report_confusion_matrix, recall_m, precision_m, f1_m
import sys
import time

BATCH_SIZE = 32
test_dir = sys.argv[1]
#test_dir = '/Users/qianxi/Desktop/Leon/2022-2024/2022fall/644/project/split_data/test_new'

root_path = "/Users/qianxi/Desktop/Leon/2022-2024/2022fall/644/project/code/644model/mobilenet"
acc_list = {}
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
        test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                                        shuffle=True,
                                                                        label_mode="int",
                                                                        batch_size=None,
                                                                        image_size=IMG_SIZE)

        test_dataset_size=1000
        model = tf.keras.models.load_model(full_path,custom_objects={"f1_m":f1_m,"precision_m":precision_m,"recall_m":recall_m}, compile=True)
        time1 = time.time()
        loss, accuracy, f1, precision, recall = model.evaluate(test_dataset)
        time2 = time.time()

        diff_mill = (time2-time1)*1000 / test_dataset_size
        print("difference",)
        print('Test accuracy :', accuracy)
        acc_list[model_folder] = diff_mill
print(acc_list)

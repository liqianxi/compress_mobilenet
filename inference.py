import tensorflow as tf
import keras
import shutil
import json

from datetime import datetime
from helpers import report_confusion_matrix, recall_m, precision_m, f1_m
import sys
import time
import os


BATCH_SIZE = 32
test_dir = sys.argv[1]
#test_dir = '/Users/qianxi/Desktop/Leon/2022-2024/2022fall/644/project/split_data/train/testset'
model_type = sys.argv[2]
root_path = f"/Users/qianxi/Desktop/Leon/2022-2024/2022fall/644/project/code/644model/{model_type}"
acc_list = {}
for batch_size in [1, 16, 32, 64, 128]:
    tmp_dir = {}
    for model_folder in os.listdir(root_path):
        if model_folder != ".DS_Store" and ".json" not in model_folder:
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
                                                                            batch_size=batch_size,
                                                                            image_size=IMG_SIZE)

            test_dataset_size=1000
            model = tf.keras.models.load_model(full_path,custom_objects={"f1_m":f1_m,"precision_m":precision_m,"recall_m":recall_m}, compile=True)
            time1 = time.time()
            loss, accuracy, f1, precision, recall = model.evaluate(test_dataset,batch_size=batch_size)
            time2 = time.time()

            diff_mill = (time2-time1)*1000 / test_dataset_size
            print("difference",diff_mill)
            print('Test accuracy :', accuracy)
            tmp_dir[model_folder] = diff_mill
    acc_list[batch_size] = tmp_dir

with open(root_path+f"/{model_type}_latency.json",'w') as obj:
    obj.write(json.dumps(acc_list)) 


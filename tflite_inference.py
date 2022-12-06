import tensorflow as tf

import numpy as np

import copy,time,json
import os

def evaluate_model(model_path,test_dir):
    if "_96" in model_path:
        IMG = 96
    elif "_224" in model_path:
        IMG = 224
    elif "_160" in model_path:
        IMG = 160

    IMG_SIZE = (IMG,IMG)

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Allocate tensors
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                                 shuffle=True,
                                                                 label_mode="int",
                                                                 batch_size=None,
                                                                 image_size=IMG_SIZE)
    # Run predictions on ever y image in the "test" dataset.
    prediction_digits = []
    """
    [{'name': 'serving_default_input_2:0', 'index': 0, 
    'shape': array([  1, 224, 224,   3], dtype=int32), 
    'shape_signature': array([ -1, 224, 224,   3], dtype=int32),
     'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 
     'quantization_parameters': {'scales': array([], dtype=float32), 
     'zero_points': array([], dtype=int32), 'quantized_dimension': 0},
      'sparsity_parameters': {}}]

    """
    #classify_lite = interpreter.get_signature_runner('serving_default')
    #print(interpreter.get_input_details())
    test_labels = []
    time_sum = 0
    for i, test_image in enumerate(test_dataset):
        label = np.asarray(test_image)[1]
        raw = np.asarray(test_image)[0]#
        real= raw.numpy().reshape((1, IMG, IMG, 3))

        #print(classify_lite(input_2=array2))
        test_labels.append(label.numpy().item(0))

        if i % 100 == 0:
            print('Evaluated on {n} results so far.'.format(n=i))
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.

        #test_image = np.expand_dims(real, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, real)

        # Run inference.
        time1 = time.time()
        interpreter.invoke()
        time_sum += time.time() - time1

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        output = interpreter.tensor(output_index)
        result_value = copy.deepcopy(output())[0]
        digit = 0
        if result_value >=0:
            digit = 1


        #digit = np.argmax(output()[0])
        prediction_digits.append(digit)
        #print(digit)
        #print("real",real_label)
        
        #assert 1==2

    print('\n')
    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_digits)
    accuracy = (prediction_digits == test_labels).mean()
    return accuracy, time_sum


def evaluate_quant_model(model_path,test_dir):
    if "_96" in model_path:
        IMG = 96
    elif "_224" in model_path:
        IMG = 224
    elif "_160" in model_path:
        IMG = 160

    IMG_SIZE = (IMG,IMG)

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)

    # Get input and output tensors.
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    #print(output_details)
    # Allocate tensors
    interpreter.allocate_tensors()
    input_index = input_details["index"]
    output_index = output_details["index"]
    test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                                 shuffle=True,
                                                                 label_mode="int",
                                                                 batch_size=None,
                                                                 image_size=IMG_SIZE)
    # Run predictions on ever y image in the "test" dataset.
    prediction_digits = []
    
    input_scale, input_zero_point = input_details['quantization']
    
    """
    [{'name': 'serving_default_input_2:0', 'index': 0, 
    'shape': array([  1, 224, 224,   3], dtype=int32), 
    'shape_signature': array([ -1, 224, 224,   3], dtype=int32), 
    'dtype': <class 'numpy.int8'>, 'quantization': (1.0, -128), 
    'quantization_parameters': {'scales': array([1.], dtype=float32), 
    'zero_points': array([-128], dtype=int32), 'quantized_dimension': 0}, 
    'sparsity_parameters': {}}]

    [{'name': 'StatefulPartitionedCall:0', 'index': 180, 
    'shape': array([1, 1], dtype=int32), 'shape_signature': array([-1,  1], dtype=int32), 
    'dtype': <class 'numpy.int8'>, 'quantization': (0.5939619541168213, -67), 
    'quantization_parameters': {'scales': array([0.59396195], dtype=float32), 
    'zero_points': array([-67], dtype=int32), 'quantized_dimension': 0}, 
    'sparsity_parameters': {}}]
    

    """


    #classify_lite = interpreter.get_signature_runner('serving_default')
    #print(interpreter.get_input_details())
    test_labels = []
    time_sum = 0
    for i, test_image in enumerate(test_dataset):
        label = np.asarray(test_image)[1]
        raw = np.asarray(test_image)[0] / input_scale + input_zero_point


        real= raw.numpy().astype(input_details['dtype']).reshape((1, IMG, IMG, 3))
        
        #print(classify_lite(input_2=array2))
        test_labels.append(label.numpy().item(0))

        if i % 100 == 0:
            print('Evaluated on {n} results so far.'.format(n=i))
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.

        #test_image = np.expand_dims(real, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, real)

        # Run inference.
        time1 = time.time()
        interpreter.invoke()
        time_sum += time.time() - time1

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        output = interpreter.tensor(output_index)
        scale, zero_point = output_details['quantization']
        result_value = copy.deepcopy(output())[0]
        
        tflite_output=result_value.astype(np.float32).item(0)
        #print(tflite_output,zero_point,scale)
        result_value= (tflite_output- zero_point)* scale

        digit = 0
        if result_value >=0:
            digit = 1
        #assert 1==2

        #digit = np.argmax(output()[0])
        prediction_digits.append(digit)
        #print(digit)
        #print("real",real_label)
        
        #assert 1==2

    print('\n')
    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_digits)
    accuracy = (prediction_digits == test_labels).mean()
    return accuracy, time_sum
test_dir = '/Users/qianxi/Desktop/Leon/2022-2024/2022fall/644/project/split_data/train/testset'


#norm_res,norm_latency = evaluate_quant_model("/Users/qianxi/Desktop/Leon/2022-2024/2022fall/644/project/code/644model/mobilenet/20221201_20000train_4000valtest_pretrained_80train_15finetune_96img_size_1.0alpha_value/mobilenet_quantization.tflite",test_dir)
#print(norm_res,norm_latency)
#assert 1==2
for mode in ["smallcnn","mobilenet"]:
    root_path = "/Users/qianxi/Desktop/Leon/2022-2024/2022fall/644/project/code/644model/"+mode
    result_dict = {}
    for model_folder in os.listdir(root_path):
        if model_folder != ".DS_Store" and ".json" not in model_folder:
            full_path = root_path +'/'+model_folder
            if "96img" in model_folder:
                img_size = 96
            elif "224img" in model_folder:
                img_size = 224
            elif "160img" in model_folder:
                img_size = 160

            quant_prefix = f"/{mode}_quantization.tflite"
            normal_prefix = f"/{mode}_no_quant.tflite"
            
            norm_res,norm_latency = evaluate_model(full_path+normal_prefix,test_dir)
            quant,quant_latency = evaluate_quant_model(full_path+quant_prefix,test_dir)
    
            result_dict[model_folder] = {"no_quant":[norm_res,norm_latency],"quant":[quant,quant_latency]}

    with open(root_path+"/inference_result.json","w") as obj:
        obj.write(json.dumps(result_dict))


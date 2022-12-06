import seaborn as sns
import matplotlib.pyplot as plt
import csv
import numpy as np, math
import json,re
from matplotlib.pyplot import figure

flash_size = 1024


fig_path = "/Users/qianxi/Desktop/Leon/2022-2024/2022fall/644/project/code/figures/"


def draw_size_plot_mobilenet():
    pair_list = []
    width_multiplier_list = [0.35,0.5,0.75,1.0]
    model_size_before = [1638.4, 2867.2, 5529.6, 9113.6]
    model_size_after = [623, 973, 1740.8, 2764.8]
    for i in range(len(width_multiplier_list)):
        pair_list.append((width_multiplier_list[i],model_size_before[i]))
        pair_list.append((width_multiplier_list[i],model_size_after[i]))
        

    plt.rc('axes', axisbelow=True)
    plt.grid()

    plt.scatter( x=width_multiplier_list, y=model_size_before)
    plt.scatter( x=width_multiplier_list, y=model_size_after)

    plt.ylim((0, 10000))

    for each in pair_list:
        plt.text(each[0]-0.05, each[1], f"({each[0]}, {each[1]} kb)")




    plt.axhline(flash_size,color="red",label="Storage Constraint")
    plt.legend(labels=["Before Compression","After Compression","Storage Constraint"], loc='upper left')
    plt.title("(Figure 1.1) MobileNet V2 model size with different width multiplier")
    plt.xlabel("Width Multiplier")
    plt.ylabel("Model size in kb")

    plt.savefig(fig_path+"first-1.png") 
    plt.clf()

def draw_size_plot_cnn():
    plt.grid()
    pair_list = []
    scale_list = [1,2,3]
    model_size_before = [229, 525, 1741]
    model_size_after = [64, 143, 446]

    for i in range(len(scale_list)):
        pair_list.append((scale_list[i],model_size_before[i]))
        pair_list.append((scale_list[i],model_size_after[i]))
        

    plt.rc('axes', axisbelow=True)
    

    plt.scatter( x=scale_list, y=model_size_before)
    plt.scatter( x=scale_list, y=model_size_after)

    plt.ylim((0, max(model_size_before)+500))

    for each in pair_list:
        plt.text(each[0]-0.05, each[1], f"({each[0]}, {each[1]} kb)")

    plt.axhline(flash_size,color="red",label="Storage Constraint")
    plt.legend(labels=["Before Compression","After Compression","Storage Constraint"], loc='upper left')
    plt.title("(Figure 1.2) CNN model sizes with different architectures\nlarger scale number means more convolution layers")
    plt.xlabel("Scale of the model")
    plt.ylabel("Model size in kb")
    plt.xticks(np.arange(min(scale_list), max(scale_list)+1, 1))
    plt.savefig(fig_path+"first-2.png") 
    plt.clf()

def draw_latency_plot():
    def find_components(string, find_type):
        if find_type == "cnn":
            return re.findall(r".*train_(.+)img_size_(.+)scale",string)[0]
        else:
            return re.findall(r".*finetune_(.+)img_size_(.+)alpha_value",string)[0]

    cnn_latency_path = "/Users/qianxi/Desktop/Leon/2022-2024/2022fall/644/project/code/644model/smallcnn/smallcnn_latency.json"
    mobilenet_latency_path = "/Users/qianxi/Desktop/Leon/2022-2024/2022fall/644/project/code/644model/mobilenet/mobilenet_latency.json"

    with open(cnn_latency_path) as obj:
        file_content = json.loads(obj.read())

    plt.rc('axes', axisbelow=True)
    plt.grid()
    batch_sizes = file_content.keys()
    for each_batch_size in batch_sizes:
      
        models = []
        for each in file_content[each_batch_size].keys():
            if "_96" in each:
                models.append(each.replace("_96","_096"))
            else:
                models.append(each)

        models.sort()
        
        models2=[]

        for each in models:
            if "_096" in each:
                models2.append(each.replace("_096","_96"))
            else:
                models2.append(each)
        latency_list = [file_content[each_batch_size][i] for i in models2]

        colors = np.random.rand(50)

        new_x_list = []
        for each in models2:
            pair = find_components(each, 'cnn')
            new_str = f"{pair[0]}, {pair[1]}"
            new_x_list.append(new_str)
        #x_data = [f"{[0]}, {find_components(i, 'cnn')[1]}" for i in models]
        
        plt.plot( new_x_list, latency_list)

    plt.xlabel("Input size, scale")
    plt.ylabel("Latency in millisecond(s)")
    plt.title("(Figure 2.1) How different batch sizes, input sizes \nand scales will affect the latency of the CNN model")
    plt.legend(labels=[i for i in batch_sizes], loc='upper left')

    plt.savefig(fig_path+"second-1.png")
    plt.clf()
    with open(mobilenet_latency_path) as obj:
        file_content = json.loads(obj.read())


    plt.rc('axes', axisbelow=True)
    plt.grid()
    batch_sizes = file_content.keys()
    for each_batch_size in batch_sizes:
        models = []
        for each in file_content[each_batch_size].keys():
            if "_96" in each:
                models.append(each.replace("_96","_096"))
            else:
                models.append(each)

        models.sort()
        
        models2=[]

        for each in models:
            if "_096" in each:
                models2.append(each.replace("_096","_96"))
            else:
                models2.append(each)

        latency_list = [file_content[each_batch_size][i] for i in models2]

        colors = np.random.rand(50)

        new_x_list = []
        for each in models2:
            pair = find_components(each, 'mobilenet')
            new_str = f"{pair[0]}, {pair[1]}"
            new_x_list.append(new_str)
        #x_data = [f"{[0]}, {find_components(i, 'cnn')[1]}" for i in models]
        plt.plot( new_x_list, latency_list)
        #plt.scatter( x=new_x_list, y=latency_list,s=12**2)
    plt.legend(labels=[i for i in batch_sizes], loc='upper left')
    plt.xlabel("Input size, width multiplier",labelpad=7)
    plt.ylabel("Latency in millisecond(s)")
    plt.title("(Figure 2.2) How different batch sizes, input sizes \nand scales will affect the latency of MobileNet V2 model")
    
    plt.xticks(rotation = 45)
    plt.savefig(fig_path+"second-2.png")



def draw_accuracy_plot_mobilenet():
    json_path = "/Users/qianxi/Desktop/Leon/2022-2024/2022fall/644/project/code/644model/mobilenet/inference_result.json"

    with open(json_path) as obj:
        content = json.loads(obj.read())

    size_list = [96,160,224]
    width_list = [0.35,0.5,0.75,1.0]

    data_list = [{96:[0.86,0.86,0.88,0.88],160:[0.89,0.89,0.91,0.92],224:[0.91,0.92,0.93,0.93]},
                 {96:[0.851,0.855,0.879, 0.875],160:[0.891,0.89,0.908,0.907],224:[0.892,0.912, 0.92,0.921]}]
    plt.rc('axes', axisbelow=True)
    plt.grid()
    color_dict = {0.35:'r',0.5:"g",0.75:"y",1.0:"k"}
    legend_list = []
    for i, turn in enumerate(data_list):
        temp_str = "Before quantization, width multiplier="
        if i==1:
            line_type = '--'
            temp_str = "After quantization, width multiplier="
        else:
            line_type = "-"
        for width in range(len(width_list)):
            legend_list.append(temp_str+str(width_list[width]))
            x_data = [str(j) for j in size_list]
            y_data = [turn[size][width] for size in turn.keys()]
            #print(x_data,y_data)
            plt.plot( x_data, y_data,color=color_dict[width_list[width]],linestyle=line_type)
        #plt.scatter( x=x_label_list, y=accuracy_after)

    plt.ylim((0.84,0.97))

    #plt.axhline(flash_size,color="red",label="Storage Constraint")
    plt.legend(legend_list, loc='upper left')
    plt.title("(Figure 3.1) MobileNet V2 accuracy with different width multipliers")
    plt.xlabel("Image Size")
    plt.ylabel("Model accuracy")
    plt.xticks(rotation = 45)
    plt.savefig(fig_path+"third-1.png") 

    plt.clf()

def draw_accuracy_plot_cnn():
    json_path = "/Users/qianxi/Desktop/Leon/2022-2024/2022fall/644/project/code/644model/smallcnn/inference_result.json"

    with open(json_path) as obj:
        content = json.loads(obj.read())

    size_list = [96,160,224]
    width_list = [1,2,3]

    data_list = [{96:[0.61, 0.62, 0.62],160:[0.63, 0.59, 0.65],224:[0.61, 0.62, 0.61]},
                 {96:[0.61, 0.635, 0.593],160:[0.591, 0.627, 0.643],224:[0.621, 0.611, 0.607]}]
    plt.rc('axes', axisbelow=True)
    plt.grid()
    color_dict = {1:'r',2:"g",3:"y"}
    legend_list = []
    for i, turn in enumerate(data_list):
        temp_str = "Before quantization, scale="
        if i==1:
            line_type = '--'
            temp_str = "After quantization, scale="
        else:
            line_type = "-"
        for width in range(len(width_list)):
            legend_list.append(temp_str+str(width_list[width]))
            x_data = [str(j) for j in size_list]
            y_data = [turn[size][width] for size in turn.keys()]
            #print(x_data,y_data)
            plt.plot( x_data, y_data,color=color_dict[width_list[width]],linestyle=line_type)
        #plt.scatter( x=x_label_list, y=accuracy_after)

    plt.ylim((0.55,0.7))

    #plt.axhline(flash_size,color="red",label="Storage Constraint")
    plt.legend(legend_list, loc='upper left')
    plt.title("(Figure 3.2) CNN accuracy with different scales")
    plt.xlabel("Image Size")
    plt.ylabel("Model accuracy")
    plt.xticks(rotation = 45)
    plt.savefig(fig_path+"third-2.png") 

    plt.clf()

def draw_latency_plot_cnn():


    size_list = [96,160,224]
    width_list = [1,2,3]

    data_list = [{96:[5.7,2.51,2.59],160:[7.38,11.39,12.68],224:[23.46, 22.36, 23]},
                 {96:[207.93,178.95,184.31],160:[500.09, 609, 791],224:[1185.48, 1352.83, 1510.53]}]
    plt.rc('axes', axisbelow=True)
    plt.grid()
    color_dict = {1:'r',2:"g",3:"y"}
    legend_list = []
    for i, turn in enumerate(data_list):
        temp_str = "Before quantization, scale="
        if i==1:
            line_type = '--'
            temp_str = "After quantization, scale="
        else:
            line_type = "-"
        for width in range(len(width_list)):
            legend_list.append(temp_str+str(width_list[width]))
            x_data = [str(j) for j in size_list]
            y_data = np.log([turn[size][width] for size in turn.keys()])
            #print(x_data,y_data)
            plt.plot( x_data, y_data,color=color_dict[width_list[width]],linestyle=line_type)
        #plt.scatter( x=x_label_list, y=accuracy_after)

    #plt.ylim((0.55,0.7))

    #plt.axhline(flash_size,color="red",label="Storage Constraint")
    plt.legend(legend_list, loc='upper left')
    plt.title("(Figure 3.4) CNN log-latency with different scales")
    plt.xlabel("Image Size")
    plt.ylabel("Model log(latency)")
    plt.xticks(rotation = 45)
    plt.savefig(fig_path+"third-4.png") 

    plt.clf()

def draw_latency_plot_mobilenet():
    json_path = "/Users/qianxi/Desktop/Leon/2022-2024/2022fall/644/project/code/644model/mobilenet/inference_result.json"

    with open(json_path) as obj:
        content = json.loads(obj.read())

    size_list = [96,160,224]
    width_list = [0.35,0.5,0.75,1.0]

    data_list = [{96:[1.67,2.18, 3.26, 3.93],160:[3.74, 4.55, 6.98, 8.61],224:[6.23,8.55,15.8, 18.22]},
                 {96:[33.67, 53.07, 110.66, 163.64],160:[90.27, 140.48, 297.13, 424.16],224:[170.9, 276.27, 588.39, 842.72]}]
    plt.rc('axes', axisbelow=True)
    plt.grid()
    color_dict = {0.35:'r',0.5:"g",0.75:"y",1.0:"k"}
    legend_list = []
    for i, turn in enumerate(data_list):
        temp_str = "Before quantization, width multiplier="
        if i==1:
            line_type = '--'
            temp_str = "After quantization, width multiplier="
        else:
            line_type = "-"
        for width in range(len(width_list)):
            legend_list.append(temp_str+str(width_list[width]))
            x_data = [str(j) for j in size_list]
            y_data = np.log([turn[size][width] for size in turn.keys()])
            #print(x_data,y_data)
            plt.plot( x_data, y_data,color=color_dict[width_list[width]],linestyle=line_type)
        #plt.scatter( x=x_label_list, y=accuracy_after)

    #plt.ylim((0.84,0.97))

    #plt.axhline(flash_size,color="red",label="Storage Constraint")
    plt.legend(legend_list, loc='upper left')
    plt.title("(Figure 3.3) MobileNet V2 log-latency with different width multipliers")
    plt.xlabel("Image Size")
    plt.ylabel("Model log(latency)")
    plt.xticks(rotation = 45)
    plt.savefig(fig_path+"third-3.png") 

    plt.clf()

figure(figsize=(6, 6), dpi=80)
draw_latency_plot()

assert 1==2
draw_size_plot_mobilenet()


draw_size_plot_cnn()

plt.grid()
figure(figsize=(10, 6), dpi=80)
draw_latency_plot()

import seaborn as sns
import matplotlib.pyplot as plt
import csv
import numpy as np
import json,re
from matplotlib.pyplot import figure

flash_size = 1024
figure(figsize=(12, 6), dpi=80)

fig_path = "/Users/qianxi/Desktop/Leon/2022-2024/2022fall/644/project/code/figures/"
width_multiplier_list = [0.35,0.5,0.75,1.0]
model_size_before = [1638.4, 2867.2, 5529.6, 9113.6]
model_size_after = [623, 973, 1740.8, 2764.8]

def draw_size_plot_mobilenet():
    pair_list = []
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




    plt.axhline(flash_size,color="red",label="Flash Constraint")
    plt.legend(labels=["Before Compression","After Compression","Flash Constraint"])
    plt.title("(Figure 1) MobileNet V2 model size with different width multiplier")
    plt.xlabel("Width Multiplier")
    plt.ylabel("Model Size in kb")

    plt.savefig(fig_path+"first.png") 

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
        models = [i for i in file_content[each_batch_size].keys()]
        models.sort()
        latency_list = [file_content[each_batch_size][i] for i in models]

        colors = np.random.rand(50)

        new_x_list = []
        for each in models:
            pair = find_components(each, 'cnn')
            new_str = f"{pair[0]}, {pair[1]}"
            new_x_list.append(new_str)
        #x_data = [f"{[0]}, {find_components(i, 'cnn')[1]}" for i in models]
        
        plt.scatter( x=new_x_list, y=latency_list)

    plt.xlabel("Input size, scale")
    plt.ylabel("Latency in millisecond(s)")
    plt.title("(Figure 2.1) How different batch sizes, input sizes \nand scales will affect the latency of the CNN model")
    plt.legend(labels=[i for i in batch_sizes])

    plt.savefig(fig_path+"second-1.png")
    plt.clf()
    with open(mobilenet_latency_path) as obj:
        file_content = json.loads(obj.read())


    plt.rc('axes', axisbelow=True)
    plt.grid()
    batch_sizes = file_content.keys()
    for each_batch_size in batch_sizes:
        models = [i for i in file_content[each_batch_size].keys()]
        models.sort()
       
        latency_list = [file_content[each_batch_size][i] for i in models]

        colors = np.random.rand(50)

        new_x_list = []
        for each in models:
            pair = find_components(each, 'mobilenet')
            new_str = f"{pair[0]}, {pair[1]}"
            new_x_list.append(new_str)
        #x_data = [f"{[0]}, {find_components(i, 'cnn')[1]}" for i in models]
        
        plt.scatter( x=new_x_list, y=latency_list)
    plt.legend(labels=[i for i in batch_sizes])
    plt.xlabel("Input size, width multiplier",labelpad=7)
    plt.ylabel("Latency in millisecond(s)")
    plt.title("(Figure 2.2) How different batch sizes, input sizes \nand scales will affect the latency of MobileNet V2 model")
    
    plt.xticks(rotation = 45)
    plt.savefig(fig_path+"second-2.png")
            
     

draw_latency_plot()
#draw_size_plot_mobilenet()
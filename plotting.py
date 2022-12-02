import seaborn as sns
import matplotlib.pyplot as plt
import csv

flash_size = 1024
ram_size = 256

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

draw_size_plot_mobilenet()
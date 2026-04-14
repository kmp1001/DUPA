import numpy as np
import matplotlib.pyplot as plt

is_data = {
    "4encoder8decoder":[46.01, 61.47, 69.73, 74.26],
    "6encoder6decoder":[53.11, 71.04, 79.83, 83.85],
    "8encoder4decoder":[54.06, 72.96, 80.49, 85.94],
    "10encoder2decoder": [49.25, 67.59, 76.00, 81.12],
}

fid_data = {
    "4encoder8decoder":[31.40, 22.80, 20.13, 18.61],
    "6encoder6decoder":[27.61, 20.42, 17.95, 16.86],
    "8encoder4decoder":[27.12, 19.90, 17.78, 16.32],
    "10encoder2decoder": [29.70, 21.75, 18.95, 17.65],
}

sfid_data = {
    "4encoder8decoder":[6.88, 6.44, 6.56, 6.56],
    "6encoder4decoder":[6.83, 6.50, 6.49, 6.63],
    "8encoder4decoder":[6.76, 6.70, 6.83, 6.63],
    "10encoder2decoder": [6.81, 6.61, 6.53, 6.60],
}

pr_data = {
    "4encoder8decoder":[0.55006, 0.59538, 0.6063, 0.60922],
    "6encoder6decoder":[0.56436, 0.60246, 0.61668, 0.61702],
    "8encoder4decoder":[0.56636, 0.6038, 0.61832, 0.62132],
    "10encoder2decoder": [0.55612, 0.59846, 0.61092, 0.61686],
}

recall_data = {
    "4encoder8decoder":[0.6347, 0.6495, 0.6559, 0.662],
    "6encoder6decoder":[0.6477, 0.6497, 0.6594, 0.6589],
    "8encoder4decoder":[0.6403, 0.653, 0.6505, 0.6618],
    "10encoder2decoder": [0.6342, 0.6492, 0.6536, 0.6569],
}

x = [100, 200, 300, 400]
# colors = ["#70d6ff", "#ff70a6", "#ff9770", "#ffd670", "#e9ff70"]

colors = ["#52b69a", "#34a0a4", "#168aad", "#1a759f"]

metric_data = {
    "FID50K" : fid_data,
    # "SFID" : sfid_data,
    "InceptionScore" : is_data,
    "Precision" : pr_data,
    "Recall" : recall_data,
}

for key, data in metric_data.items():
    # plt.rc('axes.spines', **{'bottom': True, 'left': True, 'right': False, 'top': False})
    for i, (name, v) in enumerate(data.items()):
        name = name.replace("encoder", "En")
        name = name.replace("decoder", "De")
        plt.plot(x, v, label=name, color=colors[i], linewidth=5.0, marker="o", markersize=10)
    plt.legend(fontsize="14")
    plt.xticks([100, 150, 200, 250, 300, 350, 400])
    plt.grid(linestyle="-.", alpha=0.6, linewidth=0.5)
    plt.ylabel(key, weight="bold")
    plt.xlabel("Training iterations(K steps)", weight="bold")
    plt.savefig("output/base++_{}.pdf".format(key), bbox_inches='tight',)
    plt.close()
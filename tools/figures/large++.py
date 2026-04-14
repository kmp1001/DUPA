import numpy as np
import matplotlib.pyplot as plt

is_data = {
    "10encoder14decoder":[80.48, 104.48, 113.01, 117.29],
    "12encoder12decoder":[85.52, 109.91, 118.18, 121.77],
    "16encoder8decoder":[92.72, 116.30, 124.32, 126.37],
    "20encoder4decoder":[94.95, 117.84, 125.66, 128.30],
}

fid_data = {
    "10encoder14decoder":[15.17, 10.40, 9.32, 8.66],
    "12encoder12decoder":[13.79, 9.67, 8.64, 8.21],
    "16encoder8decoder":[12.41, 8.99, 8.18, 8.03],
    "20encoder4decoder":[12.04, 8.94, 8.03, 7.98],
}

sfid_data = {
    "10encoder14decoder":[5.49, 5.00, 5.09, 5.14],
    "12encoder12decoder":[5.37, 5.01, 5.07, 5.09],
    "16encoder8decoder":[5.43, 5.11, 5.20, 5.31],
    "20encoder4decoder":[5.36, 5.23, 5.21, 5.50],
}

pr_data = {
    "10encoder14decoder":[0.6517, 0.67914, 0.68274, 0.68104],
    "12encoder12decoder":[0.66144, 0.68146, 0.68564, 0.6823],
    "16encoder8decoder":[0.6659, 0.68342, 0.68338, 0.67912],
    "20encoder4decoder":[0.6716, 0.68088, 0.68798, 0.68098],
}

recall_data = {
    "10encoder14decoder":[0.6427, 0.6512, 0.6572, 0.6679],
    "12encoder12decoder":[0.6429, 0.6561, 0.6622, 0.6693],
    "16encoder8decoder":[0.6457, 0.6547, 0.6665, 0.6773],
    "20encoder4decoder":[0.6483, 0.6612, 0.6684, 0.6711],
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
        plt.plot(x, v, label=name, color=colors[i], linewidth=5.0, marker="o", markersize=8)
    plt.legend(fontsize="14")
    plt.grid(linestyle="-.", alpha=0.6, linewidth=0.5)
    plt.xticks([100, 150, 200, 250, 300, 350, 400])
    plt.ylabel(key, weight="bold")
    plt.xlabel("Training iterations(K steps)", weight="bold")
    plt.savefig("output/large++_{}.pdf".format(key), bbox_inches='tight')
    plt.close()
import numpy as np
import matplotlib.pyplot as plt




fid_data = {
    "4encoder8decoder":[64.16, 48.04, 39.88, 35.41],
    "6encoder4decoder":[67.71, 48.26, 39.30, 34.91],
    "8encoder4decoder":[69.4, 49.7, 41.56, 36.76],
}

sfid_data = {
    "4encoder8decoder":[7.86, 7.48, 7.15, 7.07],
    "6encoder4decoder":[8.54, 8.11, 7.40, 7.40],
    "8encoder4decoder":[8.42, 8.27, 8.10, 7.69],
}

is_data = {
    "4encoder8decoder":[20.37, 29.41, 36.88, 41.32],
    "6encoder4decoder":[20.04, 30.13, 38.17, 43.84],
    "8encoder4decoder":[19.98, 29.54, 35.93, 42.025],
}

pr_data = {
    "4encoder8decoder":[0.3935, 0.4687, 0.5047, 0.5271],
    "6encoder4decoder":[0.3767, 0.4686, 0.50876, 0.5266],
    "8encoder4decoder":[0.37, 0.45676, 0.49602, 0.5162],
}

recall_data = {
    "4encoder8decoder":[0.5604, 0.5941, 0.6244, 0.6338],
    "6encoder4decoder":[0.5295, 0.595, 0.6287, 0.6378],
    "8encoder4decoder":[0.51, 0.596, 0.6242, 0.6333],
}

x = [100, 200, 300, 400]
colors = ["#70d6ff", "#ff70a6", "#ff9770", "#ffd670", "#e9ff70"]
metric_data = {
    "FID" : fid_data,
    # "SFID" : sfid_data,
    "InceptionScore" : is_data,
    "Precision" : pr_data,
    "Recall" : recall_data,
}

for key, data in metric_data.items():
    for i, (name, v) in enumerate(data.items()):
        name = name.replace("encoder", "En")
        name = name.replace("decoder", "De")
        plt.plot(x, v, label=name, color=colors[i], linewidth=3, marker="o")
    plt.legend()
    plt.xticks(x)
    plt.ylabel(key, weight="bold")
    plt.xlabel("Training iterations(K steps)", weight="bold")
    plt.savefig("output/base_{}.pdf".format(key), bbox_inches='tight')
    plt.close()
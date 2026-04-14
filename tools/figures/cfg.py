import numpy as np
import matplotlib.pyplot as plt

cfg_data = {
    "[0,      1]":{
        "cfg":[1.0,   1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9, 2.0],
        "FID":[9.23, 6.61, 5.08, 4.46, 4.32, 4.52, 4.86, 5.38, 5.97, 6.57, 7.13],
    },
    "[0.2,   1]":{
        "cfg": [1.2,   1.4,  1.6,  1.8, 2.0],
        "FID": [5.87, 4.44, 3.96, 4.01, 4.26]
    },
    "[0.3,   1]":{
        "cfg": [1.6,   1.7,  1.8,  1.9,  2.0,  2.1,  2.2,  2.3, 2.4],
        "FID": [4.31, 4.11, 3.98, 3.89, 3.87, 3.88, 3.91, 3.96, 4.03]
    },
    "[0.35, 1]":{
        "cfg": [1.6, 1.8, 2.0, 2.1, 2.2, 2.3, 2.4, 2.6],
        "FID": [4.68, 4.22, 3.98, 3.92, 3.90, 3.88, 3.88, 3.94]
    }
}

colors = ["#ff99c8", "#fcf6bd", "#d0f4de", "#a9def9"]

for i, (name, data) in enumerate(cfg_data.items()):
    plt.plot(data["cfg"], data["FID"], label="Interval: " +name, color=colors[i], linewidth=3.5, marker="o")

plt.title("Classifer-free guidance with intervals", weight="bold")
plt.ylabel("FID10K", weight="bold")
plt.xlabel("CFG values", weight="bold")
plt.legend()
plt.savefig("./output/cfg.pdf", bbox_inches="tight")
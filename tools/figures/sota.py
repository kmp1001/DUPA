import numpy as np
import matplotlib.pyplot as plt

data = {
    "SiT-XL/2" : {
        "size": 675,
        "epochs": 1400,
        "FID": 2.06,
        "color": "#ff99c8"
    },
    "DiT-XL/2" : {
        "size": 675,
        "epochs": 1400,
        "FID": 2.27,
        "color": "#fcf6bd"
    },
    "REPA-XL/2" : {
        "size": 675,
        "epochs": 800,
        "FID": 1.42,
        "color": "#d0f4de"
    },
    # "MAR-H" : {
    #     "size": 973,
    #     "epochs": 800,
    #     "FID": 1.55,
    # },
    "MDTv2" : {
        "size": 675,
        "epochs": 920,
        "FID": 1.58,
        "color": "#e4c1f9"
    },
    # "VAVAE+LightningDiT" : {
    #     "size": 675,
    #     "epochs": [64, 800],
    #     "FID": [2.11, 1.35],
    # },
    "DDT-XL/2": {
        "size": 675,
        "epochs": [80, 256],
        "FID": [1.52, 1.31],
        "color": "#38a3a5"
    },
    "DDT-L/2": {
        "size": 400,
        "epochs": 80,
        "FID": 1.64,
        "color": "#5bc0be"
    },
}

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for k, spec in data.items():
    plt.scatter(
        # spec["size"],
        spec["epochs"],
        spec["FID"],
        label=k,
        marker="o",
        s=spec["size"],
        color=spec["color"],
    )
    x = spec["epochs"]
    y = spec["FID"]
    if isinstance(spec["FID"], list):
        x = spec["epochs"][-1]
        y = spec["FID"][-1]
        plt.plot(
            spec["epochs"],
            spec["FID"],
            color=spec["color"],
            linestyle="dotted",
            linewidth=4
        )
        # plt.annotate("",
        #              xytext=(spec["epochs"][0], spec["FID"][0]),
        #              xy=(spec["epochs"][1], spec["FID"][1]), arrowprops=dict(arrowstyle="--"), weight="bold")
    plt.text(x+80, y-0.05, k, fontsize=13)

plt.text(200, 1.45, "4x Training Acc", fontsize=12, color="#38a3a5", weight="bold")
# plt.arrow(200, 1.42, 520, 0, linewidth=2, fc='black', ec='black', hatch="x", head_width=0.05, head_length=0.05)

plt.annotate("",
    xy=(700, 1.42), xytext=(200, 1.42),
             arrowprops=dict(arrowstyle='<->', color='black', linewidth=2),
             )
ax.grid(linestyle="-.", alpha=0.6, linewidth=0.5)
plt.gca().set_xlim(0, 1800)
plt.gca().set_ylim(1.15, 2.5)
plt.xticks([80, 256, 800, 1000, 1200, 1400, 1600, ])
plt.xlabel("Training Epochs", weight="bold")
plt.ylabel("FID50K on ImageNet256x256", weight="bold")
plt.savefig("output/sota.pdf", bbox_inches="tight")
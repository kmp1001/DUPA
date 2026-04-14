import scipy
import numpy as np
import matplotlib.pyplot as plt

def timeshift(t, s=1.0):
    return t/(t+(1-t)*s)

data = {
    "shift 1.0": [8.99, 6.36, 5.03, 4.21, 3.6, 3.23, 2.80],
    "shift 1.5": [6.08, 4.26, 3.43, 2.99, 2.73, 2.54, 2.33],
    "shift 2.0": [5.57, 3.81, 3.11, 2.75, 2.54, 2.43, 2.26],
    "shift 3.0": [7.26, 4.48, 3.43, 2.97, 2.72, 2.57, 2.38],
}
# plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})

# colors = ["#ff99c8", "#fcf6bd", "#d0f4de", "#a9def9"]

colors = ["#52b69a", "#34a0a4", "#168aad", "#1a759f"]
steps = [5, 6, 7, 8, 9, 10, 12]
for i ,(k, v)in enumerate(data.items()):
    plt.plot(steps, v, color=colors[i], label=k, linewidth=4, marker="o")

# plt.title("FID50K of different steps of different timeshift", weight="bold")
plt.ylabel("FID50K", weight="bold")
plt.xlabel("Num of inference steps", weight="bold")
plt.grid(linestyle="-.", alpha=0.6, linewidth=0.5)
# plt.legend()
# plt.legend()
plt.savefig("output/timeshift_fid.pdf", bbox_inches="tight", pad_inches=0)
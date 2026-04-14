import scipy
import numpy as np
import matplotlib.pyplot as plt

def timeshift(t, s=1.0):
    return t/(t+(1-t)*s)

# colors = ["#ff99c8", "#fcf6bd", "#d0f4de", "#a9def9"]
colors = ["#52b69a", "#34a0a4", "#168aad", "#1a759f"]
# plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})
t = np.linspace(0, 1, 100)
shifts = [1.0, 1.5, 2, 3]
for i , shift in enumerate(shifts):
    plt.plot(t, timeshift(t, shift), color=colors[i], label=f"shift {shift}", linewidth=4)

# plt.annotate("", xytext=(0, 0), xy=(0.0, 1.05), arrowprops=dict(arrowstyle="->"), weight="bold")
# plt.annotate("", xytext=(0, 0), xy=(1.05, 0.0), arrowprops=dict(arrowstyle="->"), weight="bold")
# plt.title("Respaced timesteps with various shift value", weight="bold")
# plt.gca().set_xlim(0, 1.0)
# plt.gca().set_ylim(0, 1.0)
plt.grid(linestyle="-.", alpha=0.6, linewidth=0.5)

plt.ylabel("Respaced Timesteps", weight="bold")
plt.xlabel("Uniform Timesteps", weight="bold")
plt.legend(loc="upper left", fontsize="12")
plt.savefig("output/timeshift.pdf", bbox_inches="tight", pad_inches=0)
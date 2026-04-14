import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0.001, 0.999, 100)
def snr(t):
    return np.log((1-t)/t)
def pds(t):
    return np.clip(((1-t)/t)**2, a_max=0.5, a_min=0.0)
print(pds(t))
plt.figure(figsize=(16, 4))
plt.plot(t, snr(t), color="#ff70a6", linewidth=3, marker="o")
# plt.plot(t, pds(t), color="#ff9770", linewidth=3, marker="o")
plt.ylabel("log-SNR", weight="bold")
plt.xlabel("Timesteps", weight="bold")
plt.xticks([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
plt.gca().invert_xaxis()
plt.show()
# plt.savefig("output/logsnr.pdf", bbox_inches='tight')
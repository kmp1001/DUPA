import numpy as np
import matplotlib.pyplot as plt


def lw(x, b=0):
    x = np.clip(x, a_min=0.001, a_max=0.999)
    snr = x/(1-x)
    logsnr = np.log(snr)
    # print(logsnr)
    # return logsnr
    weight =  1 / (1 + np.exp(-logsnr - b))#*(1-x)**2
    return weight #/weight.max()

x = np.arange(0.2, 0.8, 0.001)
print(1/(x*(1-x)))
for b in [0, 1, 2, 3]:
    y = lw(x, b)
    plt.plot(x, y, label=f"b={b}")
plt.legend()
plt.show()
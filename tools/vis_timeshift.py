import scipy
import numpy as np
import matplotlib.pyplot as plt

def timeshift(t, s=1.0):
    return t/(t+(1-t)*s)

def gaussian(t):
    gs = 1+scipy.special.erf((t-t.mean())/t.std())

def rs2(t, s=2.0):
    factor1 = 1.0 #s/(s+(1-s)*t)**2
    factor2 = np.log(t.clip(0.001, 0.999)/(1-t).clip(0.001, 0.999))
    return factor1*factor2


t = np.linspace(0, 1, 100)
# plt.plot(t, timeshift(t, 1.0))
respaced_t = timeshift(t, s=5)
delats = (respaced_t[1:] - respaced_t[:-1])
# plt.plot(t, timeshift(t, 1.5))
plt.plot(rs2(t))
plt.show()
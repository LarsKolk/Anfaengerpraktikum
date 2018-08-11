from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.misc import factorial
import scipy.stats

n = np.genfromtxt('vert.txt', unpack=True)
N = n/10 # Bq

mu, sigma = 4.68, 0.07 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)
k = np.random.poisson(mu, 1000)

plt.hist(k, bins=np.linspace(2, 8, 60), normed=True, facecolor='red', alpha=0.5, label='Poisson')
count, bins, ignored = plt.hist(s, bins=np.linspace(2, 8, 60), normed=True, facecolor='green', alpha=0.5, label='Gauß')
plt.hist(N, bins=np.linspace(2, 8, 60), normed=True, facecolor='blue', alpha=0.5, label='Messwerte')
# plt.hist([k, s, N], bins=np.linspace(2, 8, 50), normed=True, label=['Poissonverteilung', 'Gaußverteilung', 'Messwerte'])
# y = mlab.normpdf(bins, mu, sigma)
# l = plt.plot(bins, y, 'r--', linewidth=2, label='Gauß')
plt.xlabel(r'$N \: / \: \mathrm{Bq}$')
plt.ylabel(r'Häufigkeit')
plt.tight_layout()
plt.legend()
plt.show()

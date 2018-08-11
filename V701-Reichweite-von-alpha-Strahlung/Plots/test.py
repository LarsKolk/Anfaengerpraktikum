from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.misc import factorial

arch = "arch.txt"

datos = []
for item in open(arch,'r'):
    item = item.strip()
    if item != '':
        try:
            datos.append(float(item))
        except ValueError:
            pass

# best fit of data
(mu, sigma) = norm.fit(datos)

# the histogram of the data
n, bins, patches = plt.hist(datos, 30, normed=1, facecolor='green', alpha=0.75, label='Werte')

# add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2, label='Gauß')

# poisson function, parameter lamb is the fit parameter
def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)

plt.plot(bins, poisson(bins, mu), 'b--', lw=2, label='Poisson')


#plot
# plt.xlabel('Smarts')
# plt.ylabel('Probability')
# plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

print(mu, sigma)

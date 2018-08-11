import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

x, y = np.genfromtxt('gravitation.txt', unpack=True)
x *= 1e-2
y *= 1e-3

def f(x, a, b):
    return a * x + b

param, cov = curve_fit(f, x, y)
x_plot = np.linspace(0, 8, 100)

plt.plot(x, y, 'b.', label='Messwerte')
plt.xlabel(r'$r / \mathrm{cm}$')
plt.ylabel(r'$B \,/\, \mathrm{mT}$')

plt.plot(x_plot, f(x_plot, *param), 'r-', label = 'Fit')
# plt.errorbar(r, B, yerr=0.5, fmt='rx')

plt.xlim(0, 8)
plt.ylim(0, 5)
plt.grid()
plt.legend(loc='upper left')
# plt.show()
# plt.savefig('grav.pdf')

err = np.sqrt(np.diag(cov))

print('a =', param[0], '+-', err[0])
print('b =', param[1], '+-', err[1])

my = 1/param[0]*10*0.0014
print(my)

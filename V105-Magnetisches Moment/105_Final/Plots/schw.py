import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

x, y = np.genfromtxt('Schwingung.txt', unpack=True)

def f(x, a, b):
    return a * x + b

param, cov = curve_fit(f, x, y)
x_plot = np.linspace(0, 1500, 10000)

plt.plot(x, y, 'b.', label='Messwerte')
plt.xlabel(r'$B^{-1} / \mathrm{T^{-1}}$')
plt.ylabel(r'$T^2 \,/\, \mathrm{s^2}$')

plt.plot(x_plot, f(x_plot, *param), 'r-', label = 'Fit')
# plt.errorbar(r, B, yerr=0.5, fmt='rx')

plt.xlim(0, 1500)
plt.ylim(0, 7)
plt.grid()
plt.legend(loc='upper left')
# plt.show()
# plt.savefig('schw.pdf')

err = np.sqrt(np.diag(cov))

print('a =', param[0], '+-', err[0])
print('b =', param[1], '+-', err[1])

my = 4*np.pi**2*2/5*0.1421*(2.55e-2)**2*1/param[0]
print(my)

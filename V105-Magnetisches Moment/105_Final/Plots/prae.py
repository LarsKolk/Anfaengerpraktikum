import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

x, y, err_y = np.genfromtxt('Präzession.txt', unpack=True)
# x *= 1e-3
# y *= 1e-3

def f(x, a, b):
    return a * x + b

param, cov = curve_fit(f, x, y)
x_plot = np.linspace(0, 8, 100)

plt.plot(x, y, 'b.', label='Messwerte')
plt.xlabel(r'$B / \mathrm{mT}$')
plt.ylabel(r'$T_p^{-1} \,/\, \mathrm{mHz}$')

plt.plot(x_plot, f(x_plot, *param), 'r-', label = 'Fit')
# plt.errorbar(x, y, yerr=err_y, fmt='b.')

plt.xlim(0, 6)
plt.ylim(0, 350)
plt.grid()
plt.legend(loc='upper left')
plt.show()
# plt.savefig('prae.pdf')

err = np.sqrt(np.diag(cov))

print('a =', param[0], '+-', err[0])
print('b =', param[1], '+-', err[1])

my = param[0]*2*np.pi*2/5*0.1421*(2.55e-2)**2*2*np.pi*5
print(my)

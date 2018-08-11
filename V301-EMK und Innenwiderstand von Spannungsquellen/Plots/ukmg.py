import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

x, y = np.genfromtxt('ukmg.txt', unpack='true')
x *= 1e-3


def f(x, a, b):
    return -a * x + b


param, cov = curve_fit(f, x, y)
x_plot = np.linspace(0, 360, 6000)
plt.plot(x_plot, f(x_plot, *param), 'b-', label='Fit')

plt.plot(x, y, 'rx', label='Messwerte')
plt.xlabel(r'$I \,/\, \mathrm{mA}$')
plt.ylabel(r'$U_k \,/\, \mathrm{V}$')
plt.xlim(21, 84)
plt.ylim(1.74, 2.11)

plt.legend()
plt.grid()
# plt.show()
# plt.savefig('ukmg.pdf')

err = np.sqrt(np.diag(cov))

print('Experimentelle Werte')
print('a =', param[0], '+-', err[0])
print('b =', param[1], '+-', err[1])

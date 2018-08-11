import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

x, y = np.genfromtxt('recht.txt', unpack='true')


def f(x, a, b):
    return a * x**(-b)


param, cov = curve_fit(f, x, y)
x_plot = np.linspace(1e-10, 5000, 6000)
plt.plot(x_plot, f(x_plot, *param), 'b-', label='Fit')
plt.plot(x, y, 'rx', label='Messwerte')
plt.xlabel(r'$\nu \,/\, \mathrm{kHz}$')
plt.ylabel(r'$U_n \,/\, U_1$')
plt.xscale('log')
plt.yscale('log')
plt.xlim(300, 4500)
plt.ylim(0, 1.5)

plt.legend()
# plt.grid()
# plt.show()
# plt.savefig('recht.pdf')

err = np.sqrt(np.diag(cov))


print('Experimentelle Werte')
print('b =', param[1], '+-', err[1])

print('Theoriewerte')
print('b =', 1)

print('Abweichung')
xx = np.abs((param[1]-1)/1)

print('... von b =', xx)

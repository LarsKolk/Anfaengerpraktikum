import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

x, y = np.genfromtxt('led.txt', unpack='true')


def f(x, a, b):
    return a * x**(-b)


param, cov = curve_fit(f, x, y)
x_plot = np.linspace(2, 13, 6000)
plt.plot(x_plot, f(x_plot, *param), 'b-', label='Fit')
plt.plot(x, y, 'rx', label='Messwerte')
plt.xlabel(r'$r \,/\, \mathrm{cm}$')
plt.ylabel(r'$U \,/\, \mathrm{V}$')
plt.xlim(2.7, 13)
# plt.ylim(0.09999, 2.1)
plt.xscale('log')
plt.yscale('log')

plt.legend()
# plt.grid()
# plt.show()
# plt.savefig('led.pdf')

err = np.sqrt(np.diag(cov))

print('Experimentelle Werte')
print('Steigung =', param[1], '+-', err[1])

print('Theoriewerte')
print('Steigung =', 2)

print('Abweichung')
xx = np.abs((2-param[1])/2)*100

print('... der Steigung =', xx, '%')

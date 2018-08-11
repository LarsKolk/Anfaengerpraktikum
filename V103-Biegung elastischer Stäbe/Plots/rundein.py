import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit

x, D0, Dm = np.genfromtxt('rundein.txt', unpack=True)
x *= 1e-2
D0 *= 1e-3
Dm *= 1e-3

L = 50.1
L *= 1e-2

s = 1.183
r = 0.005


def D(x):
    return (Dm - D0) * 10**3


def F(x):
    return (L * x**2 - x**3 / 3)


# np.savetxt('test1.txt', np.column_stack([x, D0, Dm, D(x), F(x)]), fmt="%2.2f", header="x D0 Dm D F")


def g(a, m):
    return m * a


param, cov = fit(g, F(x), D(x))
x_plot = np.linspace(0, 65, 100)

plt.plot(F(x), D(x), 'b.', label='Messwerte')

plt.plot(x_plot, g(x_plot, *param), 'r-', label='lineare Regression')

plt.xlabel(r'$\left(L x^2 - \frac{x^3}{3}\right) \,/\, \mathrm{m^3}$')
plt.ylabel(r'$D(x) \,/\, 10^{-3} \, \mathrm{m}$')

plt.xlim(0, 0.065)
plt.ylim(0, 8)

plt.legend(loc='best')
plt.grid()
# plt.show()
# plt.savefig('rundein.pdf')

err = np.sqrt(np.diag(cov))

print('a =', param[0], '+-', err[0])
# print('b =', param[1], '+-', err[1])

E = (s * 10)/(2*param[0]*(np.pi*r**4)/4)
dE = (s * 10)/(2 * (np.pi*r**4)/4 * param[0]**2) * err[0]

print('E =', E, '+-', dE)

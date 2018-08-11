import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

xx, yy = np.genfromtxt('fit25.txt', unpack='true')
x = np.log(xx)
y = np.log(yy)

def f(x, a, b):
    return a*x+b

param, cov = curve_fit(f, x, y)
x_plot = np.linspace(0, 4.2, 1000)
plt.plot(x_plot, f(x_plot, *param), 'b-', label='Fit')

plt.plot(x, y, 'rx', label='Messwerte')
plt.xlabel(r'$\log{(U)}$')
plt.ylabel(r'$\log{(I)}$')
plt.legend()
# plt.show()
plt.savefig('fit25.pdf')


err = np.sqrt(np.diag(cov))

print(param[0], '+-', err[0])

abw = np.abs(param[0]-3/2)/(3/2) * 100
print(abw, '%')

import matplotlib.pyplot as plt
import numpy as np

x, y = np.genfromtxt('kurz.txt', unpack=True)
# a, b = np.genfromtxt('lang.txt', unpack=True)

plt.plot(x, y, 'rx', label='Messwerte')
# plt.plot(a, b, 'bx', label='Messwerte')
plt.xlabel(r'$x \,/\, \mathrm{cm}$')
plt.ylabel(r'$B \,/\, \mathrm{mT}$')

plt.grid()
plt.legend()
# plt.show()
plt.savefig('kurz.pdf')

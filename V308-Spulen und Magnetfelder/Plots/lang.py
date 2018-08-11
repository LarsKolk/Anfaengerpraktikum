import matplotlib.pyplot as plt
import numpy as np

x, y = np.genfromtxt('lang.txt', unpack=True)
my = 4 * np.pi * 10**(-7)
n = 300
I = 1
l = 15.5e-2

B = my * n/l * I * 10**3

plt.axhline(y=B, color='blue', linestyle='-', label='Theoriekurve')
plt.plot(x, y, 'rx', label='Messwerte')
plt.xlabel(r'$x \,/\, \mathrm{cm}$')
plt.ylabel(r'$B \,/\, \mathrm{mT}$')

plt.grid()
plt.legend()
# plt.show()
# plt.savefig('lang.pdf')

print(B)

import matplotlib.pyplot as plt
import numpy as np

a, b = np.genfromtxt('helm1.txt', unpack=True)
# c, d = np.genfromtxt('helm2.txt', unpack=True)
# e, f = np.genfromtxt('helm3.txt', unpack=True)

n = 100
my = 4 * np.pi * 10**(-7)
I = 3
R = (125/2) * 10**(-3)
d = 7e-2

B = n * (my * I * R**2)/(R**2 + (d/2)**2)**(3/2) * 10**3

plt.axhline(y=B, color='blue', linestyle='-', label='Theoriekurve')
plt.plot(a, b, 'rx', label='Messwerte')
# plt.plot(c, d, 'bx', label='Messwerte')
# plt.plot(e, f, 'kx', label='Messwerte')
plt.xlabel(r'$x \,/\, \mathrm{cm}$')
plt.ylabel(r'$B \,/\, \mathrm{mT}$')

plt.grid()
plt.legend()
# plt.show()
# plt.savefig('helm1.pdf')

print(B)

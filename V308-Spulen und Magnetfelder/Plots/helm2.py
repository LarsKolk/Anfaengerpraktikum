import matplotlib.pyplot as plt
import numpy as np

c, d = np.genfromtxt('helm2.txt', unpack=True)

n = 100
my = 4 * np.pi * 10**(-7)
I = 3
R = (125/2) * 10**(-3)
D = 9e-2

B = n * (my * I * R**2)/(R**2 + (D/2)**2)**(3/2) * 10**3

plt.axhline(y=B, color='blue', linestyle='-', label='Theoriekurve')
print(B)

plt.plot(c, d, 'rx', label='Messwerte')
plt.xlabel(r'$x \,/\, \mathrm{cm}$')
plt.ylabel(r'$B \,/\, \mathrm{mT}$')

plt.grid()
plt.legend()
# plt.show()
# plt.savefig('helm2.pdf')

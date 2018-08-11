import numpy as np
import matplotlib.pyplot as plt

a, b = np.genfromtxt('kenn20.txt', unpack='true')
plt.plot(a, b, 'rx', label='I = 2,0 A')

c, d = np.genfromtxt('kenn21.txt', unpack='true')
plt.plot(c, d, 'bx', label='I = 2,1 A')

e, f = np.genfromtxt('kenn23.txt', unpack='true')
plt.plot(e, f, 'gx', label='I = 2,3 A')

g, h = np.genfromtxt('kenn24.txt', unpack='true')
plt.plot(g, h, 'kx', label='I = 2,4 A')

i, j = np.genfromtxt('kenn25.txt', unpack='true')
plt.plot(i, j, 'yx', label='I = 2,5 A')

plt.xlabel(r'$U \,/\, \mathrm{V}$')
plt.ylabel(r'$I \,/\, \mathrm{mA}$')
plt.legend()
# plt.show()
plt.savefig('kenn.pdf')

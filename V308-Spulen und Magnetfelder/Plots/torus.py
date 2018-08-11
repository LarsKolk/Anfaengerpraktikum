import matplotlib.pyplot as plt
import numpy as np

a, b = np.genfromtxt('torus1.txt', unpack=True)
c, d = np.genfromtxt('torus2.txt', unpack=True)
e, f = np.genfromtxt('torus3.txt', unpack=True)

# ypos = [706.65, 91.7]
#
# for yc in ypos:
#     plt.axhline(y=yc, color='xkcd:grey', linestyle='--')
#
# plt.axvline(x=-0.55, color='xkcd:grey', linestyle='--')

plt.plot(c, d, 'bx', label='absteigende Hysteresekurve')
plt.plot(e, f, 'kx', label='aufsteigende Hysteresekurve')
plt.plot(a, b, 'rx', label='Neukurve')
plt.xlabel(r'$I \,/\, \mathrm{A}$')
plt.ylabel(r'$B \,/\, \mathrm{mT}$')

plt.grid()
plt.legend()
# plt.show()
plt.savefig('torus.pdf')

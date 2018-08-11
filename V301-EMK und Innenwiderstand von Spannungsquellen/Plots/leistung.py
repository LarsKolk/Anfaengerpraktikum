import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
import numpy as np

x, y = np.genfromtxt('uk.txt', unpack='true')
x *= 1e-3

P = x * y
Ra = y/x
U0 = 1.36403829568
Ri = 6.81003699796

t = np.linspace(0, 60, 1000)
N = U0**2/(t+Ri)**2 * t

# np.savetxt('test.txt', np.column_stack([Ra, P]), header="R P")

plt.plot(t, N, 'b-', label='Theoriekurve')
plt.plot(Ra, P, 'rx', label='Messwerte')
plt.xlabel(r'$R_a \,/\, \mathrm{\Omega}$')
plt.ylabel(r'$P \,/\, \mathrm{W}$')
plt.xlim(0, 60)
plt.ylim(0, 0.07)

plt.legend()
plt.grid()
# plt.show()
plt.savefig('leistung.pdf')

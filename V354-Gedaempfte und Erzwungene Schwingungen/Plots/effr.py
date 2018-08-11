import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

x, y = np.genfromtxt('effr.txt', unpack='true')
# x *= 1e-6


def f(x, a, b, c):
    return a * np.exp(-b * x) + c


# np.savetxt('test.txt', np.column_stack([t, Uc]), header='t Uc')

param, cov = curve_fit(f, x, y, p0=(1, 1e-6, 1))
x_plot = np.linspace(0, 400, 1000)

plt.plot(x, y, 'b.', label='Messwerte')
plt.xlabel(r'$t \,/\, \mathrm{μs}$')
plt.ylabel(r'$U_c \,/\, \mathrm{V}$')

plt.plot(x_plot, f(x_plot, *param), 'r-', label='Fit')
# plt.errorbar(r, B, yerr=0.5, fmt='rx')

plt.xlim(0, 305)
plt.ylim(10, 19)
plt.grid()
plt.legend()
# plt.show()
# plt.savefig('effr.pdf')

err = np.sqrt(np.diag(cov))

# print('a =', param[0], '+-', err[0])
print('b =', param[1]*10**6, '+-', err[1]*10**6)
# print('c =', param[2], '+-', err[2])

s = param[1]*10**6
L = 3.53e-3
errL = 0.03e-3

#### Experimentelle Werte ####

R = 2 * L * s
errR = np.sqrt((2*s*errL)**2+(2*L*err[1]*10**6)**2)

print('Experimentelle Werte')
print('R_eff =', R, '+-', errR)

T = 1/s
errT = 1/s**2 * err[1]*10**6

print('T_ex =', T, '+-', errT)

#### Theoretische Werte ####

print('Theoretische Werte')

R1 = 30.3
errR1 = 0.1
Rg = 50

print('R_eff =', R1+Rg, '+-', errR1)

T2 = 2*L/(R1+Rg)
errT2 = np.sqrt((2/(R1+Rg)*errL)**2+(-2*L/(R1+Rg)**2*errR1)**2)
print('T_ex =', T2, '+-', errT2)

print('Abweichung')
yy = np.abs((R1+Rg-R)/(R1+Rg))
print('... von R_eff =', yy)

xx = np.abs((T2-T)/T2)
print('... von T_ex =', xx)

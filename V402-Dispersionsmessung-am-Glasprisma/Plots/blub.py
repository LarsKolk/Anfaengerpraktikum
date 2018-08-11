import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const
import uncertainties.unumpy as unp
import matplotlib.mlab as mlab
from scipy.stats import poisson
from scipy.special import factorial

def noms(x):
    return unp.nominal_values(x)

def stds(x):
    return unp.std_devs(x)

# d = 2,3cm

## Reichweite

p, N, CH = np.genfromtxt('reich.txt', unpack=True)
p0 = 1013 # mbar

x = 23 * p/p0 # mm

Nmax = sum(N[1:7])/len(N[1:7])
print('Nmax/2 =', Nmax/2)

def f(x,a,b):
    return a*x+b

param, cov = curve_fit(f, x[-6:-2], N[-6:-2])
x_plot = np.linspace(9, 16, 10000)
plt.plot(x_plot, f(x_plot, *param), 'b-', label= 'Fit')

plt.plot(x[1:], N[1:], 'rx', label='Werte')
# plt.axhline(Nmax/2, color='orange', linestyle='--')

plt.xlabel(r'$x \: / \: \mathrm{mm}$')
plt.ylabel(r'$N$')
plt.legend()
# plt.show()
# plt.savefig('reich23mm.pdf')

plt.clf()

err = np.sqrt(np.diag(cov))

a = unp.uarray(param[0], err[0])
b = unp.uarray(param[1], err[1])

print('a =', a, '1/mm')
print('b =', b)

Rm = (Nmax/2 - b)/a
print('Rm =', Rm, 'mm')

Ealpha = (Rm/3.1)**(2/3)
print('E_alpha =', Ealpha, 'MeV')

## Energie
E = 4/CH[0] * CH

param2, cov2 = curve_fit(f, x, E)
x_plot = np.linspace(-0.2, 16.1, 10000)
plt.plot(x_plot, f(x_plot, *param2), 'b-', label= 'Fit')

plt.plot(x, E, 'rx', label='Werte')

plt.xlabel(r'$x \: / \: \mathrm{mm}$')
plt.ylabel(r'$E \: / \: \mathrm{MeV}$')
plt.legend()
# plt.show()
# plt.savefig('energie23mm.pdf')

plt.clf()

err2 = np.sqrt(np.diag(cov2))

a2 = unp.uarray(param2[0], err2[0])
b2 = unp.uarray(param2[1], err2[1])

print('a2 =', a2, 'MeV/mm')
print('b2 =', b2, 'MeV')

# d = 2,4cm (Ersatzwerte)

## Reichweite

p, CH, N = np.genfromtxt('messung_5_24mm.txt', unpack=True)
p0 = 1013 # mbar

x = 24 * p/p0 # mm

Nmax2 = sum(N[:12])/len(N[:12])
print('Nmax/2 =', Nmax2/2)

def f(x,a,b):
    return a*x+b

param3, cov3 = curve_fit(f, x[-6:], N[-6:])
x_plot = np.linspace(17, 25, 10000)
plt.plot(x_plot, f(x_plot, *param3), 'b-', label= 'Fit')

plt.plot(x, N, 'rx', label='Werte')
# plt.axhline(Nmax/2, color='orange', linestyle='--')

plt.xlabel(r'$x \: / \: \mathrm{mm}$')
plt.ylabel(r'$N$')
plt.legend()
# plt.show()
# plt.savefig('reich24mm.pdf')

plt.clf()

err3 = np.sqrt(np.diag(cov3))

a3 = unp.uarray(param3[0], err3[0])
b3 = unp.uarray(param3[1], err3[1])

print('a3 =', a3, '1/mm')
print('b3 =', b3)

Rm2 = (Nmax2/2 - b3)/a3
print('Rm =', Rm2, 'mm')

Ealpha2 = (Rm2/3.1)**(2/3)
print('E_alpha =', Ealpha2, 'MeV')

print('Abweichung: %0.2f %%' % (noms(np.abs(Rm2-Rm)/Rm2) * 100))

## Energie
E = 4/CH[0] * CH

param4, cov4 = curve_fit(f, x[:-4], E[:-4])
x_plot = np.linspace(-0.2, 25, 10000)
plt.plot(x_plot, f(x_plot, *param4), 'b-', label= 'Fit')

plt.plot(x, E, 'rx', label='Werte')

plt.xlabel(r'$x \: / \: \mathrm{mm}$')
plt.ylabel(r'$E \: / \: \mathrm{MeV}$')
plt.legend()
# plt.show()
# plt.savefig('energie24mm.pdf')

plt.clf()

err4 = np.sqrt(np.diag(cov4))

a4 = unp.uarray(param4[0], err4[0])
b4 = unp.uarray(param4[1], err4[1])

print('a4 =', a4, 'MeV/mm')
print('b4 =', b4, 'MeV')

print('Abweichung: %0.2f %%' % (noms(-np.abs(a4-a2)/a4) * 100))

# Verteilung

n = np.genfromtxt('vert.txt', unpack=True)
N = n/10 # Bq

nn = sum(N)/len(N)
errN = np.sqrt(1/(len(N)*(len(N)-1)) * sum((N-nn)**2))

Nmit = unp.uarray(nn, errN)
print(Nmit)

mu, sigma = 4.68, 0.07 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)
k = np.random.poisson(mu, 1000)

# count2, bins2, ignored2 = plt.hist(k, 20, normed=True, facecolor='red', alpha=0.75, label='Poisson')
# count, bins, ignored = plt.hist(s, 20, normed=True, facecolor='green', alpha=0.75, label='Gauß')
# plt.hist(N, 20, normed=True, facecolor='blue', alpha=0.75, label='Messwerte')
plt.hist([k, s, N], bins=np.linspace(2, 8, 50), normed=True, label=['Poissonverteilung', 'Gaußverteilung', 'Messwerte'])
# y = mlab.normpdf(bins, mu, sigma)
# l = plt.plot(bins, y, 'r--', linewidth=2, label='Gauß')
plt.xlabel(r'$N \: / \: \mathrm{Bq}$')
plt.ylabel(r'Häufigkeit')
plt.tight_layout()
plt.legend()
# plt.show()
# plt.savefig('vert.pdf')

plt.clf()

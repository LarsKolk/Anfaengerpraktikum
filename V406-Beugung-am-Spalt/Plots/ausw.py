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

Id = 0.002 #muA
L = 975 #mm
l = 635e-9 * 10**3 #mm

# Einzelspalt
x, I = np.genfromtxt('einzel.txt', unpack=True)
I -= Id

In = I/max(I)

def f(x, A0, b, x0):
    return A0**2*b**2*(l/(np.pi*b*np.sin((x-x0)/L)))**2*(np.sin((np.pi*b*np.sin((x-x0)/L))/l))**2

x_plot = np.linspace(9.95, 42.05, 10000)
param, cov = curve_fit(f, x, In, p0=(13.5, 0.075, 25.25))
plt.plot(x_plot, f(x_plot, *param), 'b-', label= r'$\mathrm{Fit} \:\: p_0=(13.5, 0.075, 25.25)$')

# plt.plot(x_plot, f(x_plot, 13.5, 0.075, 25.25), 'g-', label=r'Anpassung per Hand')

plt.plot(x, In, 'rx', label=r'Messwerte')

plt.xlabel(r'$x \:/\: \mathrm{mm}$')
plt.ylabel(r'$I \:/\: \mathrm{μA}$')

plt.legend(loc='best')
plt.tight_layout()
#plt.show()
#plt.savefig('einzel.pdf')

plt.clf()

err = np.sqrt(np.diag(cov))

A0_1 = unp.uarray(param[0], err[0])
b_1 = unp.uarray(param[1], err[1])
x0_1 = unp.uarray(param[2], err[2])

print('---------------------------------------')

print('Einzelspalt')

print('A0_1 =', A0_1, 'muA/mm')
print('b_1 =', b_1, 'mm')
print('x0_1 =', x0_1, 'mm')

b1th = 0.075

abw = np.abs(b_1-b1th)/b1th * 100

print('b_1,theo =', b1th, 'mm')
print('Abweichung: %0.2f %%' % (noms(abw)))

fehl = np.abs(b_1-b1th)/stds(b_1)
print('Fehlerintervall: %0.0f' % (noms(fehl)))


# Doppelspalt 1
x, I = np.genfromtxt('1doppel.txt', unpack=True)
I -= Id

In = I/max(I)

def g(x, A0, b, x0, s):
    return A0**2*b**2*(l/(np.pi*b*np.sin((x-x0)/L)))**2*(np.sin((np.pi*b*np.sin((x-x0)/L))/l))**2*(np.cos((np.pi*s*np.sin((x-x0)/L))/l))**2

x_plot = np.linspace(18.5, 31.35, 10000)
param2, cov2 = curve_fit(g, x, In, p0=(3.5, 0.15, 25.25, 0.25))
plt.plot(x_plot, g(x_plot, *param2), 'b-', label= r'$\mathrm{Fit} \:\: p_0=(3.5, 0.15, 25.25, 0.25)$')
plt.plot(x_plot, f(x_plot, param2[0], param2[1], param2[2]), color='springgreen', linestyle='-', label=r'Einhüllende')

#plt.plot(x_plot, g(x_plot, 3.5, 0.15, 25.25, 0.25), 'g-', label=r'Anpassung per Hand')

plt.plot(x, In, 'rx', label=r'Messwerte')

plt.xlabel(r'$x \:/\: \mathrm{mm}$')
plt.ylabel(r'$I \:/\: \mathrm{μA}$')

plt.legend(loc='best')
plt.tight_layout()
#plt.show()
#plt.savefig('1doppel.pdf')

plt.clf()

err2 = np.sqrt(np.diag(cov2))

A0_2 = unp.uarray(param2[0], err2[0])
b_2 = unp.uarray(param2[1], err2[1])
x0_2 = unp.uarray(param2[2], err2[2])
s_2 = unp.uarray(param2[3], err2[3])

print('---------------------------------------')
print('Doppelspalt 1')

print('A0_2 =', A0_2, 'muA/mm')
print('b_2 =', b_2, 'mm')
print('x0_2 =', x0_2, 'mm')
print('s_2 =', s_2, 'mm')

b2th = 0.15

abw = np.abs(b_2-b2th)/b2th * 100

print('b_2,theo =', b2th, 'mm')
print('Abweichung: %0.2f %%' % (noms(abw)))

fehl = np.abs(b_2-b2th)/stds(b_2)
print('Fehlerintervall: %0.0f' % (noms(fehl)))

s2th = 0.25

abw = np.abs(s_2-s2th)/s2th * 100

print('s_2,theo =', s2th, 'mm')
print('Abweichung: %0.2f %%' % (noms(abw)))

fehl = np.abs(s_2-s2th)/stds(s_2)
print('Fehlerintervall: %0.0f' % (noms(fehl)))

# Doppelspalt 1
x, I = np.genfromtxt('2doppel.txt', unpack=True)
xx, yyy = np.genfromtxt('kunst.txt', unpack=True)
yyy -= Id
I -= Id

yy = yyy/max(I)

In = I/max(I)

x_plot = np.linspace(21.35, 29.6, 10000)
param3, cov3 = curve_fit(g, xx, yy, p0=(7, 0.3, 26, 0.5))
#param3, cov3 = curve_fit(g, x, In, p0=(7, 0.3, 26, 0.5))
plt.plot(x_plot, g(x_plot, *param3), 'b-', label= r'$\mathrm{Fit} \:\: p0=(7, 0.3, 26, 0.5)$')
plt.plot(x_plot, f(x_plot, param3[0], param3[1], param3[2]), color='springgreen', linestyle='-', label=r'Einhüllende')


#plt.plot(x_plot, g(x_plot, 7, 0.15, 25.5, 0.5), 'g-', label=r'Anpassung per Hand')

plt.plot(x, In, 'rx', label=r'Messwerte')
#plt.plot(xx, yy, 'kx', label=r'"Kunst"')

plt.xlabel(r'$x \:/\: \mathrm{mm}$')
plt.ylabel(r'$I \:/\: \mathrm{μA}$')

plt.legend(loc='best')
plt.tight_layout()
#plt.show()
#plt.savefig('2doppel.pdf')

err3 = np.sqrt(np.diag(cov3))

A0_3 = unp.uarray(param3[0], err3[0])
b_3 = unp.uarray(param3[1], err3[1])
x0_3 = unp.uarray(param3[2], err3[2])
s_3 = unp.uarray(param3[3], err3[3])

print('---------------------------------------')
print('Doppelspalt 2')

print('A0_3 =', A0_3, 'muA/mm')
print('b_3 =', b_3, 'mm')
print('x0_3 =', x0_3, 'mm')
print('s_3 =', s_3, 'mm')

abw = np.abs(b_3-b2th)/b2th * 100

print('b_3,theo =', b2th, 'mm')
print('Abweichung: %0.2f %%' % (noms(abw)))

fehl = np.abs(b_3-b2th)/stds(b_3)
print('Fehlerintervall: %0.0f' % (noms(fehl)))

s3th = 0.5

abw = np.abs(s_3-s3th)/s3th * 100

print('s_3,theo =', s3th, 'mm')
print('Abweichung: %0.2f %%' % (noms(abw)))

fehl = np.abs(s_3-s3th)/stds(s_3)
print('Fehlerintervall: %0.0f' % (noms(fehl)))

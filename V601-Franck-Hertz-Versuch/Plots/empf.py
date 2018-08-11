import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const

xx = []
yy = []

b, a = np.genfromtxt('mess210.txt', unpack='true')
d, c = np.genfromtxt('mess250.txt', unpack='true')
f, e = np.genfromtxt('mess290.txt', unpack='true')
h, g = np.genfromtxt('mess320.txt', unpack='true')
j, i = np.genfromtxt('mess350.txt', unpack='true')

# b *= 1e-2
# d *= 1e-2
# f *= 1e-2
# h *= 1e-2
# j *= 1e-2

def s(x, m, n):
    return m*x+n

## U = 210 V ##
param, cov = curve_fit(s, a, b)
x_plot = np.linspace(-35, 15, 1000)
plt.plot(x_plot, s(x_plot, *param), 'k-', label= r'$U_B = 210 \, \mathrm{V}$')
plt.plot(a, b, 'kx')

err = np.sqrt(np.diag(cov))

xx.append(param[0])
yy.append(err[0])

## U = 250 V ##
param, cov = curve_fit(s, c, d)
x_plot = np.linspace(-35, 15, 1000)
plt.plot(x_plot, s(x_plot, *param), 'b-', label= r'$U_B = 250 \, \mathrm{V}$')
plt.plot(c, d, 'bx')

err = np.sqrt(np.diag(cov))

xx.append(param[0])
yy.append(err[0])

## U = 290 V ##
param, cov = curve_fit(s, e, f)
x_plot = np.linspace(-35, 15, 1000)
plt.plot(x_plot, s(x_plot, *param), 'y-', label= r'$U_B = 290 \, \mathrm{V}$')
plt.plot(e, f, 'yx')

err = np.sqrt(np.diag(cov))

xx.append(param[0])
yy.append(err[0])

## U = 320 V ##
param, cov = curve_fit(s, g, h)
x_plot = np.linspace(-35, 15, 1000)
plt.plot(x_plot, s(x_plot, *param), 'g-', label= r'$U_B = 320 \, \mathrm{V}$')
plt.plot(g, h, 'gx')

err = np.sqrt(np.diag(cov))

xx.append(param[0])
yy.append(err[0])

## U = 350 V ##
param, cov = curve_fit(s, i, j)
x_plot = np.linspace(-35, 15, 1000)
plt.plot(x_plot, s(x_plot, *param), 'r-', label= r'$U_B = 350 \, \mathrm{V}$')
plt.plot(i, j, 'rx')

err = np.sqrt(np.diag(cov))

xx.append(param[0])
yy.append(err[0])

plt.ylabel(r'$D \,/\, \mathrm{cm}$')
plt.xlabel(r'$U_D \,/\, \mathrm{V}$')
plt.legend()
# plt.ylim(-0.2, 4.2)
# plt.show()
# plt.savefig('empf.pdf')

print(xx)
print(yy)

plt.clf()

zz = [1/210*10**3, 1/250*10**3, 1/290*10**3, 1/320*10**3, 1/350*10**3]
param, cov = curve_fit(s, zz, xx)
x_plot = np.linspace(0.0028*10**3, 0.0048*10**3, 1000)
plt.plot(x_plot, s(x_plot, *param), 'b-', label='Fit')
plt.plot(zz, xx, 'rx', label='berechnete Werte')
plt.xlabel(r'$\frac{1}{U_B} \: / \: 10^{-3} \: \frac{1}{\mathrm{V}}$')
plt.ylabel(r'$\frac{D}{U_d} \: / \: \frac{\mathrm{cm}}{\mathrm{V}}$')
plt.legend()
# plt.show()
# plt.savefig('konst.pdf')

print(param[0]*10**3, '+-', err[0]*10**3)

p = 1.9
L = 14.3 + 1.03
d = (0.38+0.95)/2

theo = p*L/(2*d)
print('theo =',theo)

abw = np.abs(theo - param[0]*10**3)/theo * 100
print('Abweichung =', abw, '%')

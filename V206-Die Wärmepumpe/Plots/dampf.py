import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

t, x, y = np.genfromtxt('temp.txt', unpack='true')
a, b = np.genfromtxt('druck.txt', unpack='true')

x += 273.15
y += 273.15

z = np.log(b)
zz = 1/x


def f(x, c, d):
    return -c * x + d


param, cov = curve_fit(f, zz, z)
x_plot = np.linspace(3.08*10**(-3), 3.42*10**(-3), 6000)
plt.plot(x_plot*10**3, f(x_plot, *param), 'b-', label='lineare Regression')

plt.plot(zz*10**3, z, 'rx', label='Dampfdruckkurve')
plt.xlabel(r'$\frac{1}{T_1} \,/\, 10^{-3} \, \mathrm{\frac{1}{K}}$')
plt.ylabel(r'$\ln \left(\frac{p_b}{p_0} \right)$')

plt.xlim(3.08, 3.42)
plt.ylim(1.4, 2.6)

plt.grid()
plt.legend()

# plt.show()
# plt.savefig('dampf.pdf')

err = np.sqrt(np.diag(cov))
print('c =', param[0], '+-', err[0])
print('d =', param[1], '+-', err[1])

R = 8.314
L = param[0] * R
errL = R * err[0]

print('L =', L, '+-', errL)

print('Massendurchsatz')
zeit = [180, 600, 1020, 1560]
leistung = [120, 120, 120, 125]
tem1 = [22.6+273.15, 30.7+273.15, 37.3+273.15, 44.0+273.15]
tem2 = [18.0+273.15, 12.3+273.15, 7.6+273.15, 3.5+273.15]
pa = [4.4e5, 3.9e5, 3.5e5, 3.2e5]
pb = [6.1e5, 8.0e5, 9.0e5, 11.0e5]
leis = [120, 120, 120, 125]
kupfer = 750
wasser = 4.182*10**3*4
molar = 115


def g(x, a, b, c):
    return a * x**2 + b * x + c


param2, cov2 = curve_fit(g, t, y)
y_plot = np.linspace(0, 2200, 6000)

err2 = np.sqrt(np.diag(cov2))

i = 0
while i < 4:
    yy = 2 * param2[0] * zeit[i] + param2[1]
    erryy = np.sqrt((2 * zeit[i] * err2[0])**2+(err2[1])**2)
    m = (wasser + kupfer) * np.abs(yy) * 1/L
    m2 = m * molar
    errm = np.sqrt(((wasser + kupfer)/L * erryy)**2 + (-(wasser + kupfer)/L**2 * yy * errL)**2)
    errm2 = molar * errm
    print('m', i, '=', m, '+-', errm, 'mol/s')
    print('m', i, '=', m2, '+-', errm2, 'g/s')
    i += 1

print('mechanische Kompressorleistung')
T0 = 273.15
p0 = 1e5
k = 1.14
rho0 = 5.51 * 10**3

i = 0
while i < 4:
    yy = 2 * param2[0] * zeit[i] + param2[1]
    erryy = np.sqrt((2 * zeit[i] * err2[0])**2+(err2[1])**2)
    m = (wasser + kupfer) * np.abs(yy) * 1/L
    m2 = m * molar
    errm = np.sqrt(((wasser + kupfer)/L * erryy)**2 + (-(wasser + kupfer)/L**2 * yy * errL)**2)
    errm2 = molar * errm
    rho = (rho0 * T0 * pa[i])/(tem2[i] * p0)
    N = 1/(k-1) * (pb[i] * (pa[i]/pb[i])**(1/k) - pa[i]) * 1/rho * m2
    errN = 1/(k-1) * (pb[i] * (pa[i]/pb[i])**(1/k) - pa[i]) * 1/rho * errm2
    wirk = N / leis[i] * 100
    # print('rho', i, '=', rho)
    print('N', i, '=', N, '+-', errN)
    print('Wirkungsgrad', i, '=', wirk)
    i += 1

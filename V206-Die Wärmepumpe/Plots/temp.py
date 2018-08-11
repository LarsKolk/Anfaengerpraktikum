import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

t, x, y = np.genfromtxt('temp.txt', unpack='true')

x += 273.15
y += 273.15


def f(x, a, b, c):
    return a * x**2 + b * x + c


param, cov = curve_fit(f, t, x)
x_plot = np.linspace(0, 2200, 6000)
plt.plot(x_plot, f(x_plot, *param), color='xkcd:green', linestyle='-', label=r'$\mathrm{Regression \,\, T_1}$')

param2, cov2 = curve_fit(f, t, y)
y_plot = np.linspace(0, 2200, 6000)
plt.plot(y_plot, f(y_plot, *param2), color='xkcd:grey', linestyle='-', label=r'$\mathrm{Regression \,\, T_2}$')

plt.plot(t, x, 'kx', label=r'$\mathrm{Messwerte \,\, T_1}$')
plt.plot(t, y, 'bx', label=r'$\mathrm{Messwerte \,\, T_2}$')

plt.xlim(1e-2, 2165)
plt.ylim(0.5+273.15, 50.2+273.15)

plt.xlabel(r'$t \,/\, \mathrm{s}$')
plt.ylabel(r'$T \,/\, \mathrm{K}$')

plt.grid()
plt.legend()

# plt.show()
# plt.savefig('temp.pdf')

err = np.sqrt(np.diag(cov))
err2 = np.sqrt(np.diag(cov2))

print('Regression T1')
print('a =', param[0], '+-', err[0])
print('b =', param[1], '+-', err[1])
print('c =', param[2], '+-', err[2])

print('Regression T2')
print('a =', param2[0], '+-', err2[0])
print('b =', param2[1], '+-', err2[1])
print('c =', param2[2], '+-', err2[2])

print('Aufgabe c)')
zeit = [180, 600, 1020, 1560]
leistung = [120, 120, 120, 125]
tem1 = [22.6+273.15, 30.7+273.15, 37.3+273.15, 44.0+273.15]
tem2 = [18.0+273.15, 12.3+273.15, 7.6+273.15, 3.5+273.15]

print('T1')
for t in zeit:
    xx = 2 * param[0] * t + param[1]
    errxx = np.sqrt((2 * t * err[0])**2+(err[1])**2)
    print(xx, '+-', errxx)

print('T2')
for t in zeit:
    yy = 2 * param2[0] * t + param2[1]
    erryy = np.sqrt((2 * t * err2[0])**2+(err2[1])**2)
    print(yy, '+-', erryy)

print('Aufgabe d)')
kupfer = 750
wasser = 4.182*10**3*4

print('EXPERIMENTELLE WERTE')
print('T1')
i = 0
while i < 4:
    xx = 2 * param[0] * zeit[i] + param[1]
    errxx = np.sqrt((2 * t * err[0])**2+(err[1])**2)
    ny = (wasser + kupfer) * xx * 1/leistung[i]
    errny = (wasser + kupfer)/leistung[i] * errxx
    print('Experimentell:', ny, '+-', errny)
    nytheo = tem1[i]/(tem1[i]-tem2[i])
    print('Theoretisch:', nytheo)
    abw = np.abs((ny-nytheo)/nytheo) * 100
    print('Abweichung:', abw, '%')
    i += 1

print('T2')
i = 0
while i < 4:
    yy = 2 * param2[0] * zeit[i] + param2[1]
    erryy = np.sqrt((2 * t * err2[0])**2+(err2[1])**2)
    ny2 = (wasser + kupfer) * np.abs(yy) * 1/leistung[i]
    errny2 = (wasser + kupfer)/leistung[i] * erryy
    print('Experimentell:', ny2, '+-', erryy)
    nytheo = tem1[i]/(tem1[i]-tem2[i])
    print('Theoretisch:', nytheo)
    abw2 = np.abs((ny2-nytheo)/nytheo) * 100
    print('Abweichung:', abw2, '%')
    i += 1

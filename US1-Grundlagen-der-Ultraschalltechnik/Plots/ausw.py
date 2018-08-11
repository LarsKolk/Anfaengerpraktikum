import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const
import uncertainties.unumpy as unp

ctheo = 2730

def noms(x):
    return unp.nominal_values(x)

def stds(x):
    return unp.std_devs(x)

# Schallgeschwindigkeit mit dem Impuls-Echo-Verfahren

l, U1, U2, dt = np.genfromtxt('echo.txt', unpack=True)

# l *= 1e-2
# dt *= 1e-6

def f(x,a,b):
    return a*x+b

param, cov = curve_fit(f, dt/2, l)
x_plot = np.linspace(0, 46, 10000)
plt.plot(x_plot, f(x_plot, *param), 'b-', label= 'Fit')

plt.plot(dt/2, l, 'rx', label='Messwerte')
plt.xlabel(r'$\frac{\mathrm{\Delta} t}{2} \:/\: \mathrm{\mu s}$')
plt.ylabel(r'$L \:/\: \mathrm{cm}$')
plt.xlim(0, 46)
plt.legend()
# plt.show()
plt.savefig('echo.pdf')

plt.clf()

err = np.sqrt(np.diag(cov))

c1 = unp.uarray(param[0]*10**4, err[0]*10**4)

# print('test:', c1, 'm/s')

print('a1 = c1 =', param[0]*10**4, '+-', err[0]*10**4, 'm/s')
print('b1 =', param[1], '+-', err[1], 'cm')
print('Abweichung:', noms(np.abs(c1-ctheo)/ctheo * 100), '%')

# Schallgeschwindigkeit mit dem Durchschallungs-Verfahren

l, dt = np.genfromtxt('durch.txt', unpack=True)

param2, cov2 = curve_fit(f, dt, l)
x_plot = np.linspace(0, 47, 10000)
plt.plot(x_plot, f(x_plot, *param2), 'b-', label= 'Fit')

plt.plot(dt, l, 'rx', label='Messwerte')
plt.xlabel(r'$\mathrm{\Delta} t \:/\: \mathrm{\mu s}$')
plt.ylabel(r'$L \:/\: \mathrm{cm}$')
plt.xlim(0, 47)
plt.legend()
# plt.show()
# plt.savefig('durch.pdf')

plt.clf()

err2 = np.sqrt(np.diag(cov2))

print('a2 = c2 =', param2[0]*10**4, '+-', err2[0]*10**4, 'm/s')
print('b2 =', param2[1], '+-', err2[1], 'cm')

c2 = unp.uarray(param2[0]*10**4, err2[0]*10**4)
print('Abweichung:', noms(np.abs(c2-ctheo)/ctheo * 100), '%')

cmit = (c1+c2)/2

print('c_mit = %4.2f +- %4.2f' % (noms(cmit), stds(cmit)))

abw = np.abs(cmit-ctheo)/ctheo * 100

print('Abweichung = %0.2f %%' % (noms(abw)))

# Dämpfung mit dem Impuls-Echo-Verfahren ## Irgendwas passt hier nicht

l, U1, U2, dt = np.genfromtxt('echo.txt', unpack=True)
lU = np.log(U2)

param3, cov3 = curve_fit(f, l, lU)
x_plot = np.linspace(3, 12.5, 10000)
plt.plot(x_plot, f(x_plot, *param3), 'b-', label= 'Fit')

plt.plot(l, lU, 'rx', label='Messwerte')
plt.xlabel(r'$L \:/\: \mathrm{cm}$')
plt.ylabel(r'$\lg{(U)}$')
# plt.xlim(0, 46)
plt.legend()
# plt.show()
# plt.savefig('daem.pdf')

plt.clf()

err3 = np.sqrt(np.diag(cov3))

print('a =', param3[0]*10**2, '+-', err2[0]*10**2, '1/m')
print('b =', param3[1], '+-', err3[1])

a = unp.uarray(param3[0]*10**2, err3[0]*10**2)

alth = 57
abw = np.abs(alth+a)/alth * 100

print('Abweichung:  %0.2f %%' % (noms(abw)))


# FFT- / Cepstrum-Zeug

## Cepstrum-Werte
t11 = 4.9e-6
t12 = 8.74e-6
t13 = 13.21e-6

## A-Scan-Werte
t21 = 4.8e-6
t22 = 8.6e-6
t23 = 13.4e-6

d1th = 0.6
d2th = 1.2
d3th = 1.8

d1c = cmit*t11/2*10**2
d2c = cmit*t12/2*10**2
d3c = cmit*t13/2*10**2

d1s = cmit*t21/2*10**2
d2s = cmit*t22/2*10**2
d3s = cmit*t23/2*10**2

print('d1_Cep =', d1c, 'cm')
print('d2_Cep =', d2c, 'cm')
print('d3_Cep =', d3c, 'cm')
print('d1_Scan =', d1s, 'cm')
print('d2_Scan =', d2s, 'cm')
print('d3_Scan =', d3s, 'cm')

d1m = (d1c+d1s)/2
d2m = (d2c+d2s)/2
d3m = (d3c+d3s)/2

print('d1_mit =', d1m, 'cm')
print('d2_mit =', d2m, 'cm')
print('d3_mit =', d3m, 'cm')

abw1 = np.abs(d1m-d1th)/d1th * 100
abw2 = np.abs(d2m-d2th)/d2th * 100
abw3 = np.abs(d3m-d3th)/d3th * 100

print('Abweichung 1:  %0.2f %%' % (noms(abw1)))
print('Abweichung 2:  %0.2f %%' % (noms(abw2)))
print('Abweichung 3:  %0.2f %%' % (noms(abw3)))


# Auge

t1 = 12.1e-6
t2 = 17.8e-6
t3 = 26.9e-6
t4 = 70.5e-6

c1 = 1483
c2 = 2500
c3 = 1410

s1 = c1*t1/2
s2 = c1*(t2-t1)/2
s3 = c2*(t3-t2)/2
s4 = c3*(t4-t3)/2

print('Hornhaut - Iris: %0.2f cm' % (s1*100))
print('Iris - Anfang Linse: %0.2f cm' % (s2*100))
print('Anfang Linse - Ende Linse: %0.2f cm' % (s3*100))
print('Ende Linse - Retina: %0.2f cm' % (s4*100))

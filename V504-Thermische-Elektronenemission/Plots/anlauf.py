import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const

xx, yy = np.genfromtxt('anlauf.txt', unpack='true')
x = xx - 1e6*yy*10**(-9)
print(x)
y = np.log(yy*10**(-9))

e = const.value('electron volt')
k = const.value('Boltzmann constant')
h = const.value('Planck constant')
m = const.value('electron mass')

def f(x, a, b):
    return a*x+b

param, cov = curve_fit(f, x, y)
x_plot = np.linspace(-0.02, 1, 1000)
plt.plot(x_plot, f(x_plot, *param), 'b-', label='Fit')

plt.plot(x, y, 'rx', label='Messwerte')
plt.xlabel(r'$U \,/\, \mathrm{V}$')
plt.ylabel(r'$\log{(I)}$')
plt.legend()
# plt.show()
# plt.savefig('anlauf.pdf')


err = np.sqrt(np.diag(cov))

# print(param[0], '+-', err[0])

T = - e/(k*param[0])
errT = np.abs(err[0]*e/(k*param[0]**2))

print(T, '+-', errT, 'K')

x = [2.0, 2.1, 2.3, 2.4, 2.5]
y = [3.5, 4.1, 4.7, 5.2, 5.6]
sig = 5.7e-12
ff = 0.32
eta = 0.28
Temp = []

i = 0
while i<5:
    tt = ((x[i]*y[i]-1)/(ff*eta*sig))**(1/4)
    Temp.append(tt)
    i += 1

print('Temperatur =', Temp)

satt = [0.088, 0.207, 0.685]
phi = []

i = 0
while i<3:
    ss = - (k*Temp[i])/e * np.log((satt[i]*h**3)/(4*np.pi*e*m*ff*k**2*Temp[i]**2))
    phi.append(ss)
    i += 1

print('Austrittarbeit =', phi)

phim = (phi[0]+phi[1]+phi[2])/3
errphim = np.sqrt(1/6 * ((phim-phi[0])**2+(phim-phi[1])**2+(phim-phi[2])**2))

print(phim, '+-', errphim)

lit = 4.54

abw = np.abs(lit-phim)/lit * 100
print(abw, '%')

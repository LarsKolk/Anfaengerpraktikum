import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

x, y = np.genfromtxt('freq.txt', unpack='true')
x *= 1e3

Rg = 50
R2 = 271.6
R = Rg+R2
errR2 = 0.3
L = 3.53e-3
errL = 0.03e-3
C = 5.015e-9
errC = 0.015e-9

t = np.linspace(0, 19, 25)

nu = np.linspace(0, 70)


def Uc(t, U0):
    return U0 / np.sqrt((1 - L*C*(2*np.pi*t)**2)**2 + (2*np.pi*t*R*C)**2)


param, cov = curve_fit(Uc, x, y)
x_plot = np.linspace(0, 1000, 10000)*10**3
plt.plot(x_plot, Uc(x_plot, *param), 'b-', label='Theoriekurve')


xposition = [23*10**3, 45*10**3]
for xc in xposition:
    plt.axvline(x=xc, color='xkcd:grey', linestyle='--')

plt.plot(x, y, 'rx', label='Messwerte')
plt.plot((0, 56*10**3), (12.59, 12.59), color='xkcd:grey', linestyle='--')
plt.plot(23*10**3, 12.59, 'k.')
plt.plot(45*10**3, 12.59, 'k.')
plt.xlabel(r'$\nu \,/\, \mathrm{Hz}$')
plt.ylabel(r'$U_c \,/\, \mathrm{V}$')
plt.xlim(10*10**3, 56*10**3)
plt.ylim(10, 19)
plt.text(20*10**3, 13, r'$\nu_-$', fontsize=12)
plt.text(45.5*10**3, 12.9, r'$\nu_+$', fontsize=12)


plt.grid()
plt.legend()

# plt.show()
plt.savefig('freq.pdf')

print('Experimentelle Werte')

q1 = 1/(C*(Rg+R2)*35.5e3*2*np.pi)
errq1 = np.sqrt((-1/((R2+Rg)**2*C*35.5e3*2*np.pi)*errR2)**2+(-1/((R2+Rg)*C**2*35.5e3*2*np.pi)*errC)**2)
print('q =', q1, '+-', errq1)

y = (45-23)*10**3
print('Breite der Resonanzkurve =', y)

print('Theoriewerte')

q2 = np.sqrt(L*C)/((R2+Rg)*C)
errq2 = np.sqrt((1/2*1/(R2+Rg)*1/np.sqrt(L*C)*errL)**2+(-1/2*1/(R2+Rg)*np.sqrt(L/C**3)*errC)**2+(-np.sqrt(L/C)*1/(R2+Rg)**2*errR2)**2)
print('q =', q2, '+-', errq2)

x = (Rg+R2)/(2*np.pi*L)
errx = np.sqrt((-(Rg+R2)/(2*np.pi*L**2)*errL)**2+(1/(2*np.pi*L)*errR2)**2)

print('Breite der Resonanzkurve =', x, '+-', errx)

print('Abweichung')
xx = np.abs((x-y)/x)

print('... von der Breite der Resonanzkurve =', xx)

yy = np.abs((q2-q1)/q2)
print('... von q =', yy)

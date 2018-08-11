import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

x, y = np.genfromtxt('knoise.txt', unpack='true')
U0 = 10


def f(x, a, b, c):
    return a * np.cos(b*x + c)


t = np.linspace(0, 360, 1000)
Uout = 2/np.pi * U0 * np.cos(t*np.pi/180)

param, cov = curve_fit(f, x, y, p0=(1, 1/60, 1))
x_plot = np.linspace(0, 360, 6000)
plt.plot(x_plot, f(x_plot, *param), 'b-', label='Fit')

plt.plot(t, Uout, 'g--', label='Theoriekurve')
plt.plot(x, y, 'rx', label='Messwerte')
plt.xlabel(r'$\Phi \,/\, \mathrm{°}$')
plt.ylabel(r'$U \,/\, \mathrm{V}$')
plt.xlim(0, 360)
plt.ylim(-7, 7)

plt.legend()
plt.grid()
# plt.show()
# plt.savefig('knoise.pdf')

err = np.sqrt(np.diag(cov))

print('Experimentelle Werte')
print('Amplitude =', param[0], '+-', err[0])
print('Frequenz =', param[1], '+-', err[1])
print('Phase =', param[2]*180/np.pi, '+-', err[2]*180/np.pi)

print('Theoriewerte')
print('Amplitude =', 2/np.pi * U0)
print('Frequenz =', np.pi/180)
print('Phase =', 0)

print('Abweichung')
xx = np.abs((2/np.pi * U0-param[0])/(2/np.pi * U0))*100
yy = np.abs((np.pi/180-param[1])/(np.pi/180))*100

print('... der Amplitude =', xx, '%')
print('... der Frequenz =', yy, '%')

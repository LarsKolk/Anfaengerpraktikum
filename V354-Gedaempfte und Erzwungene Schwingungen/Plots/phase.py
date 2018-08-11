import matplotlib.pyplot as plt
import numpy as np


x, y = np.genfromtxt('phase.txt', unpack='true')
x *= 1e3

Rg = 50
R2 = 271.6
R = Rg + R2
errR = 0.3
L = 3.53e-3
errL = 0.03e-3
C = 5.015e-9
errC = 0.015e-9

t = np.linspace(1e-10, 100, 10000)*10**3
phi = np.arctan(((2*np.pi*t)**2*L*C-1)/(2*np.pi*t*R*C))*180/np.pi+92.5

yposition = [45, 135]
for yc in yposition:
    plt.axhline(y=yc, color='xkcd:grey', linestyle='--')

xposition = [31.75*10**3, 42.5*10**3]
for xc in xposition:
    plt.axvline(x=xc, color='xkcd:grey', linestyle='--')

plt.axhline(y=76.75, color='g', linestyle='--')
plt.axvline(x=35.5*10**3, color='g', linestyle='--')

plt.plot(35.5*10**3, 76.75, 'k.')
plt.plot(31.75*10**3, 45, 'k.')
plt.plot(42.5*10**3, 135, 'k.')
plt.text(36*10**3, 69.75, r'$\nu_\mathrm{res}$', fontsize=12)
plt.text(32.25*10**3, 38, r'$\nu_2$', fontsize=12)
plt.text(43*10**3, 128, r'$\nu_1$', fontsize=12)

plt.plot(t, phi, 'b-', label='Theoriekurve')

plt.plot(x, y, 'rx', label='Messwerte')
plt.xlabel(r'$\nu \,/\, \mathrm{Hz}$')
plt.ylabel(r'$\Phi \,/\, \mathrm{°}$')
plt.xlim(10*10**3, 56*10**3)
plt.ylim(0, 160)

plt.grid()
plt.legend()
# plt.show()
plt.savefig('phase.pdf')

print('Experimentelle Werte')
print('ny_res =', 35.5*10**3)
print('ny_1 =', 42.5*10**3)
print('ny_2 =', 31.75*10**3)

print('Theoriewerte')

a = -1/(4*np.pi*L*C**2*np.sqrt(1/(L*C)-R**2)/(2*L**2))
b = -R/(4*np.pi*L**2*np.sqrt(1/(L*C)-R**2/(2*L**2)))
c = 1/(4*np.pi*np.sqrt(1/(L*C)-R**2/(2*L**2)))*(-1/(L**2*C)+2*R**2/(2*L**3))

nyres = np.sqrt(1/(L*C)-R**2/(2*L**2))/(2*np.pi)
errnyres = np.sqrt((a*errC)**2+(b*errR)**2+(c*errL)**2)
print('ny_res =', nyres, '+-', errnyres)

ny1 = 1/(2*np.pi)*(R/(2*L)+np.sqrt((R**2/(4*L**2))+(1/(L*C))))
d = -1/(4*np.pi*L*C**2*np.sqrt((R**2/(4*L**2)+1/(L*C))))
e = 1/(2*L)*(1/(2*L)+R/(4*L**2*np.sqrt(R**2/(4*L**2)+1/(L*C))))
f = 1/(2*np.pi)*(-R/(2*L**2)+1/(2*np.sqrt(R**2/(4*L**2)+1/(L*C)))*((-2*R**2)/(4*L**3)-1/(L**2*C)))
errny1 = np.sqrt((d*errC)**2+(e*errR)**2+(f*errL)**2)

print('ny_1 =', ny1, '+-', errny1)

ny2 = 1/(2*np.pi)*(-R/(2*L)+np.sqrt((R**2/(4*L**2))+(1/(L*C))))
g = 1/(2*L)*(-1/(2*L)+R/(4*L**2*np.sqrt(R**2/(4*L**2)+1/(L*C))))
h = 1/(2*np.pi)*(R/(2*L**2)+1/(2*np.sqrt(R**2/(4*L**2)+1/(L*C)))*((-2*R**2)/(4*L**3)-1/(L**2*C)))

errny2 = np.sqrt((d*errC)**2+(g*errR)**2+(h*errL)**2)

print('ny_2 =', ny2, '+-', errny2)

print('Abweichung')

xx = np.abs((nyres-35.5*10**3)/nyres)
print('... von ny_res =', xx)

yy = np.abs((ny1-42.5*10**3)/ny1)
print('... von ny_1 =', yy)

zz = np.abs((ny2-31.75*10**3)/ny2)
print('... von ny_2 =', zz)

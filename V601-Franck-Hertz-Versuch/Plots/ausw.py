import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const

e0 = const.value('electron volt')
h0 = const.value('Planck constant')
c0 = const.value('speed of light in vacuum')
E1_theo = 4.9 #eV
Uion_theo = 10.437

## mittlere freie Weglänge
T = np.genfromtxt('weg.txt')
T += 273.15

psatt = []

for i in T:
    psatt.append(5.5e7 * np.exp(-6876/i))

w = []
for i in psatt:
    w.append(0.0029/i)

verh = []
for i in w:
    verh.append(1/i)

print(verh)

# np.savetxt('test.txt', np.column_stack([T, w, verh]), header='T w Verhaeltnis', fmt='%0.2f')

## Skalierungsfaktoren berechnen
a, b = np.genfromtxt('skal.txt', unpack='true')
c, d = np.genfromtxt('skal2.txt', unpack='true')

f=[]
fx=0

for ax in a:
    f.append(2/ax)

f1 = (sum(f)/len(f))

i = 0
while i<len(f):
    fx += (f1-f[i])**2
    i += 1

f1err = np.sqrt(1/(len(f)*(len(f)-1))*fx)
print(f1, '+-', f1err)

f=[]
fx=0

for bx in b:
    f.append(2/bx)

f2 = (sum(f)/len(f))

i = 0
while i<len(f):
    fx += (f2-f[i])**2
    i += 1

f2err = np.sqrt(1/(len(f)*(len(f)-1))*fx)
print(f2, '+-', f2err)

f=[]
fx=0

for cx in c:
    f.append(10/cx)

f3 = (sum(f)/len(f))

i = 0
while i<len(f):
    fx += (f3-f[i])**2
    i += 1

f3err = np.sqrt(1/(len(f)*(len(f)-1))*fx)
print(f3, '+-', f3err)

f=[]
fx=0

for dx in d:
    f.append(10/dx)

f4 = (sum(f)/len(f))

i = 0
while i<len(f):
    fx += (f4-f[i])**2
    i += 1

f4err = np.sqrt(1/(len(f)*(len(f)-1))*fx)
print(f4, '+-', f4err)

## Steigungen
a11, a12 = np.genfromtxt('a1.txt', unpack='true')
a11 *= f1

i = 0
sta11 = []
sta12 = []
while i<len(a11)-1:
    sta11.append((a11[i]+a11[i+1])/2)
    sta12.append((a12[i+1]-a12[i])/(a11[i+1]-a11[i]))
    i += 1

plt.plot(sta11, sta12, 'rx', label='berechnete Werte')
plt.xlabel(r'$U_A \: / \: \mathrm{V}$')
# plt.axvline(9.231617647058824, color='green', linestyle='--')
plt.ylabel('Steigung')
plt.legend()
# plt.savefig('a1.pdf')
# plt.show()

plt.clf()

a21, a22 = np.genfromtxt('a2.txt', unpack='true')
a21 *= f2
i = 0
sta21 = []
sta22 = []
while i<len(a21)-1:
    sta21.append((a21[i]+a21[i+1])/2)
    sta22.append((a22[i+1]-a22[i])/(a21[i+1]-a21[i]))
    i += 1

plt.plot(sta21, sta22, 'rx', label='berechnete Werte')
plt.xlabel(r'$U_A \: / \: \mathrm{V}$')
plt.ylabel('Steigung')
plt.legend()
# plt.savefig('a2.pdf')
# plt.show()

plt.clf()

## Kontaktpotential 1
kk = 11-9.23
print('k=', kk, 'V')

## Franck-Hertz Abstände
xx = np.genfromtxt('b.txt', unpack='true')
xx *= f3
sx = 0

xmit = sum(xx)/len(xx)

for i in xx:
    sx += (xmit-i)**2

xerr = np.sqrt(1/(len(xx)*(len(xx)-1))*sx)

print(xmit, '+-', xerr, 'V')

## Anregungsenergie
E1 = e0 * xmit

E1err = np.abs(e0 * xerr)

print('E1 =', E1, '+-', E1err, 'J')
abwE1 = np.abs(xmit-E1_theo)/E1_theo * 100
print(abwE1, '%')

## emittierte Wellenlänge
lam = (h0*c0)/E1
errlam = np.abs(-(h0*c0)/E1**2 * E1err)
print('lambda =' , lam*10**9, '+-', errlam*10**9, 'nm')

# ## Kontaktpotential 2
# kk2 = 1.8*f3 - xmit
# print('k2 =', kk2, 'V')

# Ionisierungsenergie
x, y = np.genfromtxt('c.txt', unpack='true')
y += -6.75
x *= f4

def j(x, a, b):
    return a*x+b

param, cov = curve_fit(j, x, y)
x_plot = np.linspace(18, 33, 1000)
plt.axhline(0, color='green', linestyle='--')
# plt.axvline(1.13+18)
plt.plot(x_plot, j(x_plot, *param), 'b-', label= 'Fit')
plt.plot(x, y, 'rx', label='Messwerte')
plt.xlabel(r'$U_B \: / \: \mathrm{V}$')
plt.ylabel(r'$I_S$')
plt.legend()
# plt.show()
# plt.savefig('c.pdf')

err = np.sqrt(np.diag(cov))

schnitt = -param[1]/param[0]

Uion = schnitt - kk
errUion = np.sqrt((-1/param[0] * err[1])**2 + (param[1]/param[0]**2 * err[0])**2)
print('Uion =', Uion, '+-', errUion)

abwUion = np.abs(Uion-Uion_theo)/Uion_theo *100
print(abwUion, '%')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const

d = 201.4e-12
e0 = const.value('electron volt')
h0 = const.value('Planck constant')
c0 = const.value('speed of light in vacuum')
R0 = const.value('Rydberg constant times hc in eV')
a0 = const.value('fine-structure constant')

## Bragg Bedingung überprüfen

x, y = np.genfromtxt('Werte/bragg.txt', unpack='true')

plt.plot(x, y, 'r-', label='Messwerte')
plt.axvline(27.6, color='b', linestyle='--', label='Glanzwinkel')
plt.xlabel(r'$2 \vartheta \: / \: \mathrm{°}$')
plt.ylabel(r'$\mathrm{Rate} \: / \: \mathrm{\frac{Imp}{s}}$')
plt.legend()
# plt.show()
# plt.savefig('bragg.pdf')

plt.clf()

abw = np.abs(28-27.6)/28 * 100
print('Abweichung Glanzwinkel:', abw, '%')

## Emissionsspektrum

x, y = np.genfromtxt('Werte/emission.txt', unpack='true')

plt.plot(x, y, 'r-', label='Messwerte')
plt.axvline(40.4, color='b', linestyle='--', label=r'$K_\alpha$')
plt.axvline(44.4, color='k', linestyle='--', label=r'$K_\beta$')
plt.xlabel(r'$2 \vartheta \: / \: \mathrm{°}$')
plt.ylabel(r'$\mathrm{Rate} \: / \: \mathrm{\frac{Imp}{s}}$')
# plt.axvline(x[4], color='k', linestyle='--')
plt.legend()
# plt.show()
# plt.savefig('emission.pdf')

plt.clf()

lam = 2*d*np.sin(np.deg2rad(x[4]/2))
print(x[4]/2)
print('lam =', lam, 'm')

lamtheo = (h0 * c0)/(e0 * 35*10**3)
print('lamtheo =', lamtheo, 'm')

lamab = np.abs(lam-lamtheo)/lamtheo * 100
print('Abweichung:', lamab, '%')

# Auflösungsvermögen
plt.plot(x[77:85], y[77:85])
plt.axvline(39.75)
plt.axvline(40.622)
# plt.show()

plt.clf()

E2 = (h0*c0)/(2*d*np.sin(np.deg2rad(40.622/2)))/e0
E1 = (h0*c0)/(2*d*np.sin(np.deg2rad(39.75/2)))/e0

dEb = (E1-E2)
print(dEb)

plt.plot(x[88:96], y[88:96])
plt.axvline(44.2)
plt.axvline(45.29)
# plt.show()

plt.clf()

E22 = (h0*c0)/(2*d*np.sin(np.deg2rad(45.29/2)))/e0
E12 = (h0*c0)/(2*d*np.sin(np.deg2rad(44.2/2)))/e0

dEa = (E12-E22)
print(dEa)

mitdE = (dEb+dEa)/2
errmit = np.sqrt(1/2 * ((mitdE-dEb)**2+(mitdE-dEa)**2))
print('Mittel =', mitdE, '+-', errmit, 'eV')

# Abschirmkonstanten

Ealpha = (h0*c0)/(2*d*np.sin(np.deg2rad(44.4/2)))/e0
Ebeta = (h0*c0)/(2*d*np.sin(np.deg2rad(40.4/2)))/e0
print('E_Alpha =', Ealpha, 'eV')
print('E_Beta =', Ebeta, 'eV')

sig1 = 29 - np.sqrt(Ebeta/R0)
print('Sigma1 =', sig1)

sig2 = 29 - 2*np.sqrt((29-sig1)**2-Ealpha/R0)
print('Sigma2 =', sig2)

Eathe = 8.05e3
Ebthe = 8.91e3

sig1t = 29 - np.sqrt(Ebthe/R0)
print('Sigma1_theo =', sig1t)

sig2t = 29 - 2*np.sqrt((29-sig1)**2-Eathe/R0)
print('Sigma2_theo =', sig2t)

abw1 = np.abs(sig1-sig1t)/sig1t * 100
abw2 = np.abs(sig2-sig2t)/sig2t * 100

print('Abweichung sig1:', abw1, '%')
print('Abweichung sig2:', abw2, '%')

## Brom

x, y = np.genfromtxt('Werte/brom.txt', unpack='true')

plt.plot(x, y, 'r-', label='Messwerte')
plt.axvline(26.6, color='b', linestyle='--', label='K-Linie')
plt.xlabel(r'$2 \vartheta \: / \: \mathrm{°}$')
plt.ylabel(r'$\mathrm{Rate} \: / \: \mathrm{\frac{Imp}{s}}$')
plt.legend()
# plt.show()
# plt.savefig('brom.pdf')

plt.clf()

Ebrom = (h0*c0)/(2*d*np.sin(np.deg2rad(26.6/2)))/e0
print('E_Brom =', Ebrom, 'eV')
sig = 35 - np.sqrt(Ebrom/R0)
print('Sigma =', sig)
abwsig = np.abs(sig-3.52)/3.52 * 100
print('Abweichung:', abwsig, '%')

## Zink

x, y = np.genfromtxt('Werte/zink.txt', unpack='true')

plt.plot(x, y, 'r-', label='Messwerte')
plt.axvline(37.2, color='b', linestyle='--', label='K-Linie')
plt.xlabel(r'$2 \vartheta \: / \: \mathrm{°}$')
plt.ylabel(r'$\mathrm{Rate} \: / \: \mathrm{\frac{Imp}{s}}$')
plt.legend()
# plt.show()
# plt.savefig('zink.pdf')

plt.clf()

Ezink = (h0*c0)/(2*d*np.sin(np.deg2rad(37.2/2)))/e0
print('E_Zink =', Ezink, 'eV')
sig = 30 - np.sqrt(Ezink/R0)
print('Sigma =', sig)
abwsig = np.abs(sig-3.37)/3.37 * 100
print('Abweichung:', abwsig, '%')

## Strontium

x, y = np.genfromtxt('Werte/strontium.txt', unpack='true')

plt.plot(x, y, 'r-', label='Messwerte')
plt.axvline(22.2, color='b', linestyle='--', label='K-Linie')
plt.xlabel(r'$2 \vartheta \: / \: \mathrm{°}$')
plt.ylabel(r'$\mathrm{Rate} \: / \: \mathrm{\frac{Imp}{s}}$')
plt.legend()
# plt.show()
# plt.savefig('strontium.pdf')

plt.clf()

Estront = (h0*c0)/(2*d*np.sin(np.deg2rad(22.2/2)))/e0
print('E_Strontium =', Estront, 'eV')
sig = 38 - np.sqrt(Estront/R0)
print('Sigma =', sig)
abwsig = np.abs(sig-3.58)/3.58 * 100
print('Abweichung:', abwsig, '%')

## Zirkonium

x, y = np.genfromtxt('Werte/zirkonium.txt', unpack='true')

plt.plot(x, y, 'r-', label='Messwerte')
plt.axvline(19.8, color='b', linestyle='--', label='K-Linie')
plt.xlabel(r'$2 \vartheta \: / \: \mathrm{°}$')
plt.ylabel(r'$\mathrm{Rate} \: / \: \mathrm{\frac{Imp}{s}}$')
plt.legend()
# plt.show()
# plt.savefig('zirkonium.pdf')

plt.clf()

Ezir = (h0*c0)/(2*d*np.sin(np.deg2rad(19.8/2)))/e0
print('E_Zirkonium =', Ezir, 'eV')
sig = 40 - np.sqrt(Ezir/R0)
print('Sigma =', sig)
abwsig = np.abs(sig-3.62)/3.62 * 100
print('Abweichung:', abwsig, '%')

## Moseley-Gesetz

EK = [Ebrom, Ezink, Estront, Ezir]
Z = [35, 30, 38, 40]

x = Z
y = np.sqrt(EK)

def j(x, a, b):
    return a*x+b

param, cov = curve_fit(j, x, y)
x_plot = np.linspace(28, 42, 1000)
plt.plot(x_plot, j(x_plot, *param), 'b-', label= 'Fit')

plt.plot(x, y, 'rx', label='Werte')
plt.xlabel(r'$Z$')
plt.ylabel(r'$\sqrt{E_K} \: / \: \sqrt{\mathrm{eV}}$')
plt.legend()
# plt.show()
# plt.savefig('moseley.pdf')

plt.clf()

err = np.sqrt(np.diag(cov))

Eryd = param[0]**2
errEryd = np.sqrt((2*param[0]*err[0])**2)
abw = np.abs(Eryd-R0)/R0 * 100
print('E_Ryd =', Eryd, '+-', errEryd, 'eV')
print('Abweichung =', abw, '%')

## Quecksilber

x, y = np.genfromtxt('Werte/quecksilber.txt', unpack='true')

plt.plot(x, y, 'r-', label='Messwerte')
plt.axvline(24.8, color='b', linestyle='--', label=r'$\mathrm{L_{II}-Kante}$')
plt.axvline(29.2, color='k', linestyle='--', label=r'$\mathrm{L_{III}-Kante}$')
plt.xlabel(r'$2 \vartheta \: / \: \mathrm{°}$')
plt.ylabel(r'$\mathrm{Rate} \: / \: \mathrm{\frac{Imp}{s}}$')
plt.legend()
# plt.show()
# plt.savefig('quecksilber.pdf')

plt.clf()

ELII = (h0*c0)/(2*d*np.sin(np.deg2rad(24.8/2)))/e0
ELIII = (h0*c0)/(2*d*np.sin(np.deg2rad(29.2/2)))/e0
print('E_II =', ELII, 'eV')
print('E_III =', ELIII, 'eV')
delta = ELII - ELIII
print('E_II - E_III =', delta, 'eV')
deltath = 14.21e3 - 12.3e3
sigL = 80 - np.sqrt((4/a0*np.sqrt(delta/R0) - 5*delta/R0)*(1 + 19/32*a0**2*delta/R0))
sigtheo = 80 - np.sqrt((4/a0*np.sqrt(deltath/R0) - 5*delta/R0)*(1 + 19/32*a0**2*deltath/R0))

print('Sigma_L =', sigL)
print('Sigma_theo =', sigtheo)

abw = np.abs(sigtheo-sigL)/sigtheo * 100
print('Abweichung:', abw, '%')

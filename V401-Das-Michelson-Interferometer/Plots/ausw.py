import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const
import uncertainties.unumpy as unp
import matplotlib.mlab as mlab
from scipy.stats import poisson
from scipy.special import factorial

def noms(x):
    return unp.nominal_values(x)

def stds(x):
    return unp.std_devs(x)

# Wellenlände des Lasers bestimmen

s1, s2, z = np.genfromtxt('lam.txt', unpack=True)

s = np.abs(s1-s2) *10**(-3) # in m

a = s[:-2]/5.017
b = s[-2:]/5.046

d = np.concatenate((a,b))
print('d =', d*10**3, 'mm')

well = 2*d/z

mitlam = sum(well)/len(well)
errlam = np.sqrt(1/(len(well)*(len(well)-1)) * sum((well-mitlam)**2))

lam = unp.uarray(mitlam, errlam)

lamtheo = 635e-9 #m

abw = np.abs(lamtheo-lam)/lamtheo * 100

print('Wellenlaenge des Lasers:', lam*10**9, 'nm')
print('Theoriewert:', lamtheo*10**9, 'nm')
print('Abweichung: %0.2f %%' % noms(abw))

fehl = np.abs(lamtheo-lam)/stds(lam)
print('Fehlerintervall: %0.2f' % (noms(fehl)))

# np.savetxt('../Text/Tabellen/lam.txt', np.column_stack([s*10**3, d*10**3, z, well*10**9]), fmt='%0.2f %0.2f  %0.0f  %0.2f', header='s d z \lambda \n mm mm - nm')

# Brechungsindex von Luft bestimmen

dp, z = np.genfromtxt('n.txt', unpack=True)

p0 = 1.0132 #bar
b = 50e-3 #m
t0 = 273.15 #K
t = 294.15 #K


xn = 1 + z * lamtheo * t * p0 / (2 * b * t0 * dp)

mitn = sum(xn)/len(xn)
errn = np.sqrt(1/(len(xn)*(len(xn)-1)) * sum((xn-mitn)**2))

n = unp.uarray(mitn, errn)
ntheo = 1.00028

abw = np.abs(n-ntheo)/ntheo * 100

print('Brechungsindex von Luft:', n)
print('Theoriewert:', ntheo)
print('Abweichung: %0.5f %%' % noms(abw))

fehl = np.abs(ntheo-n)/stds(n)
print('Fehlerintervall: %0.2f' % (noms(fehl)))

np.savetxt('../Text/Tabellen/n.txt', np.column_stack([dp, z, xn]), fmt='%0.2f  %0.0f  %0.5f', header='p" z n \n bar - -')

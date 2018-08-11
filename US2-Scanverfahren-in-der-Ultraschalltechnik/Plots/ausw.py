import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const
import uncertainties.unumpy as unp


def noms(x):
    return unp.nominal_values(x)

def stds(x):
    return unp.std_devs(x)

# Fehlstellen mit A-Scan
n, dtheo, toben, tunten = np.genfromtxt('mess.txt', unpack=True)
dtheo *= 1e-2
toben *= 1e-6
tunten *= 1e-6
h = 8e-2
c = 2730

soben = 1/2*toben*c-0.2e-2 # Abstand Oberseite-Loch
sunten = 1/2*tunten*c-0.2e-2 # Abstand Unterseite-Loch

d = (h-soben-sunten) # Durchmesser der Löcher

abw1 = np.abs(dtheo-d)/dtheo * 100

print(d*1000, 'mm')
print(abw1, '%')

# Fehlstellen mit dem B-Scan
n2, xx, yy = np.genfromtxt('bwerte.txt', unpack=True)

uf = 5e-6/27 # Umrechnungsfaktor s/px

bsoben = uf * xx * c/2 - 0.2e-2
bsunten = uf * yy * c/2 - 0.2e-2

btoben = uf * xx
btunten = uf * yy

bd = h-bsoben-bsunten

abw2 = np.abs(dtheo-bd)/dtheo * 100

print(bd*1000, 'mm')
print(abw2, '%')


# Herzmodell
uh = 1/36 # Umrechnungsfaktor s/px

t, h = np.genfromtxt('herz.txt', unpack=True)
t *= uh
h *= uf

hm = sum(h)/len(h)
herr = np.sqrt(1/(len(h)*(len(h)-1))*sum((h-hm)**2))

hmit = unp.uarray(hm, herr)
print('Hoehe in s:', hmit, 's')


tm = sum(t)/(len(t)-1)
terr = np.sqrt(1/((len(t)-1)*(len(t)-2))*(sum((t-tm)**2)-(t[0]-tm)**2))

tmit = unp.uarray(tm, terr)
print('Periodendauer:', tmit, 's')

f = 1/tmit
print('Herzfrequenz:', f, 'Hz')


Dh = 5e-2
a = Dh/2
tsys = 34.49e-6
tdia = 47.51e-6
cw = 1483

s = cw/2 * hmit
print('Hoehe:', s*100, 'cm')

V = s*np.pi/6*(3*a**2+s**2)
print('Schlagvolumen:', V*10**6, 'ml')
print('Herzzeitvolumen:', V*10**6*f,'ml/s', '=', V*10**3*f*60, 'l/min')

# np.savetxt('blockA.txt', np.column_stack([n, dtheo*10**2, toben*10**6, soben*10**3, tunten*10**6, sunten*10**3, d*10**3, abw1]), fmt='%0.0f %0.1f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f', header='Lochnummer dtheo toben soben tunten sunten d Abweichung \n - cm mus mm mus mm mm -')
# np.savetxt('blockB.txt', np.column_stack([n, dtheo*10**2, btoben*10**6, btunten*10**6, bd*10**3, abw2]), fmt='%0.0f %0.1f %0.2f %0.2f %0.2f %0.2f', header='Lochnummer dtheo toben tunten d Abweichung \n - cm mus mus mm -')
# np.savetxt('werte1.txt', np.column_stack([n, dtheo*10**2, toben*10**6, soben*10**3, tunten*10**6, sunten*10**3, d*10**3, abw1, btoben*10**6, btunten*10**6, bd*10**3, abw2]), fmt='%0.0f %0.1f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f')
# np.savetxt('werte2.txt', np.column_stack([t, h*10**6]), fmt='%0.2f %0.2f', header='t h \n s mus')

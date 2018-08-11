import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const
import uncertainties.unumpy as unp
import matplotlib.mlab as mlab
from scipy.stats import poisson
from scipy.special import factorial
from newline import newline

def noms(x):
    return unp.nominal_values(x)

def stds(x):
    return unp.std_devs(x)

# Phi berechnen

fl, fr = np.genfromtxt('phi.txt', unpack=True)

f = 1/2 * (fr-fl)

# np.savetxt('wphi.txt', np.column_stack([fl, fr, f]), fmt='%0.1f  %0.1f  %0.1f', header='\phi_l \phi_r \phi \n ° ° °')

mitphi = sum(f)/len(f)
errphi = np.sqrt(1/(len(f)*(len(f)-1)) * sum((f-mitphi)**2))

phi = unp.uarray(mitphi, errphi)

print('Phi = %0.2f +- %0.2f "Grad"' % (noms(phi), stds(phi)))

# Eta berechnen

ly, l, omr, oml = np.genfromtxt('omega.txt', unpack=True)

eta = 180 - (omr - oml)


# Brechungsindex
foo = (eta+phi)/2
bar = phi/2

m = np.sin(np.deg2rad(noms(foo)))/np.sin(np.deg2rad(noms(bar)))

bla = np.cos(np.deg2rad((eta+noms(phi))/2))
bli = np.sin(np.deg2rad((eta+noms(phi))/2))
blab = np.sin(np.deg2rad(noms(phi)/2))
blub = np.cos(np.deg2rad(noms(phi)/2))

errn = np.sqrt(((1/2 * bla * blab - 1/2 * bli * blub)/blab**2 * stds(phi))**2)

n = unp.uarray(m, errn)

# np.savetxt('../Text/Tabellen/n.txt', np.column_stack([l, oml, omr, eta, noms(n), stds(n)]), fmt='%0.1f  %0.1f  %0.1f %0.1f %0.2f\pm%0.2f', header='\lamda \omega_l \omega_r \eta n \n nm ° ° ° -')

def f(x, a, b):
    return a + b/x**2

def g(x, c, d):
    return c - d*x**2

param, cov = curve_fit(f, ly, noms(n)**2)
x_plot = np.linspace(400, 650, 10000)
plt.plot(x_plot, f(x_plot, *param), 'b-', label= r'$\: \lambda \gg \: \lambda_1 $')

param2, cov2 = curve_fit(g, ly, noms(n)**2)
x_plot2 = np.linspace(400, 650, 10000)
plt.plot(x_plot2, g(x_plot2, *param2), color='gold', linestyle='-', label= r'$\: \lambda \ll \: \lambda_1 $')

plt.plot(ly, noms(n)**2, 'rx', label='Messwerte')

# p1 = [l[0], noms(n[0])**2]
# p2 = [l[-1], noms(n[-1])**2]

# newline(p1,p2)

plt.xlabel(r'$\lambda \: / \: \mathrm{nm}$')
plt.ylabel(r'$n^2(\lambda)$')
plt.legend(loc='best')
plt.tight_layout()
# plt.show()
plt.savefig('n.pdf')

err = np.sqrt(np.diag(cov))
err2 = np.sqrt(np.diag(cov2))

s1 = 1/(len(n)-2) * sum((noms(n)**2 - param[0] - param[1]/ly**2)**2)
s2 = 1/(len(n)-2) * sum((noms(n)**2 - param2[0] - param2[1]*ly**2)**2)
print('s_n^2 =', s1)
print('s_n,^2 =', s2)

a = unp.uarray(param[0], err[0])
b = unp.uarray(param[1], err[1])

print('A0 =', a)
print('A2 =', b, 'nm^2')

print('A,0 =', param2[0], '+-', err2[0])
print('A,2 =', param2[1], '+-', err2[1], '1/nm^2')

# Abbesche Zahl
lc = 656
ld = 589
lf = 486

def h(x):
    return np.sqrt(noms(a)+noms(b)/x**2)

def errh(x):
    return np.sqrt((1/(2*np.sqrt(noms(a)+noms(b)/x**2)) * stds(a))**2 + (1/(2*x**2*np.sqrt(noms(a)+noms(b)/x**2)) * stds(b))**2)

nd = unp.uarray(h(ld), errh(ld))
nc = unp.uarray(h(lc), errh(lc))
nf = unp.uarray(h(lf), errh(lf))

print('nd =', nd)
print('nc =', nc)
print('nf =', nf)

nu = (nd - 1)/(nf - nc)

print('nu =', nu)

nutheo = 26.5
print('nu_theo =', nutheo)

abw = np.abs(nutheo-nu)/nutheo * 100

print('Abweichung: %0.2f %%' % noms(abw))

# Auflösungsvermögen

bre = 3e-2 * 10**9 #nm

def A(x):
    return bre * noms(b)/(x**3*np.sqrt(noms(a)+noms(b)/x**2))

def errA(x):
    return np.sqrt(((bre*noms(b))/(2*x**3*(noms(a)+noms(b)/x**2)**(3/2)) * stds(a))**2+(((2*noms(a)*x**2+noms(b))*bre)/(2*x**3*np.sqrt(noms(a)+noms(b)/x**2)*(noms(a)*x**2+noms(b))) * stds(b))**2)

Ad = unp.uarray(A(ld), errA(ld))
Ac= unp.uarray(A(lc), errA(lc))
Af = unp.uarray(A(lf), errA(lf))

print('Aufloesungsvermoegen lambda_d =', Ad)
print('Aufloesungsvermoegen lambda_c =', Ac)
print('Aufloesungsvermoegen lambda_f =', Af)

# Nächste Absorptionsstelle

l1a = np.sqrt(noms(b)/(noms(a)-1))
errl1 = np.sqrt(((noms(b)/(noms(a)-1))**(3/2)/(2*noms(b)) * stds(a))**2+(np.sqrt(noms(b)/(noms(a)-1))/(2*noms(b)) * stds(b))**2)

l1 = unp.uarray(l1a, errl1)

print('am naechsten gelegene Absorptionsstelle:', l1, 'nm')

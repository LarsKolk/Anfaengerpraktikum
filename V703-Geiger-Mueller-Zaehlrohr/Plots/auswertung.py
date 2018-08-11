import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const

U, n, I = np.genfromtxt('mess.txt', unpack='true')
errn = np.sqrt(n)
N = n/60
errN = errn/60

# Charackteristik

(_, caps, _) = plt.errorbar(U, N, yerr=errN, fmt='x', color='r', capsize=5, ecolor='b', label='Messwerte mit Poisson-Fehler')

for cap in caps:
    cap.set_markeredgewidth(1)

plt.xlabel(r'$U \:/\: \mathrm{V}$')
plt.ylabel(r'$N \:/\: \mathrm{Bq}$')
plt.legend()
# plt.show()
# plt.savefig('char.pdf')

plt.clf()

## Plateauanstieg

(_, caps, _) = plt.errorbar(U[7:-6], N[7:-6], yerr=errN[7:-6], fmt='x', color='r', capsize=5, ecolor='b', label='Plateau mit Poisson-Fehler')

for cap in caps:
    cap.set_markeredgewidth(1)

def f(x, a, b):
    return a*x+b

param, cov = curve_fit(f, U[7:-6], N[7:-6])
x_plot = np.linspace(300, 700, 1000)
plt.plot(x_plot, f(x_plot, *param), 'g-', label= 'Fit')

plt.xlabel(r'$U \:/\: \mathrm{V}$')
plt.ylabel(r'$N \:/\: \mathrm{Bq}$')
plt.legend()
# plt.show()
# plt.savefig('plateau.pdf')

plt.clf()

err = np.sqrt(np.diag(cov))
print('Steigung:', param[0], '+-', err[0], 'Bq/V')
print('y-Achsenabschnitt:', param[1], '+-', err[1], 'Bq')

st = (N[-7]-N[7])/(U[-7]-U[7]) * 100
print('Steigung %:', st, '%/100V')

# np.savetxt('werte1.txt', np.column_stack([U, n, N, errN]), fmt='%0.0f  %0.0f  %0.2f\pm%0.2f', header='U N N/t \n V - Bq')

# Totzeit

Utot = 550 #V

## Oszilloskop

tot = 80 #μs
erh = 240 #μs

## Zwei-Quellen-Methode

n1 = 36318
errn1 = np.sqrt(n1)
n2 = 30562
errn2 = np.sqrt(n2)
n12 = 65063
errn12 = np.sqrt(n12)

N1 = n1/120
errN1 = errn1/120
N2 = n2/120
errN2 = errn2/120
N12 = n12/120
errN12 = errn12/120

print('N1 =', N1, '+-', errN1, 'Bq')
print('N2 =', N2, '+-', errN2, 'Bq')
print('N12 =', N12, '+-', errN12, 'Bq')

tot2 = (N1+N2-N12)/(2*N1*N2) * 10**6 #μs
errtot2 = np.sqrt(((N12-N2)/(2*N1**2*N2) * errN1)**2+((N12-N1)/(2*N1*N2**2) * errN2)**2+(-1/(2*N1*N2) * errN12)**2) * 10**6 #μs

print('Totzeit1 =', tot)
print('Totzeit2 =', tot2, '+-', errtot2)

abw = np.abs(tot2-tot)/(tot2) * 100
print('Abweichung:', abw, '%')

# Ladungsfreisetzung
e0 = const.value('electron volt')

U, n, I = np.genfromtxt('mess0.txt', unpack='true')
I *= 1e-6
errn = np.sqrt(n)
N = n/60
errN = errn/60
Q = I/N
Qe = Q/e0

errQe = np.abs(-I/N**2 * errN) / e0

plt.errorbar(U, Qe, yerr=errQe, fmt='x', color='r', capsize=5, ecolor='b', barsabove=True, label='Werte mit Poisson-Fehler')

param2, cov2 = curve_fit(f, U, Qe)
x_plot = np.linspace(300, 700, 1000)
plt.plot(x_plot, f(x_plot, *param2), 'g-', label= 'Fit')

plt.xlabel(r'$U \:/\: \mathrm{V}$')
plt.ylabel(r'$\frac{\Delta Q}{e}$')
plt.legend()
# plt.show()
# plt.savefig('lad.pdf')

err2 = np.sqrt(np.diag(cov2))

print('a =', param2[0], '+-', err2[0])
print('b =', param2[1], '+-', err2[1])

# np.savetxt('werte.txt', np.column_stack([U, n, N, errN, I*10**6, Qe*10**(-9), errQe*10**(-9)]), fmt='%0.0f  %0.0f  %0.2f\pm%0.2f  %0.1f  %0.2f\pm%0.2f', header='U N N/t I \symup{\Delta Q}/e \n V - Bq \muA -')

from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(U,Qe)
print("r-squared:", r_value**2 * 100, '%')

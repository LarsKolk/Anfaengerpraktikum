import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const

null = 225/900

# Halbwertszeit von Indium

t, n = np.genfromtxt('ind.txt', unpack='true')

N = n/240 - null
errN = np.sqrt(n-240*null)/240

y = np.log(N)
erry1 = np.log(N+errN)-np.log(N)
erry2 = np.log(N)-np.log(N-errN)

def f(x,a,b):
    return -a*x+b

param, cov = curve_fit(f, t, y)
x_plot = np.linspace(0, 3700, 10000)
plt.plot(x_plot, f(x_plot, *param), 'g-', label= 'Fit')

plt.errorbar(t, y, yerr=[erry1, erry2], fmt='x', color='r', ecolor='b', capsize=5, barsabove=True, label='Werte mit Poisson-Fehler')
plt.xlabel(r'$t \:/\: s$')
plt.ylabel(r'$\ln{\left(N \right)}$')
plt.legend()
# plt.show()
# plt.savefig('ind.pdf')

plt.clf()

err = np.sqrt(np.diag(cov))

print('a =', param[0], '+-', err[0])
print('b =', param[1], '+-', err[1])

halb = np.log(2)/param[0]
errhalb = np.abs(-np.log(2)/param[0]**2 * err[0])

print('Halbwertszeit von Indium:', halb/60, '+-', errhalb/60, 'min')

bla = np.exp(param[1])
errbla = np.abs(bla * err[1])
print(bla, '+-', errbla)

halbth = 54.29*60

abw = np.abs(halb-halbth)/halbth * 100
print('Abweichung:', abw, '%')

# np.savetxt('werte1.txt', np.column_stack([t, n, N, errN, y, erry1, erry2]), fmt='%0.0f  %0.0f  %0.2f\pm%0.2f %0.2f %0.2f %0.2f', header='t N N/t \ln{N + \sigma}-\ln{N} \ln{N}-\ln{N - \sigma} \n s - Bq - - -')

# Rhodium

t, n = np.genfromtxt('rhod.txt', unpack='true')

N = n/18 - null
errN = np.sqrt(n-18*null)/18

y = np.log(N)
erry1 = np.log(N+errN)-np.log(N)
erry2 = np.log(N)-np.log(N-errN)

param2, cov2 = curve_fit(f, t[-18:], y[-18:])
x_plot = np.linspace(400, 800, 10000)
plt.plot(x_plot, f(x_plot, *param2), color='lime', linestyle='-', label= 'Fit')

plt.errorbar(t[-18:], y[-18:], yerr=[erry1[-18:], erry2[-18:]], fmt='x', color='r', ecolor='b', capsize=5, barsabove=True, label='Werte mit Poisson-Fehler')
plt.xlabel(r'$t \:/\: s$')
plt.ylabel(r'$\ln{\left(N \right)}$')
plt.legend()
# plt.show()
# plt.savefig('rhod1.pdf')

plt.clf()

err2 = np.sqrt(np.diag(cov2))

# np.savetxt('werte2.txt', np.column_stack([t, n, N, errN, y, erry1, erry2]), fmt='%0.0f  %0.0f  %0.2f\pm%0.2f %0.2f %0.2f %0.2f', header='t N N/t \ln{N + \sigma}-\ln{N} \ln{N}-\ln{N - \sigma} \n s - Bq - - -')


print('a2 =', param2[0], '+-', err2[0])
print('b2 =', param2[1], '+-', err2[1])

halb = np.log(2)/param2[0]
errhalb = np.abs(-np.log(2/param2[0]**2 * err2[0]))

print('Halbwertszeit von Rhodium_104i:', halb, '+-', errhalb, 's')

N12 = N-np.exp(-t*param2[0]+param2[1])
errN12 = np.sqrt((1*errN)**2+(t*np.exp(-param2[0]*t+param2[1])*err2[0])**2+(-np.exp(-param2[0]*t+param2[1])*err2[1])**2)

std1 = np.log(N12[:11]+errN12[:11])-np.log(N12[:11])
std2 = np.log(N12[:11])-np.log(N12[:11]-errN12[:11])

param3, cov3 = curve_fit(f, t[:11], np.log(N12[:11]))
x_plot = np.linspace(0, 210, 10000)
plt.plot(x_plot, f(x_plot, *param3), color='lime', linestyle='-', label= 'Fit')

plt.errorbar(t[:11], np.log(N12[:11]), yerr=[std1, std2], fmt='x', color='r', ecolor='b', capsize=5, barsabove=True, label='Werte mit Poisson-Fehler')
plt.xlabel(r'$t \:/\: s$')
plt.ylabel(r'$\ln{\left(N \right)}$')
plt.legend()
# plt.show()
# plt.savefig('rhod2.pdf')

plt.clf()

err3 = np.sqrt(np.diag(cov3))

print('a3 =', param3[0], '+-', err3[0])
print('b3 =', param3[1], '+-', err3[1])

halb = np.log(2)/param3[0]
errhalb = np.abs(-np.log(2/param3[0]**2 * err3[0]))

print('Halbwertszeit von Rhodium_104:', halb, '+-', errhalb, 's')

x_plot = np.linspace(15, 800, 10000)
plt.plot(t, y, 'rx', label='Werte')
plt.plot(x_plot, np.exp(-x_plot*param2[0]+param2[1])+np.exp(-x_plot*param3[0]+param2[1]), 'k-', label='Gesamt')
plt.legend()
# plt.show()

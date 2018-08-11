from linregress import linregress
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import scipy.constants as const
import math as ma

N_0 = 225/900
#2 Rhodium
t2,N2 = np.genfromtxt('rhod0.txt', unpack = True)
N2_err = np.sqrt(N2-N_0*18)/18
N2 = N2/18-N_0
N2_log = np.log(N2)
N2_err2 = N2-N2_err
N2_log_err = [np.log(N2+N2_err)-np.log(N2), np.log(N2)-np.log(N2_err2)]
N2 = unp.uarray(N2, N2_err)
t2 = t2*18

paramsLinear2, errorsLinear2, sigma_y = linregress(t2[-18:], N2_log[-18:])
steigung2 = unp.uarray(paramsLinear2[0], errorsLinear2[0])
achsenAbschnitt2 = unp.uarray(paramsLinear2[1], errorsLinear2[1])

paramsLinear1, errorsLinear1, sigma_y = linregress(t2[:11], np.log(noms(N2)[:11]-np.exp(t2[:11]*paramsLinear2[0]+paramsLinear2[1])))
steigung1 = unp.uarray(paramsLinear1[0], errorsLinear1[0])
achsenAbschnitt1 = unp.uarray(paramsLinear1[1], errorsLinear1[1])

x_plot1 = np.linspace(400,800)
N21 = N2[:11]-np.exp(t2[:11]*paramsLinear2[0]+paramsLinear2[1])
N21_log_err = [np.log(noms(N21)+stds(N21))-np.log(noms(N21)), np.log(noms(N21))-np.log(noms(N21)-stds(N21))]

print('Lambda21=', -steigung1)
print('N021=', unp.exp(achsenAbschnitt1))
print('tau21=', -np.log(2)/steigung1)
print('Lambda22=', -steigung2)
print('N022=', unp.exp(achsenAbschnitt2))
print('tau22=', -np.log(2)/steigung2)

print('t2[-18]=', t2[-18])
print('t2[11]=', t2[11])

plt.cla()
plt.clf()
plt.errorbar(t2[-18:], N2_log[-18:], yerr=[N2_log_err[0][-18:],N2_log_err[1][-18:]], fmt='rx', markersize=5, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='b',barsabove=True ,label='Messwerte mit Poisson-Fehler')
plt.plot(x_plot1, x_plot1*paramsLinear2[0]+paramsLinear2[1], 'orange', linewidth=0.8, label='Fit2')
plt.xlabel(r'$t \:/\: \mathrm{s}$')
plt.ylabel(r'$\ln{\left(N_\mathrm{Rh_{104i}} \right)}$')
# plt.xlim(25*10,44*10+10)
plt.legend(loc="best")
# plt.savefig('rhodi.pdf')
# plt.show()

plt.cla()
plt.clf()
x_plot2 = np.linspace(0,240)

plt.errorbar(t2[:11], np.log(noms(N2)[:11]-np.exp(t2[:11]*paramsLinear2[0]+paramsLinear2[1])), yerr=[N21_log_err[0][:11],N21_log_err[1][:11]], fmt='rx', markersize=5, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='b',barsabove=True ,label='Messwerte mit Poisson-Fehler')
plt.plot(x_plot2, x_plot2*paramsLinear1[0]+paramsLinear1[1], 'g', linewidth=0.8, label='Fit1')
plt.xlabel(r'$t \:/\: \mathrm{s}$')
plt.ylabel(r'$\ln{\left(N_\mathrm{Rh_{104}} \right)}$')
# plt.xlim(0,20*10+10)
plt.legend(loc="best")
# plt.savefig('rhod.pdf')
# plt.show()

plt.cla()
plt.clf()
x_plot3 = np.linspace(0,800)
N_t = np.exp(x_plot3*paramsLinear1[0]+paramsLinear1[1])+np.exp(x_plot3*paramsLinear2[0]+paramsLinear2[1])

plt.errorbar(t2, N2_log, yerr=[N2_log_err[0],N2_log_err[1]], fmt='rx', markersize=5, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='b',barsabove=True ,label='Messwerte mit Poisson-Fehler')
plt.plot(x_plot2, x_plot2*paramsLinear1[0]+paramsLinear1[1], 'g', linewidth=0.8, label='Fit 1')
plt.plot(x_plot3, x_plot3*paramsLinear2[0]+paramsLinear2[1], 'orange', linewidth=0.8, label='Fit 2')
plt.plot(x_plot3, np.log(N_t), 'k-', linewidth=0.8, label='N(t)')
plt.xlabel(r'$t \:/\: \mathrm{s}$')
plt.ylabel(r'$\ln{\left(N \right)}$')
# plt.xlim(0,44*10+10)
plt.legend(loc="best")
# plt.savefig('rhodges.pdf')
# plt.show()

plt.cla()
plt.clf()
x_plot = np.linspace(0, 800)
plt.errorbar(t2, noms(N2), yerr=stds(N2), fmt='rx', markersize=5, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='b',barsabove=True ,label='Messwerte mit Poisson-Fehler')
# plt.errorbar(t2[44:], noms(N2)[44:], yerr=stds(N2)[44:], color='grey', fmt='x', markersize=5, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='lightgrey',barsabove=True ,label='ungenutzte Messwerte')
plt.plot(x_plot, N_t, 'k-', linewidth=0.8, label='N(t)')
plt.xlabel(r'$t \:/\: \mathrm{s}$')
plt.ylabel(r'$\ln{\left(N \right)}$')
# plt.xlim(0,60*10+10)
plt.legend(loc="best")
# plt.savefig('rhodgesnolog.pdf')
# plt.show()

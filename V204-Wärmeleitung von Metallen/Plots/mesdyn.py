import numpy as np
import matplotlib.pyplot as plt

An, Af, deltat = np.genfromtxt('mes.txt', unpack='true')
t, T1, T2, T3, T4, T5, T6, T7, T8 = np.genfromtxt('GLXportRun2.txt', unpack='true')
t *= 2

plt.plot(t, T1, 'b:', label='Temperaturverlauf T1')
plt.plot(t, T2, 'r:', label='Temperaturverlauf T2')

plt.xlabel(r'$t \,/\, \mathrm{s}$')
plt.ylabel(r'$T \,/\, \mathrm{°C}$')
# plt.xlim(0, 60)
# plt.ylim(0, 0.07)

plt.legend()
plt.grid()
# plt.show()
# plt.savefig('mesdyn.pdf')

Anahx = 0
Afernx = 0
dtx = 0
errAnahx = 0
errAfernx = 0
errdtx = 0
for i in range(10):
    Anahx += An[i]
    Afernx += Af[i]
    dtx += deltat[i]

Anah = Anahx/10
Afern = Afernx/10
dt = dtx/10

for i in range(10):
    errAnahx += (An[i] - Anah)**2
    errAfernx += (Af[i] - Afern)**2
    errdtx += (deltat[i] - dt)**2

errAnah = np.sqrt(1/(10*9) * errAnahx)
errAfern = np.sqrt(1/(10*9) * errAfernx)
errdt = np.sqrt(1/(10*9) * errdtx)

print('Anah =', Anah, '+-', errAnah, 'K')
print('Afern =', Afern, '+-', errAfern, 'K')
print('dt =', dt, '+-', errdt, 's')
# print(np.log(Anah/Afern))

rho = 8520
c = 385
dx = 3e-2
XX = (rho * c * dx**2)/2

k = XX/(dt * np.log(Anah/Afern))
errk = np.sqrt((-XX/(dt**2 * np.log(Anah/Afern)) * errdt)**2 + (-XX/(dt * Anah * (np.log(Anah/Afern))**2) * errAnah)**2 + (XX/(dt * Afern * (np.log(Anah/Afern))**2) * errAfern)**2)

print('k =', k, '+-', errk)

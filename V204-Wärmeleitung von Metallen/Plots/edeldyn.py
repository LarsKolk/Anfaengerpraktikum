import numpy as np
import matplotlib.pyplot as plt

An, Af, deltat = np.genfromtxt('edel.txt', unpack='true')
t, T1, T2, T3, T4, T5, T6, T7, T8 = np.genfromtxt('GLXportRun3.txt', unpack='true')
t *= 2

plt.plot(t, T7, 'b:', label='Temperaturverlauf T7')
plt.plot(t, T8, 'r:', label='Temperaturverlauf T8')

plt.xlabel(r'$t \,/\, \mathrm{s}$')
plt.ylabel(r'$T \,/\, \mathrm{°C}$')
# plt.xlim(0, 60)
# plt.ylim(20, 46)

plt.legend()
plt.grid()
# plt.show()
# plt.savefig('edeldyn.pdf')

Anahx = 0
Afernx = 0
dtx = 0
errAnahx = 0
errAfernx = 0
errdtx = 0
for i in range(2):
    Anahx += An[i]
    Afernx += Af[i]
    dtx += deltat[i]

Anah = Anahx/2
Afern = Afernx/2
dt = dtx/2

for i in range(2):
    errAnahx += (An[i] - Anah)**2
    errAfernx += (Af[i] - Afern)**2
    errdtx += (deltat[i] - dt)**2

errAnah = np.sqrt(1/(2*1) * errAnahx)
errAfern = np.sqrt(1/(2*1) * errAfernx)
errdt = np.sqrt(1/(2*1) * errdtx)

print('Anah =', Anah, '+-', errAnah, 'K')
print('Afern =', Afern, '+-', errAfern, 'K')
print('dt =', dt, '+-', errdt, 's')
# print(np.log(Anah/Afern))

rho = 8000
c = 400
dx = 3e-2
XX = (rho * c * dx**2)/2

k = XX/(dt * np.log(Anah/Afern))
errk = np.sqrt((-XX/(dt**2 * np.log(Anah/Afern)) * errdt)**2 + (-XX/(dt * Anah * (np.log(Anah/Afern))**2) * errAnah)**2 + (XX/(dt * Afern * (np.log(Anah/Afern))**2) * errAfern)**2)

print('k =', k, '+-', errk)

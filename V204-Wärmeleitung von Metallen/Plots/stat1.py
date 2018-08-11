import numpy as np
import matplotlib.pyplot as plt

t, T1, T2, T3, T4, T5, T6, T7, T8 = np.genfromtxt('GLXportRun1.txt', unpack='true')
t *= 5

plt.plot(t, T1, 'b:', label='Temperaturverlauf T1')
plt.plot(t, T4, 'r:', label='Temperaturverlauf T4')

plt.xlabel(r'$t \,/\, \mathrm{s}$')
plt.ylabel(r'$T \,/\, \mathrm{°C}$')
# plt.xlim(0, 60)
# plt.ylim(0, 0.07)

plt.legend()
plt.grid()
# plt.show()
# plt.savefig('stat1.pdf')

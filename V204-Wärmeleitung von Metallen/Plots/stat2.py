import numpy as np
import matplotlib.pyplot as plt

t, T1, T2, T3, T4, T5, T6, T7, T8 = np.genfromtxt('GLXportRun1.txt', unpack='true')
t *= 5

plt.plot(t, T5, 'b:', label='Temperaturverlauf T5')
plt.plot(t, T8, 'r:', label='Temperaturverlauf T8')

plt.xlabel(r'$t \,/\, \mathrm{s}$')
plt.ylabel(r'$T \,/\, \mathrm{°C}$')
# plt.xlim(0, 60)
# plt.ylim(0, 0.07)

plt.legend()
plt.grid()
# plt.show()
# plt.savefig('stat2.pdf')

print('Werte nach 700s Messzeit')
print(t[139], T1[139], T4[139], T5[139], T8[139])

import numpy as np
import matplotlib.pyplot as plt

t, T1, T2, T3, T4, T5, T6, T7, T8 = np.genfromtxt('GLXportRun1.txt', unpack='true')
t *= 5

plt.plot(t, T7-T8, 'b:', label='T7-T8')
plt.plot(t, T2-T1, 'r:', label='T2-T1')

plt.xlabel(r'$t \,/\, \mathrm{s}$')
plt.ylabel(r'$\Delta T \,/\, \mathrm{°C}$')
# plt.xlim(0, 60)
# plt.ylim(0, 0.07)

plt.legend()
plt.grid()
# plt.show()
# plt.savefig('diff.pdf')

A = 0.4e-2 * 1.2e-2
x = 3e-2

km = 109
ks = 16

# Q = - ks * A * (T7[39] - T8[39])/x

s = [19, 59, 99, 139, 179]

print('T =', 100, 300, 500, 700, 900, 's')
print('Werte fuer Stahl')
for xs in s:
    Q1 = - ks * A * (T7[xs] - T8[xs])/x
    print(Q1, 'W')

print('Werte fuer Messing')
for xs in s:
    Q2 = -km * A * (T2[xs] - T1[xs])/x
    print(Q2, 'W')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const

D, I = np.genfromtxt('mag400.txt', unpack='true')
D *= 1e-2
mu = const.value('mag. constant')
N = 20
L = 17.5 * 10**(-2)
r = 0.282
B=[]
S=[]

i = 0
while i<9:
    s = D[i]/(D[i]**2+L**2)
    b = mu*8/np.sqrt(125)* N*I[i]/r * 1000
    B.append(b)
    S.append(s)
    i += 1

print(B) #Magnetfeld ist in mT
print(S)

def f(x, a, b):
    return a*x+b

param, cov = curve_fit(f, B, S)
x_plot = np.linspace(-0.02, 0.22, 1000)
plt.plot(x_plot, f(x_plot, *param), 'b-', label= 'Fit')
plt.plot(B, S, 'rx', label='Messwerte')
plt.xlabel(r'$B \:/\: \mathrm{mT}$')
plt.ylabel(r'$\frac{D}{D^2+L^2} \:/\: \frac{1}{\mathrm{m}}$')
plt.legend()
# plt.show()
# plt.savefig('magnet.pdf')

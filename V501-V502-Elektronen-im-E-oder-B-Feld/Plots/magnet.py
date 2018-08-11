import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const

D1, I1 = np.genfromtxt('mag400.txt', unpack='true')
D1 *= 1e-2
mu = const.value('mag. constant')
e0 = const.value('electron volt')
m0 = const.value('electron mass')
N = 20
L = 17.5 * 10**(-2)
r = 0.282
B1=[]
S1=[]

i = 0
while i<9:
    s = D1[i]/(D1[i]**2+L**2)
    b = mu*8/np.sqrt(125)* N*I1[i]/r * 1000
    B1.append(b)
    S1.append(s)
    i += 1

# print(B1) #Magnetfeld ist in mT
# print(S1)

def f(x, a, b):
    return a*x+b

param, cov = curve_fit(f, B1, S1)
x_plot = np.linspace(-0.02, 0.22, 1000)
plt.plot(x_plot, f(x_plot, *param), 'b-', label= r'Fit $U_\mathrm{B} = 400 \mathrm{V}$')
plt.plot(B1, S1, 'rx', label=r'Messwerte $U_\mathrm{B} = 400 \mathrm{V}$')

D2, I2 = np.genfromtxt('mag250.txt', unpack='true')
D2 *= 1e-2

B2=[]
S2=[]

i = 0
while i<9:
    s = D2[i]/(D2[i]**2+L**2)
    b = mu*8/np.sqrt(125)* N*I2[i]/r * 1000
    B2.append(b)
    S2.append(s)
    i += 1

# print(B2) #Magnetfeld ist in mT
# print(S2)

param2, cov2 = curve_fit(f, B2, S2)
x_plot2 = np.linspace(-0.02, 0.22, 1000)
plt.plot(x_plot2, f(x_plot2, *param2), 'k-', label= r'Fit $U_\mathrm{B} = 250 \mathrm{V}$')
plt.plot(B2, S2, 'gx', label=r'Messwerte $U_\mathrm{B} = 250 \mathrm{V}$')

plt.xlabel(r'$B \:/\: \mathrm{mT}$')
plt.ylabel(r'$\frac{D}{D^2+L^2} \:/\: \frac{1}{\mathrm{m}}$')
plt.legend()
# plt.show()
# plt.savefig('magnet.pdf')

err = np.sqrt(np.diag(cov))
err2 = np.sqrt(np.diag(cov2))

print('a_250 =', param2[0], '+-', err2[0])
print('a_400 =', param[0], '+-', err[0])

## Spezifische Ladung

spez = 8*(param[0]*10**3)**2*400
spez2 = 8*(param2[0]*10**3)**2*250

errspez = np.abs(8*2*param[0]*10**3*400)
errspez2 = np.abs(8*2*param2[0]*10**3*250)

print('(e/m)_250 =', spez2*10**(-11), '+-', errspez*10**(-11), '* 10^(11)')
print('(e/m)_400 =', spez*10**(-11), '+-', errspez2*10**(-11), '* 10^(11)')

mit = (spez+spez2)/2
errmit = np.sqrt(1/2 * ((spez-mit)**2+(spez2-mit)**2))

print('(e/m) =', mit*10**(-11), '+-', errmit*10**(-11), '* 10^(11)')

theo = e0/m0

print('theo =', theo*10**(-11), '* 10^(11)')
abw = np.abs(theo-mit)/theo * 100

print('Abweichung:', abw, '%')

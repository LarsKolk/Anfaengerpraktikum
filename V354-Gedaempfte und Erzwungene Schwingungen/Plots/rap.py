import numpy as np

L = 3.53e-3
errL = 0.03e-3
C = 5.015e-9
errC = 0.015e-9

print('Experimentelle Werte')
Rap = 1.29*10**3
print('R_ap =', Rap)

print('Theoriewerte')
Rap2 = 2 * np.sqrt(L/C)
errRap2 = np.sqrt((1/np.sqrt(L*C)*errL)**2+(-1/np.sqrt(C**3*L)*errC)**2)

print('R_ap =', Rap2, '+-', errRap2)

print('Abweichung')
xx = np.abs((Rap2-Rap)/Rap2)
print('... von R_ap =', xx)

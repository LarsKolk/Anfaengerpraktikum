import numpy as np

Mr = 0.3792
Me = 0.1671
Lr = 0.575
Le = 0.59
r = 0.005
h = 0.01


def d1(r, m, L):
    return m / (np.pi * r**2 * L)


def d2(h, m, L):
    return m / (h**2 * L)


dr = d1(r, Mr, Lr)
de = d2(h, Me, Le)


print('Dichte rund:', dr)
print('Dichte eckig:', de)

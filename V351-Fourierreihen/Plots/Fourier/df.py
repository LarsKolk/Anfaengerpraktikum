import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2*np.pi, 2*np.pi)
y = [0 for _ in x]


def f(i):
    global x
    global y
    n = 1
    y = np.pi/2
    for n in range(1, i+1):
        y += - 4/np.pi * np.cos((2*n-1)*x)/(2*n-1)**2


f(1)
plt.plot(x, y, 'y-', label='Fourier, n = 1')

f(2)
plt.plot(x, y, 'r-', label='Fourier, n = 2')

f(5)
plt.plot(x, y, 'g-', label='Fourier, n = 5')

f(20)
plt.plot(x, y, 'b-', label='Fourier, n = 20')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.xlim(-2*np.pi, 2*np.pi)

plt.legend()
plt.grid()
# plt.show()
# plt.savefig('df.pdf')

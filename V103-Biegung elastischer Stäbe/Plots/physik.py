import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit

x, y = np.genfromtxt('physik.txt', unpack=True)
n=0
i=0
u=0
while (i<len(y)):
    if(y[i]<32):
        n+=1
        i+=1
    else:
        i+=1
        u+=1
print("Hilla&Kahn")
print(" Durchgefallen : ",  n, "von", len(y))
print(" Bestanden : ", u, "von", len(y))
print(" Beste Punktzahl : ",max(y))
print(" Schlechteste Punktzahl : ",-max(-y), "\n")


a, b, c = np.genfromtxt('PhysikKP.txt', unpack=True)
n=0
i=0
u=0
while (i<len(c)):
    if(c[i]<5):
        n+=1
        i+=1
    else:
        i+=1
        u+=1



print("Kroeniger&Paes")
print(" Durchgefallen : ",  n, "von", len(c))
print(" Bestanden : ", u, "von", len(c))
print(" Beste Punktzahl : - ")
print(" Schlechteste Punktzahl : - ")
#Vergleich der Matrikelnummern
i=0
j=0
n=0
d=[]
e=[]
while(i<len(x)):
    while(n<len(b)):
        if(b[n] == x[i]):
            d.extend([None])
            e.extend([None])
            d[j]=b[n]
            e[j]=y[i]
            n+=1
            j+=1
        else:
            n+=1
    n=0
    i+=1
print("\n \n Matrikelnummern, die doppelt vorkommen : ", d,"\n Insgesamt : ", len(d))

#Code, um rauszufinden, wie viele der Wiederholer nicht bestanden haben
n=0
i=0
while(n<len(e)):
    if e[i]<32:
        i+=1
    n+=1
print("\n Davon durchgefallen :", i)
print("Punkte : ", e)
#Notenliste der Mediphysiker importiert
r, s, t = np.genfromtxt('mp.txt', unpack=True)
#Vergleich der Matrikelnummern
i=0
j=0
n=0
d=[]
e=[]
while(i<len(x)):
    while(n<len(r)):
        if(r[n] == x[i]):
            d.extend([None])
            e.extend([None])
            d[j]=r[n]
            e[j]=y[i]
            n+=1
            j+=1
        else:
            n+=1
    n=0
    i+=1
print("\n \n Matrikelnummern der Mediphysiker (Hillas Klausur) : ", d,"\n Insgesamt : ", len(d))

#Code, um rauszufinden, wie viele der Mediphysiker nicht bestanden haben
n=0
i=0
while(n<len(e)):
    if e[n]<32:
        i+=1
    n+=1
print("\n Davon durchgefallen :", i)
print("Punkte : ", e)

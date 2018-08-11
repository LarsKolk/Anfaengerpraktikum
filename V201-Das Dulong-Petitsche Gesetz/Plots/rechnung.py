import numpy as np

mk = 856.84
mx = 300
my = 300
Tx = 20.7
Ty = 82.6
Tm = 50.9
cw = 4.18
mw = 600


print('Kapazitaet Kalorimeter')
cm = (cw * my * (Ty-Tm) - cw * mx * (Tm-Tx))/(Tm-Tx)
print('c_g * m_g =', cm, 'J/K')

print('Kapazitaet Graphit')
mg = 97.75

Twg1 = 22.3 + 273.15
Twg2 = 22.3 + 273.15
Twg3 = 25.6 + 273.15

Tg1 = 82.8 + 273.15
Tg2 = 92.3 + 273.15
Tg3 = 87.5 + 273.15

Tmg1 = 22.6 + 273.15
Tmg2 = 24.5 + 273.15
Tmg3 = 26.4 + 273.15

cg1 = ((cw * mw + cm) * (Tmg1-Twg1))/(mg * (Tg1-Tmg1))
print('cg1 =', cg1, 'J/(gK)')
cg2 = ((cw * mw + cm) * (Tmg2-Twg2))/(mg * (Tg2-Tmg2))
print('cg2 =', cg2, 'J/(gK)')
cg3 = ((cw * mw + cm) * (Tmg3-Twg3))/(mg * (Tg3-Tmg3))
print('cg3 =', cg3, 'J/(gK)')

cg = (cg1+cg2+cg3)/3

a = [cg1, cg2, cg3]

i = 0
xx = 0
while i < 3:
    xx += (a[i]-cg)**2
    i += 1

errcg = np.sqrt(1/(3*2) * xx)

print('cg =', cg, '+-', errcg, 'J/(gK)')

cgtheo = 0.715
cgabw = np.abs((cgtheo-cg)/cgtheo)*100

print('cg_theo =', cgtheo, 'J/(gK)')
print('Abweichung =', cgabw, '%')

print('Kapazitaet Blei')
mb = 535.33

Twb1 = 26.0 + 273.15
Twb2 = 26.6 + 273.15
Twb3 = 30.3 + 273.15

Tb1 = 84.4 + 273.15
Tb2 = 93 + 273.15
Tb3 = 94.2 + 273.15

Tmb1 = 28.3 + 273.15
Tmb2 = 29.4 + 273.15
Tmb3 = 30.8 + 273.15

cb1 = ((cw * mw + cm) * (Tmb1-Twb1))/(mb * (Tb1-Tmb1))
print('cb1 =', cb1, 'J/(gK)')
cb2 = ((cw * mw + cm) * (Tmb2-Twb2))/(mb * (Tb2-Tmb2))
print('cb2 =', cb2, 'J/(gK)')
cb3 = ((cw * mw + cm) * (Tmb3-Twb3))/(mb * (Tb3-Tmb3))
print('cb3 =', cb3, 'J/(gK)')

cb = (cb1+cb2+cb3)/3

b = [cb1, cb2, cb3]

i = 0
yy = 0
while i < 3:
    yy += (b[i]-cb)**2
    i += 1

errcb = np.sqrt(1/(3*2) * yy)

cbtheo = 0.129
cbabw = np.abs((cbtheo-cb)/cbtheo)*100

print('cb =', cb, '+-', errcb, 'J/(gK)')

print('cb_theo =', cbtheo, 'J/(gK)')
print('Abweichung =', cbabw, '%')

print('Kapazitaet Aluminium')
ma = 106.58

Twa1 = 30.4 + 273.15
Twa2 = 32.6 + 273.15
Twa3 = 34.0 + 273.15

Ta1 = 90.5 + 273.15
Ta2 = 87.0 + 273.15
Ta3 = 83.0 + 273.15

Tma1 = 31.5 + 273.15
Tma2 = 34.2 + 273.15
Tma3 = 35.2 + 273.15

ca1 = ((cw * mw + cm) * (Tma1-Twa1))/(ma * (Ta1-Tma1))
print('ca1 =', ca1, 'J/(gK)')
ca2 = ((cw * mw + cm) * (Tma2-Twa2))/(ma * (Ta2-Tma2))
print('ca2 =', ca2, 'J/(gK)')
ca3 = ((cw * mw + cm) * (Tma3-Twa3))/(ma * (Ta3-Tma3))
print('ca3 =', ca3, 'J/(gK)')

ca = (ca1+ca2+ca3)/3

c = [ca1, ca2, ca3]

i = 0
zz = 0
while i < 3:
    zz += (c[i]-ca)**2
    i += 1

errca = np.sqrt(1/(3*2) * zz)

print('ca =', ca, '+-', errca, 'J/(gK)')

catheo = 0.895
caabw = np.abs((catheo-ca)/catheo)*100

print('ca_theo =', catheo, 'J/(gK)')
print('Abweichung =', caabw, '%')

print('Molwaermen')
Ctheo = 3*8.314
print('C_theo =', Ctheo, 'J/(mol K)')
graphit = [2.25e6, 12, 8e-6, 33e9]
blei = [11.35e6, 207.2, 29e-6, 42e9]
alu = [2.7e6, 27, 23.5e-6, 75e9]

# GRAPHIT
Tmg = (Tmg1+Tmg2+Tmg3)/3

a = [Tmg1, Tmg2, Tmg3]

i = 0
xx = 0
while i < 3:
    xx += (a[i]-Tmg)**2
    i += 1

errTmg = np.sqrt(1/(3*2) * xx)
print('T_mg =', Tmg, '+-', errTmg)

Cvg = cg * graphit[1] - 9 * graphit[2]**2 * graphit[3] * graphit[1]/graphit[0] * Tmg

errCvg = np.sqrt((graphit[1] * errcg)**2 + (-9*graphit[2]**2 * graphit[1]/graphit[0] * errTmg)**2)
print('Cv_g =', Cvg, '+-', errCvg)

Cvgabw = np.abs((Cvg-Ctheo)/Ctheo) * 100
print('Abweichung =', Cvgabw, '%')

# Cvg1 = cg1 * graphit[1] - 9 * graphit[2]**2 * graphit[3] * graphit[1]/graphit[0] * Tmg1
# print('Cv_g1 =', Cvg1, 'J/(mol K)')
# Cvg2 = cg2 * graphit[1] - 9 * graphit[2]**2 * graphit[3] * graphit[1]/graphit[0] * Tmg2
# print('Cv_g2 =', Cvg2, 'J/(mol K)')
# Cvg3 = cg3 * graphit[1] - 9 * graphit[2]**2 * graphit[3] * graphit[1]/graphit[0] * Tmg3
# print('Cv_g3 =', Cvg3, 'J/(mol K)')
#
# Cvg = (Cvg1+Cvg2+Cvg3)/3
#
# blub = [Cvg1, Cvg2, Cvg3]
#
# i = 0
# zz = 0
# while i < 3:
#     zz += (blub[i]-Cvg)**2
#     i += 1
#
# errCvg = np.sqrt(1/6 * zz)
#
# print('Cv_g =', Cvg, '+-', errCvg)
#
# Cvgabw = np.abs((Cvg-Ctheo)/Ctheo) * 100
# print('Abweichung =', Cvgabw, '%')

# BLEI

Tmb = (Tmb1+Tmb2+Tmb3)/3

a = [Tmb1, Tmb2, Tmb3]

i = 0
xx = 0
while i < 3:
    xx += (a[i]-Tmb)**2
    i += 1

errTmb = np.sqrt(1/(3*2) * xx)
print('T_mb =', Tmb, '+-', errTmb)

Cvb = cb * blei[1] - 9 * blei[2]**2 * blei[3] * blei[1]/blei[0] * Tmb

errCvb = np.sqrt((blei[1] * errcb)**2 + (-9*blei[2]**2 * blei[1]/blei[0] * errTmb)**2)
print('Cv_b =', Cvb, '+-', errCvb)

# Cvgabw = np.abs((Cvg-Ctheo)/Ctheo) * 100
# print('Abweichung =', Cvgabw, '%')
# Cvb1 = cb1 * blei[1] - 9 * blei[2]**2 * blei[3] * blei[1]/blei[0] * Tmb1
# print('Cv_b1 =', Cvb1, 'J/(mol K)')
# Cvb2 = cb2 * blei[1] - 9 * blei[2]**2 * blei[3] * blei[1]/blei[0] * Tmb2
# print('Cv_b2 =', Cvb2, 'J/(mol K)')
# Cvb3 = cb3 * blei[1] - 9 * blei[2]**2 * blei[3] * blei[1]/blei[0] * Tmb3
# print('Cv_b3 =', Cvb3, 'J/(mol K)')
#
# Cvb = (Cvb1+Cvb2+Cvb3)/3
#
# blub = [Cvb1, Cvb2, Cvb3]
#
# i = 0
# zz = 0
# while i < 3:
#     zz += (blub[i]-Cvb)**2
#     i += 1
#
# errCvb = np.sqrt(1/6 * zz)
#
# print('Cv_b =', Cvb, '+-', errCvb)
#
# Cvbabw = np.abs((Cvb-Ctheo)/Ctheo) * 100
# print('Abweichung =', Cvbabw, '%')

# ALUMINIUM

Tma = (Tma1+Tma2+Tma3)/3

a = [Tma1, Tma2, Tma3]

i = 0
xx = 0
while i < 3:
    xx += (a[i]-Tma)**2
    i += 1

errTma = np.sqrt(1/(3*2) * xx)
print('T_ma =', Tma, '+-', errTma)

Cva = ca * alu[1] - 9 * alu[2]**2 * alu[3] * alu[1]/alu[0] * Tma

errCva = np.sqrt((alu[1] * errca)**2 + (-9*alu[2]**2 *alu[1]/alu[0] * errTma)**2)
print('Cv_a =', Cva, '+-', errCva)

Cvaabw = np.abs((Cva-Ctheo)/Ctheo) * 100
print('Abweichung =', Cvaabw, '%')

# Cva1 = ca1 * alu[1] - 9 * alu[2]**2 * alu[3] * alu[1]/alu[0] * Tma1
# print('Cv_a1 =', Cva1, 'J/(mol K)')
# Cva2 = ca2 * alu[1] - 9 * alu[2]**2 * alu[3] * alu[1]/alu[0] * Tma2
# print('Cv_a2 =', Cva2, 'J/(mol K)')
# Cva3 = ca3 * alu[1] - 9 * alu[2]**2 * alu[3] * alu[1]/alu[0] * Tma3
# print('Cv_a3 =', Cva3, 'J/(mol K)')
#
# Cva = (Cva1+Cva2+Cva3)/3
#
# blub = [Cva1, Cva2, Cva3]
#
# i = 0
# zz = 0
# while i < 3:
#     zz += (blub[i]-Cva)**2
#     i += 1
#
# errCva = np.sqrt(1/6 * zz)
#
# print('Cv_a =', Cva, '+-', errCva)
#
# Cvaabw = np.abs((Cva-Ctheo)/Ctheo) * 100
# print('Abweichung =', Cvaabw, '%')

import numpy as np
from scipy.integrate import simps, cumtrapz
from scipy.interpolate import interp1d

## FIDUCIAL COSMOLOGY
h = 0.705
om = 0.273
ob = 0.0456
ol = 0.727
'''
h = 0.71
om = 0.27
ob = 0.0448 #0.0226
ol = 1 - om
#'''

## r_s equation
og = 2.469e-5 / (h * h)
orr = og * (1 + (7.0/8.0)*np.power(4.0/11.0, 4.0/3.0)*3.04)
c = 3e5
z = 1020
ae = 1.0 / (1.0 + z)

a = np.linspace(ae/100000, ae, 1000)

ez = np.sqrt(om / (a*a*a) + ol + orr/(a*a*a*a))
hs = h * ez

fac = 1/np.sqrt(3 * (1 + (3 * ob / (4 * og))*a))
integrand = (c/100) * fac / (a*a*hs)
integral = simps(integrand, x=a)
rs = integral




#### GET FIDUCIAL DA AND H
H = 100

zs = np.linspace(0,1,10000)
ez = np.sqrt(om * (1 + zs)**3 + ol)

iez = 1 / ez

chi = cumtrapz(iez, zs, initial=0)
da = (c/(H * h)) * chi / (1 + zs)
hs = h * H * ez

davals = interp1d(zs, da)([0.44,0.6,0.73])
hvals = interp1d(zs, hs)([0.44,0.6,0.73])
print("z_d=",z," r_s=", rs)

print("D_A=",davals)
print("H=",davals)
print("D_A/r_s=",davals/rs)
print("cz/H/r_s=", 3e5*np.array([0.44,0.6,0.73])/hvals/rs)

print("My D_A/r_s=", np.array([1300, 1300, 1350]/rs))
print("My cz/H/r_s=", 3e5*np.array([0.44,0.6,0.73])/np.array([87, 90, 82])/rs)
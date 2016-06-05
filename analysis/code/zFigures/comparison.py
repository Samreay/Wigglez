import numpy as np
import matplotlib, matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

zs = np.linspace(0,3,10000)
logzs = np.log(zs)

h = 0.6774
om = 0.3089
ol = 0.6911
ok = 1 - om - ol
c = 3e5
H = 100

ez = np.sqrt(om * (1 + zs)**3 + ok * (1 + zs)**2 + ol)

iez = 1 / ez

chi = cumtrapz(iez, zs, initial=0)
da = (c/(H * h)) * chi / (1 + zs)
hs = h * H * ez



das = [1205, 1380, 1534, 1048, 1421, 1590]
dase = [114, 95, 107, 59, 20, 60]
dasz = [0.44, 0.6, 0.73, 0.35, 0.57, 2.36]


hss = [79.69, 83.80, 86.45, 82.6, 87.9, 97.3, 82.1, 96.8, 228]
hse = [2.32, 2.96, 3.27, 7.8, 6.1, 7.0, 4.9, 3.4, 8]
hzs = [0.24, 0.34, 0.43, 0.44, 0.06, 0.73, 0.35, 0.57, 2.36]


myz = [0.44, 0.6, 0.73]
myda = [1300, 1300, 1350]
mydae = [160, 180, 160]
myh = [87, 90, 82]
myhe = [16,15,13]

fig, axes = plt.subplots(ncols=2, figsize=(10,4))

axes[0].plot(logzs, da, color='r', label="Planck cosmology")
axes[0].errorbar(np.log(dasz), das, yerr=dase, fmt='o', label="External constraints")
axes[0].errorbar(np.log(myz), myda, yerr=mydae, fmt='s', label="Our constraints")

axes[1].plot(logzs, hs, 'r')
axes[1].errorbar(np.log(hzs), hss, yerr=hse, fmt='o')
axes[1].errorbar(np.log(myz), myh, yerr=myhe, fmt='s')


axes[0].set_xlim(-1.5,1)
axes[0].set_ylim(400, 1900)
axes[0].set_xlabel("$\log z$", fontsize=16)
axes[0].set_ylabel(r"$D_A(z) \ [\rm{Mpc}]$", fontsize=16)


axes[1].set_xlim(-1.5,1)
axes[1].set_ylim(65, 250)

axes[1].set_xlabel("$\log z$", fontsize=16)
axes[1].set_ylabel(r"$H(z) \ [\rm{km/s/Mpc}]$", fontsize=16)

axes[0].legend(frameon=False, loc=4)
fig.savefig("external.pdf", bbox_inches="tight", transparent=True)
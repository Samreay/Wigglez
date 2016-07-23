# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

wizcola_z0_file = "../data/wizcola/multipoles/WizCOLA_xi0xi2_combined_z0pt2_0pt6.dat"
wizcola_z1_file = "../data/wizcola/multipoles/WizCOLA_xi0xi2_combined_z0pt4_0pt8.dat"
wizcola_z2_file = "../data/wizcola/multipoles/WizCOLA_xi0xi2_combined_z0pt6_1pt0.dat"


precon_z0_file = "../data/wiggleznew/multipoles/WiggleZ_xi0xi2_all6regions_z0pt2_0pt6.dat"
precon_z1_file = "../data/wiggleznew/multipoles/WiggleZ_xi0xi2_all6regions_z0pt4_0pt8.dat"
precon_z2_file = "../data/wiggleznew/multipoles/WiggleZ_xi0xi2_all6regions_z0pt6_1pt0.dat"

recon_z0_file = "../data/wigglezrecon/WiggleZ_xiLxiT_z26.ascii"
recon_z1_file = "../data/wigglezrecon/WiggleZ_xiLxiT_z48.ascii"
recon_z2_file = "../data/wigglezrecon/WiggleZ_xiLxiT_z60.ascii"

valsw_m = []
valsw_me = []
valsw_q = []
valsw_qe = []
ssw = []
for file in [wizcola_z0_file, wizcola_z1_file, wizcola_z2_file]:
    data = np.loadtxt(file)
    ssw.append(data[:30,2])
    valsw_m.append(data[:30,3:].mean(axis=1))
    valsw_me.append(np.std(data[:30,3:], axis=1))
    valsw_q.append(data[30:,3:].mean(axis=1))
    valsw_qe.append(np.std(data[30:,3:],axis=1))
#for file in [recon_matrx_z0_file, recon_matrx_z1_file, recon_matrx_z2_file]:

ss = []
vals_m = []
vals_me = []
vals_q = []
vals_qe = []
for file in [precon_z0_file, precon_z1_file, precon_z2_file]:
    data = np.loadtxt(file)
    ss.append(data[:,2])
    vals_m.append(data[:,3])
    vals_me.append(data[:,4])
    vals_q.append(data[:,5])
    vals_qe.append(data[:,6])
    
ssr = []
vals_t = []
vals_te = []
vals_l = []
vals_le = []
for file in [recon_z0_file, recon_z1_file, recon_z2_file]:
    data = np.loadtxt(file)
    ssr.append(data[:,0])
    vals_t.append(data[:,1])
    vals_te.append(data[:,2])
    vals_l.append(data[:,4])
    vals_le.append(data[:,5])
    
f =13
fig = plt.figure(figsize=(10,10))
axes = []
ax0 = fig.add_subplot(3,3,1)
ax1 = fig.add_subplot(3,3,2, sharey=ax0)
ax2 = fig.add_subplot(3,3,3, sharey=ax0)


ax3 = fig.add_subplot(3,3,4, sharex=ax0)
ax4 = fig.add_subplot(3,3,5, sharex=ax1, sharey=ax3)
ax5 = fig.add_subplot(3,3,6, sharex=ax2, sharey=ax3)

ax6 = fig.add_subplot(3,3,7, sharex=ax0)
ax7 = fig.add_subplot(3,3,8, sharex=ax1, sharey=ax6)
ax8 = fig.add_subplot(3,3,9, sharex=ax2, sharey=ax6)

axes = np.array([[ax0,ax1,ax2],[ax3,ax4,ax5],[ax6,ax7,ax8]])
  
#fig, axes = plt.subplots(3, 3, figsize=(10,10), sharey=True)
fig.subplots_adjust(hspace=0, wspace=0)
#axes[0,0].set_title("$0.2 < z < 0.6$")
#axes[1,0].set_title("$0.4 < z < 0.8$")
#axes[2,0].set_title("$0.6 < z < 1.0$")
axes[0,0].set_title("WizCOLA mean")
axes[0,1].set_title("WiggleZ Unreconstructed")
axes[0,2].set_title("WiggleZ Reconstructed")

for i in range(3):
    axes[i, 0].yaxis.set_major_locator(MaxNLocator(6, prune="lower"))
    axes[i, 0].set_ylabel(r"$s^2 \xi(s) \ \left[ {\rm Mpc}^{2} \, h^{-2}  \right]$",fontsize=f)
    axes[2, i].set_xlabel(r"$s\ \  \left[{\rm Mpc}\, h^{-1}\right]$",fontsize=f)
    axes[2, i].xaxis.set_major_locator(MaxNLocator(6, prune="lower"))
    

    for j in range(3):
        if j == 0:
            axes[j, i].text(0.7, 0.9,'$z=0.44$', transform = axes[j, i].transAxes, fontsize=f)
        elif j == 1:
            axes[j, i].text(0.7, 0.9,'$z=0.60$', transform = axes[j, i].transAxes, fontsize=f)
        elif j == 2:
            axes[j, i].text(0.7, 0.9,'$z=0.73$', transform = axes[j, i].transAxes, fontsize=f)

for i,(s,v,e) in enumerate(zip(ssw, valsw_m, valsw_me)):
    axes[i, 0].errorbar(s, s*s*v, yerr=s*s*e, fmt='o', color='#303F9F', label=r"$\xi_0$")
        
for i,(s,v,e) in enumerate(zip(ss, vals_m, vals_me)):
    axes[i, 1].errorbar(s, s*s*v, yerr=s*s*e, fmt='o', color='#303F9F', label=r"$\xi_0$")
        
for i,(s,v,e) in enumerate(zip(ssr, vals_t, vals_te)):
    axes[i, 2].errorbar(s, s*s*v, yerr=s*s*e, fmt='o', color='#303F9F', label=r"$\xi_\perp$")


for i,(s,v,e) in enumerate(zip(ssw, valsw_q, valsw_qe)):
    axes[i,0].errorbar(s, s*s*v, yerr=s*s*e, fmt='d', color='#4CAF50', label=r"$\xi_2$")
    axes[i,0].legend(frameon=False, loc=2)
         
    
for i,(s,v,e) in enumerate(zip(ss, vals_q, vals_qe)):
    axes[i,1].errorbar(s, s*s*v, yerr=s*s*e, fmt='d', color='#4CAF50', label=r"$\xi_2$")
    axes[i,1].legend(frameon=False, loc=2)
        
for i,(s,v,e) in enumerate(zip(ssr, vals_l, vals_le)):
    axes[i,2].errorbar(s, s*s*v, yerr=s*s*e, fmt='d', color='#4CAF50', label=r"$\xi_\parallel$")
    axes[i,2].legend(frameon=False, loc=2)

#for i,(s,v,e) in enumerate(zip(ssr, vals_t, vals_te)):
#    axes[2,i].errorbar(s, s*s*v, yerr=s*s*e, fmt='o', color='b', label="Transverse")
#    axes[2,i].set_xlabel(r"$s\ \  \left[{\rm Mpc}\, h^{-1}\right]$",fontsize=f)
#    axes[2,i].set_ylabel(r"$s^2 \xi(s) \ \left[ {\rm Mpc}^{2} \, h^{-2}  \right]$",fontsize=f)
#    
#for i,(s,v,e) in enumerate(zip(ssr, vals_l, vals_le)):
#    axes[2,i].errorbar(s, s*s*v, yerr=s*s*e, fmt='o', color='r', label="Line of sight")
#    if i == 0:
#        axes[0,i].legend(frameon=False, loc=3)
#for i, cov in enumerate(rcovs):
#    axes[3,i].imshow(cov, cmap="viridis",interpolation='none')
#    axes[3,i].axvline(30, color='white', alpha=0.3)
#    axes[3,i].axhline(30, color='white', alpha=0.3)
#    axes[3,i].set_xticks([0,15,30,45,59])
#    axes[3,i].set_xticklabels([0,100,200,100,200])
#    axes[3,i].set_yticks([0,15,30,45,59])
#    axes[3,i].set_yticklabels([0,100,200,100,200])
#    axes[3,i].set_xlabel(r"$s\ \  \left[{\rm Mpc}\, h^{-1}\right]$",fontsize=f)
#    axes[3,i].set_ylabel(r"$s\ \  \left[{\rm Mpc}\, h^{-1}\right]$",fontsize=f)
#    axes[3,i].text(45, 5, r"$\xi_\parallel \times \xi_\perp$", color="white", fontsize=f)
#    axes[3,i].text(25, 5, r"$\xi_\parallel$", color="white", fontsize=f)
#    axes[3,i].text(55, 35, r"$\xi_\perp$", color="white", fontsize=f)

plt.setp(ax1.get_yticklabels(), visible=False)
plt.setp(ax2.get_yticklabels(), visible=False)

plt.setp(ax0.get_xticklabels(), visible=False)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)

plt.setp(ax3.get_xticklabels(), visible=False)
plt.setp(ax4.get_xticklabels(), visible=False)
plt.setp(ax5.get_xticklabels(), visible=False)

plt.setp(ax4.get_yticklabels(), visible=False)
plt.setp(ax5.get_yticklabels(), visible=False)
plt.setp(ax7.get_yticklabels(), visible=False)
plt.setp(ax8.get_yticklabels(), visible=False)

#plt.tight_layout()
fig.savefig("dataplot.pdf", bbox_inches="tight", dpi=300)
fig.savefig("dataplot.png", bbox_inches="tight", dpi=300)
    
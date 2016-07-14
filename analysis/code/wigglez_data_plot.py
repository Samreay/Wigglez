# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


precon_matrx_z0_file = "../data/wizcola/multipoles/cov/WizCOLA_Cij_xi0xi2_combined_z0pt2_0pt6.dat"
precon_matrx_z1_file = "../data/wizcola/multipoles/cov/WizCOLA_Cij_xi0xi2_combined_z0pt4_0pt8.dat"
precon_matrx_z2_file = "../data/wizcola/multipoles/cov/WizCOLA_Cij_xi0xi2_combined_z0pt6_1pt0.dat"

recon_matrx_z0_file = "../data/wizcola/multipoles/cov/WizCOLA_Cij_xi0xi2_combined_z0pt2_0pt6.dat"
recon_matrx_z1_file = "../data/wizcola/multipoles/cov/WizCOLA_Cij_xi0xi2_combined_z0pt4_0pt8.dat"
recon_matrx_z2_file = "../data/wizcola/multipoles/cov/WizCOLA_Cij_xi0xi2_combined_z0pt6_1pt0.dat"

precon_z0_file = "../data/wiggleznew/multipoles/WiggleZ_xi0xi2_all6regions_z0pt2_0pt6.dat"
precon_z1_file = "../data/wiggleznew/multipoles/WiggleZ_xi0xi2_all6regions_z0pt4_0pt8.dat"
precon_z2_file = "../data/wiggleznew/multipoles/WiggleZ_xi0xi2_all6regions_z0pt6_1pt0.dat"

recon_z0_file = "../data/wigglezrecon/WiggleZ_xiLxiT_z26.ascii"
recon_z1_file = "../data/wigglezrecon/WiggleZ_xiLxiT_z48.ascii"
recon_z2_file = "../data/wigglezrecon/WiggleZ_xiLxiT_z60.ascii"

covs = []
rcovs = []
for file in [precon_matrx_z0_file, precon_matrx_z1_file, precon_matrx_z2_file]:
    data = np.loadtxt(file)[:, 3].reshape(60, 60)
    #print(data.shape)
    R = np.zeros(data.shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            R[i,j] = data[i, j] / np.sqrt(data[i,i] * data[j,j])
    covs.append(R)
    
for file in [recon_matrx_z0_file, recon_matrx_z1_file, recon_matrx_z2_file]:
    data = np.loadtxt(file)[:, 3].reshape(60, 60)
    #print(data.shape)
    rcovs.append(data)
      
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
fig, axes = plt.subplots(2, 3, figsize=(12,7))

axes[0,0].set_title("$0.2 < z < 0.6$")
axes[0,1].set_title("$0.4 < z < 0.8$")
axes[0,2].set_title("$0.6 < z < 1.0$")

for i,(s,v,e) in enumerate(zip(ss, vals_m, vals_me)):
    axes[0,i].errorbar(s, s*s*v, yerr=s*s*e, fmt='o', color='b', label="Monopole")
    axes[0,i].set_xlabel(r"$s\ \  \left[{\rm Mpc}\, h^{-1}\right]$",fontsize=f)
    axes[0,i].set_ylabel(r"$s^2 \xi(s) \ \left[ {\rm Mpc}^{2} \, h^{-2}  \right]$",fontsize=f)

    
for i,(s,v,e) in enumerate(zip(ss, vals_q, vals_qe)):
    axes[0,i].errorbar(s, s*s*v, yerr=s*s*e, fmt='o', color='r', label="Quadrupole")
    if i == 0:
        axes[0,i].legend(frameon=False, loc=2)
for i, cov in enumerate(covs):
    axes[1,i].imshow(cov, cmap="viridis",interpolation='none')
    axes[1,i].axvline(30, color='white', alpha=0.3)
    axes[1,i].axhline(30, color='white', alpha=0.3)
    axes[1,i].set_xticks([0,15,30,45,59])
    axes[1,i].set_xticklabels([0,100,200,100,200])
    axes[1,i].set_yticks([0,15,30,45,59])
    axes[1,i].set_yticklabels([0,100,200,100,200])
    axes[1,i].set_xlabel(r"$s\ \  \left[{\rm Mpc}\, h^{-1}\right]$",fontsize=f)
    axes[1,i].set_ylabel(r"$s\ \  \left[{\rm Mpc}\, h^{-1}\right]$",fontsize=f)
    axes[1,i].text(45, 5, r"$\xi_0 \times \xi_2$", color="white", fontsize=f)
    axes[1,i].text(25, 5, r"$\xi_0$", color="white", fontsize=f)
    axes[1,i].text(55, 35, r"$\xi_2$", color="white", fontsize=f)


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


plt.tight_layout()
fig.savefig("dataplot.pdf", bbox_inches="tight", dpi=300)
fig.savefig("dataplot.png", bbox_inches="tight", dpi=300)
    
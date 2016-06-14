import numpy as np
import matplotlib.pyplot as plt


precon_matrx_z0_file = "../data/wizcola/multipoles/cov/WizCOLA_Cij_xi0xi2_combined_z0pt2_0pt6.dat"
precon_matrx_z1_file = "../data/wizcola/multipoles/cov/WizCOLA_Cij_xi0xi2_combined_z0pt4_0pt8.dat"
precon_matrx_z2_file = "../data/wizcola/multipoles/cov/WizCOLA_Cij_xi0xi2_combined_z0pt6_1pt0.dat"

recon_matrx_z0_file = "../data/wizcolarecon/WiZCOLA_xiL_postrecon_allmocks"
recon_matrx_z1_file = "../data/wizcolarecon/multipoles/cov/WizCOLA_Cij_xi0xi2_combined_z0pt4_0pt8.dat"
recon_matrx_z2_file = "../data/wizcolarecon/multipoles/cov/WizCOLA_Cij_xi0xi2_combined_z0pt6_1pt0.dat"

covs = []
rcovs = []
for file in [precon_matrx_z0_file, precon_matrx_z1_file, precon_matrx_z2_file]:
    data = np.loadtxt(file)[:, 3].reshape(60, 60)
    print(data.shape)
    covs.append(data)
    
for file in [recon_matrx_z0_file, recon_matrx_z1_file, recon_matrx_z2_file]:

fig, axes = plt.subplots(4, 3, figsize=(10,10))
axes[1,0].imshow(covs[0], cmap="viridis")
axes[1,1].imshow(covs[1], cmap="viridis")
axes[1,2].imshow(covs[2], cmap="viridis")
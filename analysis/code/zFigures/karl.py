import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
# Figures for Karl



# Define the data we want to plot

# Alam 2016
alam_zs = np.array([0.32, 0.57])
alam_dm = np.array([1294, 2179])
alam_dm_error = np.array([21, 35])
alam_da = alam_dm / (1 + alam_zs)
alam_da_error = alam_dm_error / (1 + alam_zs)
alam_h = np.array([78.4, 96.6])
alam_h_error = np.array([2.3, 2.4])
alam_daonh = alam_da / alam_h
alam_daonh_error = np.sqrt((alam_da_error/alam_da)**2 + (alam_h_error/alam_h)**2) * alam_daonh

#Anderson 2014
anderson_zs = np.array([0.57])
anderson_da = np.array([1321])
anderson_da_error = np.array([20])
anderson_h = np.array([96.8])
anderson_h_error = np.array([3.4])
anderson_daonh = anderson_da / anderson_h
anderson_daonh_error = np.sqrt((anderson_da_error/anderson_da)**2 + (anderson_h_error/anderson_h)**2) * anderson_daonh

# Wigglez pre recon
wig_pre_zs = np.array([0.44, 0.6, 0.73])
wig_pre_da = np.array([1330, 1280, 1340])
wig_pre_da_error_up = np.array([150, 190, 150])
wig_pre_da_error_down = np.array([150, 160, 130])
wig_pre_da_error = 0.5 * (wig_pre_da_error_up + wig_pre_da_error_down)
wig_pre_h = np.array([85, 91, 80])
wig_pre_h_error_up = np.array([19, 15, 9])
wig_pre_h_error_down = np.array([12, 14, 10])
wig_pre_h_error = 0.5 * (wig_pre_h_error_up + wig_pre_h_error_down)
wig_pre_daonh = wig_pre_da / wig_pre_h
wig_pre_daonh_error = np.sqrt((wig_pre_da_error/wig_pre_da)**2 + (wig_pre_h_error/wig_pre_h)**2) * wig_pre_daonh

# Wigglez post recon
wig_post_zs = np.array([0.60, 0.73])
wig_post_da = np.array([10.3, 9.8])
wig_post_da_error_up = np.array([0.4, 1.1])
wig_post_da_error_down = np.array([0.5, 0.4])
wig_post_czonH = np.array([11.5, 15.3])
wig_post_czonH_error_up = np.array([1.3, 1.6])
wig_post_czonH_error_down = np.array([1.6, 1.8])


# Function for getting r_s

def get_r_s():
    # Fiducial
    om = 0.273
    h = 0.705
    ob = 0.0456
    ol = 0.727
    og = 2.469e-5 / (h * h)
    orr = og * (1 + (7.0/8.0)*np.power(4.0/11.0, 4.0/3.0)*3.04)
    
    c = 3e5
    z = 1020
    ae = 1.0 / (1.0 + z)
    a = np.linspace(ae/100000, ae, 100)
    ez = np.sqrt(om / (a*a*a) + ol + orr/(a*a*a*a))
    hs = h * ez
    fac = 1/np.sqrt(3 * (1 + (3 * ob / (4 * og))*a))
    integrand = (c/100) * fac / (a*a*hs)
    integral = simps(integrand, x=a)
    return integral
rs = get_r_s()

# Adjust wig post recon to remove r_s
wig_post_da = np.array(wig_post_da) * rs
wig_post_da_error_up = np.array(wig_post_da_error_up) * rs
wig_post_da_error_down = np.array(wig_post_da_error_down) * rs
wig_post_da_error = 0.5 * (wig_post_da_error_up + wig_post_da_error_down)
wig_post_h = 3e5 * np.array(wig_post_zs) / (np.array(wig_post_czonH) * rs)
wig_post_h_error_up = (3e5 * np.array(wig_post_zs) / ((np.array(wig_post_czonH) + np.array(wig_post_czonH_error_up)) * rs)) - wig_post_h
wig_post_h_error_down = (3e5 * np.array(wig_post_zs) / ((np.array(wig_post_czonH) + np.array(wig_post_czonH_error_down)) * rs)) - wig_post_h
wig_post_h_error = 0.5 * (wig_post_h_error_up + wig_post_h_error_down)
wig_post_daonh = wig_post_da / wig_post_h
wig_post_daonh_error = np.sqrt((wig_post_da_error/wig_post_da)**2 + (wig_post_h_error/wig_post_h)**2) * wig_post_daonh


planck_om = 0.3089
planck_om_error = 0.0062
planck_h = 67.74
planck_h_error = 0.46


top_cosmology = FlatLambdaCDM(planck_h + planck_h_error, planck_om + planck_om_error)
bottom_cosmology = FlatLambdaCDM(planck_h - planck_h_error, planck_om - planck_om_error)
zs = np.linspace(0.2, 0.9, 100)
das = np.linspace(1000, 1500, 1000)
top_da = top_cosmology.angular_diameter_distance(zs).value
bottom_da = bottom_cosmology.angular_diameter_distance(zs).value
top_h = top_cosmology.H(zs).value
bottom_h = bottom_cosmology.H(zs).value
top_div = top_da / top_h
bottom_div = bottom_da / bottom_h


fig, ax = plt.subplots(nrows=3, figsize=(5,14), sharex=True)
fig.subplots_adjust(hspace=0.05)

# Plot planck cosmology
ax[0].plot(zs, bottom_da, color='k', alpha=0.2)
ax[0].plot(zs, top_da, color='k', alpha=0.2)
ax[0].fill_between(zs, bottom_da, top_da, color='k', alpha=0.2, label="Planck (2015)")
ax[1].fill_between(zs, bottom_h, top_h, color='k', alpha=0.2)
ax[2].plot(zs, top_div, color='k', alpha=0.2)
ax[2].plot(zs, bottom_div, color='k', alpha=0.2)
ax[2].fill_between(zs,top_div, bottom_div, color='k', alpha=0.2, label="Planck (2015)")

# Plot Alam points
offset = 0.005

ax[0].errorbar(alam_zs - offset, alam_da, yerr=alam_da_error, fmt='o', ms=4, label="Alam et al. (2016)", color='r')
ax[1].errorbar(alam_zs - offset, alam_h, yerr=alam_h_error, fmt='o', ms=4, color='r')
ax[2].errorbar(alam_zs - offset, alam_daonh, yerr=alam_daonh_error, fmt='o', ms=4, color='r')

# Plot Anderson points
ax[0].errorbar(anderson_zs + offset, anderson_da, yerr=anderson_da_error, fmt='o', ms=4, label="Anderson et al. (2014)", color='g')
ax[1].errorbar(anderson_zs + offset, anderson_h, yerr=anderson_h_error, fmt='o', ms=4, color='g')
ax[2].errorbar(anderson_zs + offset, anderson_daonh, yerr=anderson_daonh_error, fmt='o', ms=4, color='g')

# Plot pre-recon
ax[0].errorbar(wig_pre_zs - offset, wig_pre_da, yerr=[wig_pre_da_error_up, wig_pre_da_error_down], fmt='o', ms=4, color='b', label="WiggleZ pre-recon.")
ax[1].errorbar(wig_pre_zs - offset, wig_pre_h, yerr=[wig_pre_h_error_up, wig_pre_h_error_down], fmt='o', ms=4, color='b', label="WiggleZ pre-recon.")
ax[2].errorbar(wig_pre_zs - offset, wig_pre_daonh, yerr=wig_pre_daonh_error, fmt='o', ms=4, color='b')

# Plot post-recon
ax[0].errorbar(wig_post_zs + offset, wig_post_da, yerr=[wig_post_da_error_up, wig_post_da_error_down], fmt='s', ms=6, color='b', label="WiggleZ post-recon.")
ax[1].errorbar(wig_post_zs + offset, wig_post_h, yerr=[wig_post_h_error_up, wig_post_h_error_down], fmt='s', ms=6, color='b', label="WiggleZ post-recon.")
ax[2].errorbar(wig_post_zs + offset, wig_post_daonh, yerr=wig_post_daonh_error, fmt='s', ms=4, color='b')


ax[0].legend(frameon=False, loc=4)
ax[2].set_xlabel("$z$", fontsize=16)
ax[0].set_ylabel(r"$D_A(z)\ {\rm[Mpc]}$", fontsize=16)
ax[1].set_ylabel(r"$H(z)\ {\rm[km}\ {\rm s}^{-1}\ {\rm Mpc}^{-1}{\rm]}$", fontsize=16)
ax[2].set_ylabel(r"$D_A(z)/H(z)\ {\rm[Mpc}^2\ {\rm s}\ {\rm km}^{-1}]$", fontsize=16)

ax[1].yaxis.get_major_ticks()[-1].set_visible(False)
ax[2].yaxis.get_major_ticks()[-1].set_visible(False)

fig.savefig("karl.pdf", transparent=True, dpi=300, bbox_inches="tight")




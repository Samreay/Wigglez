from __future__ import print_function
import numpy as np
import sys
import os
import time
import scipy.stats
import pickle
from scipy.interpolate import interp1d
from scipy.integrate import simps
import hashlib
from chain import ChainConsumer
import matplotlib.pyplot as plt


if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import wizcola
    import wigglezold
    import wiggleznew
    from mcmc import *
    import methods
    from fitters import *


    precon_z0_file = "../wiggleznew/multipoles/WiggleZ_xi0xi2_all6regions_z0pt2_0pt6.dat"
    precon_z1_file = "../wiggleznew/multipoles/WiggleZ_xi0xi2_all6regions_z0pt4_0pt8.dat"
    precon_z2_file = "../wiggleznew/multipoles/WiggleZ_xi0xi2_all6regions_z0pt6_1pt0.dat"
   
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
   
   
    f = 14
    fig, axes = plt.subplots(3, 1, figsize=(5,11))
    
    axes[0].set_title("$0.2 < z < 0.6$")
    axes[1].set_title("$0.4 < z < 0.8$")
    axes[2].set_title("$0.6 < z < 1.0$")
    
    for i,(s,v,e) in enumerate(zip(ss, vals_m, vals_me)):
        axes[i].errorbar(s, s*s*v, yerr=s*s*e, fmt='o', color='b', ms=5, label="Monopole", alpha=0.5)
        axes[i].set_xlabel(r"$s\ \  \left[{\rm Mpc}\, h^{-1}\right]$",fontsize=f)
        axes[i].set_ylabel(r"$s^2 \xi(s) \ \left[ {\rm Mpc}^{2} \, h^{-2}  \right]$",fontsize=f)
        
    for i,(s,v,e) in enumerate(zip(ss, vals_q, vals_qe)):
        axes[i].errorbar(s, s*s*v, yerr=s*s*e, fmt='o', color='r', ms=5, label="Quadrupole", alpha=0.5)

            


    # Get params
    #fitters = [WigglezMultipoleBin(0),WigglezMultipoleBin(1),WigglezMultipoleBin(2)]
    params = [a[0] for a in fitters[0].getParams()]
    print(params)

    x = fitters[0].dataX
    mps = []
    qps = []
    # Get the max likelihood point
    for i, fitter in enumerate(fitters):
        directory = os.path.dirname(__file__) + "/../bWigMpBin/bWigMpBin_z%d" % i
        files = os.listdir(directory)
        chain = np.load(directory + os.sep + files[0])
        for f in files[1:]:
            chain = np.concatenate((chain, np.load(directory + os.sep + f)))
        print(chain.shape)
        chain = chain[:, 3:]
        
        c = ChainConsumer()
        c.add_chain(chain, parameters=params, name="z=%d"%i)
        summary = c.get_summary()[0]
        maxes = {k: v[1] for k,v in summary.items()}
        
        plist = [maxes[p] for p in params]
        
        model = fitter.getModel(plist, fitter.dataX)
        mp = model[:x.size]
        qp = model[x.size:]
        mps.append(mp)
        qps.append(qp)
        
        axes[i].plot(x, x*x*mp, color='b',lw=2, label=r"$\xi_0$ Best fit")
        axes[i].plot(x, x*x*qp, color='r', lw=2, ls='--', label=r"$\xi_2$ Best fit")
        if i == 0:
            axes[i].legend(frameon=False, loc=2)
    plt.tight_layout()
    fig.savefig("prerecon_result.pdf", bbox_inches='tight', dpi=300, transparent=True)    
    plt.show()
    
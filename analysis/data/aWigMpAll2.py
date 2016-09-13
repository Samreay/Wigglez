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


if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
    import wizcola
    import wigglezold
    import wiggleznew
    from mcmc import *
    import methods
    from fitters import *
    
    args = sys.argv
    t = "aWigMpAll"
    print(t)
    walk = 0
    if len(args) > 1: 
        walk = int(args[1])

    fitter = WigglezMultipoleAll()
    
    cambMCMCManager = CambMCMCManager(t, fitter, debug=True)
    cambMCMCManager.configureMCMC(numCalibrations=15,calibrationLength=1000, thinning=2, maxSteps=100000)
    import matplotlib, matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.path import Path
    from matplotlib import rc
    from pylab import *
    import matplotlib.lines as mlines
    
    
    uids = [t]
    for u in uids:
    	cambMCMCManager.consolidateData(uid=u)
    	#cambMCMCManager.testConvergence(uid=u)
    	#cambMCMCManager.getTamParameterBounds(uid=u)
    #cambMCMCManager.plotResults(uids=uids)
    #cambMCMCManager.plotResults(uids=uids, parameters=["omch2", "alpha", "epsilon"], labels=["Combined"], filename=t)
    #cambMCMCManager.plotResults(uids=uids, parameters=["omch2", "alphaPerp", "alphaParallel"], labels=["Combined"], filename=t)
    #cambMCMCManager.plotWalk("omch2", "alpha", final=True, uid=t)
    i_omch2 = cambMCMCManager.extraFitters[u].getIndex("omch2")
    i_alpha = cambMCMCManager.extraFitters[u].getIndex("alpha")
    i_epsilon = cambMCMCManager.extraFitters[u].getIndex("epsilon")
    weights = cambMCMCManager.finalSteps[u][:, 1]
    omch2 = cambMCMCManager.finalSteps[u][:, i_omch2]
    alpha = cambMCMCManager.finalSteps[u][:, i_alpha]
    epsilon = cambMCMCManager.finalSteps[u][:, i_epsilon]
    print(omch2.mean(), omch2.shape)
    print(alpha.mean(), alpha.shape)
    print(epsilon.mean(), epsilon.shape)
    data = np.vstack((weights, omch2, alpha, epsilon))
    np.save("allWigMpChain.npy", data)
    #cambMCMCManager.plotWalk("beta", "b20", final=True, uid=t)
    #cambMCMCManager.plotWalk("alpha", "epsilon", final=True, uid=t)
    #cambMCMCManager.plotWalk("lorentzian", "sigmav", final=True, uid=t)

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

def configureAndDoWalk(manager, fitter, uid, walk, chunkLength=None):
    print("Doing walk %d for uid %s" % (walk, uid))
    manager.fitter = fitter
    manager.uid = uid
    seed = int(np.abs((int(time.time()) + abs(hash(uid))) * (walk + 1)) % 1000000000)
    print("SEED IS %d" % seed)
    np.random.seed(seed)
    manager.doWalk(walk, chunkLength=chunkLength)

if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import wizcola
    import wigglezold
    import wiggleznew
    from mcmc import *
    import methods
    from fitters import *

    args = sys.argv
    t = os.path.basename(os.path.dirname(os.path.abspath(__file__))).encode("ascii")
    template = "%s_z%d"

    bins = [0,1,2]
    uids = [template%(t,b) for b in bins]
    fitters = [WigglezNewMonopoleFitter(bin=b) for b in bins]
    print(uids)
    
    job = None
    walk = None
    bin = 0
    if len(args) > 1: 
        job = int(args[1])
    if job is not None:
        walk = job / len(bins)
        bin = job % len(bins)
    
    dataDir = '/data/uqshint1/%s'%t
    print(dataDir)
    
    if job is not None:
        cambMCMCManager = CambMCMCManager(uids[bin], fitters[bin], dataDir=dataDir, debug=False)
        cambMCMCManager.configureMCMC(numCalibrations=15,calibrationLength=1000, thinning=2, maxSteps=300000)
        cambMCMCManager.configureSaving(stepsPerSave=1000)
        configureAndDoWalk(cambMCMCManager, fitters[bin], uids[bin], walk)
    else:
        cambMCMCManager = CambMCMCManager(t, fitters[0], debug=True)
        cambMCMCManager.configureMCMC(numCalibrations=15,calibrationLength=1000, thinning=2, maxSteps=300000)
        import matplotlib, matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.patches
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.path import Path
        from matplotlib import rc
        from pylab import *
        import matplotlib.lines as mlines
        

        for u,f in zip(uids, fitters):
            cambMCMCManager.consolidateData(uid=u, fitter=f)
            #cambMCMCManager.testConvergence(uid=u)
            cambMCMCManager.getTamParameterBounds(uid=u)
        #cambMCMCManager.plotResults(uids=uids)
        cambMCMCManager.plotResults(uids=uids, size=(6,6), parameters=["omch2", "alpha"], labels=["$0.2<z<0.6$","$0.4<z<0.8$","$0.6<z<1.0$"], filename=t)
        #cambMCMCManager.plotResults(uids=uids, parameters=["omch2", "alpha", "beta", "b20"])
        #cambMCMCManager.plotWalk("omch2", "alpha", final=True, uid=uids[0])
        #cambMCMCManager.plotWalk("alpha", "epsilon", final=True, uid=uids[0])
        #cambMCMCManager.plotWalk("sigmav", "omch2", final=True, uid=uids[0])
        #cambMCMCManager.plotWalk("b20", "beta", final=True, uid=uids[0])
        #cambMCMCManager.plotWalk("lorentzian", "beta", final=True, uid=uids[0])


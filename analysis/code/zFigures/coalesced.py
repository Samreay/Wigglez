from __future__ import print_function
import numpy as np
import sys
import os
from multiprocessing import Pool
import matplotlib, matplotlib.pyplot as plt
import time
import matplotlib.gridspec as gridspec
import matplotlib.patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats
from matplotlib.path import Path
from matplotlib import rc
from pylab import *
import pickle
import matplotlib.lines as mlines
from scipy.interpolate import interp1d
from scipy.integrate import simps
from joblib import Parallel, delayed
import hashlib
import wizcola
import wigglezold
from mcmc import *
import methods
from cambMCMC import *

def doFit(cov, finalData):
    uid = "ztesting123"
    f = CovarianceFitter(debug=False)
    f.setCovariance(cov)
    f.setData(finalData)
    manager = CambMCMCManager(uid, f, debug=False)
    manager.configureMCMC(numCalibrations=11,calibrationLength=1000, thinning=2, maxSteps=200000)
    manager.configureSaving(stepsPerSave=50000)

    manager.doWalk(0)
    manager.doWalk(1)
    manager.doWalk(2)
    manager.consolidateData()
    manager.testConvergence()
    manager.getTamParameterBounds(numBins=25)
    manager.plotResults()
    return manager
# doFit(cov, finals[1])


def configureAndDoWalk(manager, fitter, uid, walk, chunkLength=None):
    manager.fitter = fitter
    manager.uid = uid
    np.random.seed((int(time.time()) + abs(hash(uid))) * (walk + 1))
    manager.doWalk(walk, chunkLength=chunkLength)

if __name__ == '__main__':


    monopoleFolder = 'aIndividal'
    wedgeFolder = 'aWedges'

    monopoleFitter = WizColaSingleMultipole(0)
    wedgeFitter = WizColaSingleWedge(0)

    wigMP = 'aWigglezMultipole'
    wigWg = 'aWigglezWedges'

    #cambMCMCManager = CambMCMCManager(text, monopoleFitter, debug=True)
    try:
        raise Exception("I HATE YOU")
        print(finalCor)
    except Exception:
        finalMP = []
        finalMPW = []
        finalWdg = []
        finalWdgW = []
        for dirr, fitter in [(monopoleFolder, monopoleFitter), (wedgeFolder, wedgeFitter)]:
    
            uids = []
            labels = []
            params = {}
            convs = []
            path = os.path.dirname(os.path.realpath(__file__)) + os.sep + dirr
            for directory in [d for d in os.listdir(path) if os.path.isdir(path + os.sep + d) and not d.startswith('z')]:
                cambMCMCManager = CambMCMCManager('doom', fitter, debug=True)
                data = cambMCMCManager.consolidateData(uid=directory, path=path)
                if data > 100000:
                    uids.append(directory)
                    labels.append(directory)
                    con = np.mean(cambMCMCManager.testConvergence(uid=directory))
                    convs.append(con)
                    threshold = 0.1
                    if np.abs(con - 1) < threshold:
                        split = directory.split("z")
                        b = cambMCMCManager.getTamParameterBounds(numBins=25, uid=directory)
                        bests = {}
                        d2 = None
                        search = ['omch2', 'alpha', 'epsilon', 'alphaPerp', 'alphaParallel']
                        for s in search:
                            if b.get(s) is not None:
                                bests[s] = b[s][1]
                                if d2 is None:
                                    d2 = np.power(cambMCMCManager.finalSteps[directory][:, cambMCMCManager.fitter.getIndex(s)] - b[s][1], 2)
                                else:
                                    d2 += np.power(cambMCMCManager.finalSteps[directory][:, cambMCMCManager.fitter.getIndex(s)] - b[s][1], 2)
                        chi2 = cambMCMCManager.finalSteps[directory][d2.argmin(), 0]
                        prob = np.exp(-chi2 / 2.0)
                        b['prob'] = prob
                        print(chi2)
                        dic = {split[1]: b}
                        if params.get(split[0]) is None:
                            params[split[0]] = dic
                        else:
                            params[split[0]].update(dic)
                    else:
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! %s did not converge: %0.3f" % (directory, con))
                    #gc.collect()
            for realisation, bins in params.iteritems():
                if len(bins.keys()) == 3:
                    a1 = 1.25
                    a2 = 0.75
                    e1 = 0.35
                    e2 = -0.35

                    if (dirr == monopoleFolder):
                        if bins['0']['alpha'][1] > a1 or bins['0']['alpha'][1] < a2 or  bins['1']['alpha'][1] > a1 or bins['1']['alpha'][1] < a2 or bins['2']['alpha'][1] > a1 or bins['2']['alpha'][1] < a2:
                            continue
                        if bins['0']['epsilon'][1] > e1 or bins['0']['epsilon'][1] < e2 or bins['1']['epsilon'][1] > e1 or bins['1']['epsilon'][1] < e2 or bins['2']['epsilon'][1] > e1 or bins['2']['epsilon'][1] < e2:
                            continue
                        print(realisation)

                        finalMP.append([bins['0']['omch2'][1], bins['1']['omch2'][1], bins['2']['omch2'][1], bins['0']['alpha'][1], bins['1']['alpha'][1], bins['2']['alpha'][1], bins['0']['epsilon'][1], bins['1']['epsilon'][1], bins['2']['epsilon'][1]])
                        #finalMPW.append(bins['0']['prob']  * bins['1']['prob'] * bins['2']['prob'])
                        finalMPW.append(1)
                        #finalMP.append([bins['0']['omch2'][1], bins['0']['alpha'][1], bins['0']['epsilon'][1], bins['1']['omch2'][1], bins['1']['alpha'][1], bins['1']['epsilon'][1], bins['2']['omch2'][1], bins['2']['alpha'][1], bins['2']['epsilon'][1]])
                    else:
                        if bins['0']['alphaPerp'][1] > a1 or bins['0']['alphaPerp'][1] < a2 or  bins['1']['alphaPerp'][1] > a1 or bins['1']['alphaPerp'][1] < a2 or bins['2']['alphaPerp'][1] > a1 or bins['2']['alphaPerp'][1] < a2:
                            continue
                        if bins['0']['alphaParallel'][1] > a1 or bins['0']['alphaParallel'][1] < a2 or  bins['1']['alphaParallel'][1] > a1 or bins['1']['alphaParallel'][1] < a2 or bins['2']['alphaParallel'][1] > a1 or bins['2']['alphaParallel'][1] < a2:
                            continue
                        print(realisation)
                        finalWdg.append([bins['0']['omch2'][1], bins['1']['omch2'][1], bins['2']['omch2'][1], bins['0']['alphaPerp'][1], bins['1']['alphaPerp'][1], bins['2']['alphaPerp'][1], bins['0']['alphaParallel'][1], bins['1']['alphaParallel'][1], bins['2']['alphaParallel'][1]])
                        #finalWdg.append([bins['0']['omch2'][1], bins['0']['alphaPerp'][1], bins['0']['alphaParallel'][1], bins['1']['omch2'][1], bins['1']['alphaPerp'][1], bins['1']['alphaParallel'][1], bins['2']['omch2'][1], bins['2']['alphaPerp'][1], bins['2']['alphaParallel'][1]])
                        #finalWdgW.append(bins['0']['prob'] * bins['1']['prob'] * bins['2']['prob'])
                        finalWdgW.append(1)

    
    
        finalMP = np.array(finalMP)
        finalWdg = np.array(finalWdg)
        finalCor = []
        for name, f, w in [('mp.npy', finalMP, finalMPW),('wdg.npy', finalWdg, finalWdgW)]:
            means = np.mean(f, axis=0)
            print(f)
            n = len(means)
            vari = np.sqrt( np.sum((f - means)*(f-means), axis=0) / n )
            deviations = f - means
            wsum = sum(np.array(w)**2)
    
            covs = np.zeros((n, n))
            cor = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    covs[i,j] = np.sum(np.sqrt(w[i])*deviations[:,i] * np.sqrt(w[j])*deviations[:, j]) / (wsum)
                    
            for i in range(n):
                for j in range(n):
                    cor[i,j] = covs[i,j] / np.sqrt(covs[i,i] * covs[j,j])
    
            finalCor.append(cor)
            np.save(name, covs)
        
    cmap=methods.viridis
    #cmap = methods.shiftedColorMap(methods.viridis3, midpoint=0.5, stop=1.1)
    fig = plt.figure(figsize=(10,5), dpi=300)

    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams['axes.labelsize'] = 14
    rc('text', usetex=False)
    matplotlib.rcParams['xtick.labelsize'] = 16
    matplotlib.rcParams['ytick.labelsize'] = 16
    ax0 = fig.add_subplot(1,2,1)

    h = ax0.imshow(finalCor[0], interpolation='none', cmap=cmap, vmin=-0.5)
    plt.xticks(range(9), [r"$\Omega_{c}h^2_{0}$", r"$\Omega_{c}h^2_{1}$", r"$\Omega_{c}h^2_{2}$", r"$\alpha_0$", r"$\alpha_1$", r"$\alpha_2$", r"$\epsilon_0$", r"$\epsilon_1$", r"$\epsilon_2$"], rotation='vertical')
    plt.yticks(range(9), [r"$\Omega_{c}h^2_{0}$", r"$\Omega_{c}h^2_{1}$", r"$\Omega_{c}h^2_{2}$", r"$\alpha_0$", r"$\alpha_1$", r"$\alpha_2$", r"$\epsilon_0$", r"$\epsilon_1$", r"$\epsilon_2$"])
    #plt.xticks(range(9), [r"$\Omega_{c}h^2_{0}$", r"$\alpha_0$",  r"$\epsilon_0$",  r"$\Omega_{c}h^2_{1}$", r"$\alpha_1$", r"$\epsilon_1$",  r"$\Omega_{c}h^2_{2}$", r"$\alpha_2$",r"$\epsilon_2$"], rotation='vertical')
    #plt.yticks(range(9), [r"$\Omega_{c}h^2_{0}$", r"$\alpha_0$",  r"$\epsilon_0$",  r"$\Omega_{c}h^2_{1}$", r"$\alpha_1$", r"$\epsilon_1$",  r"$\Omega_{c}h^2_{2}$", r"$\alpha_2$",r"$\epsilon_2$"])
    cbar = fig.colorbar(h, fraction=0.0458, pad=0.04)
    cbar.ax.tick_params(labelsize=12) 

    ax1 = fig.add_subplot(1,2,2)

    h2 = ax1.imshow(finalCor[1], interpolation='none', cmap=cmap, vmin=-0.5)
    plt.xticks(range(9), [r"$\Omega_{c}h^2_{0}$", r"$\Omega_{c}h^2_{1}$", r"$\Omega_{c}h^2_{2}$", r"$\alpha_{\perp 0}$", r"$\alpha_{\perp 1}$", r"$\alpha_{\perp 2}$", r"$\alpha_{\parallel0}$", r"$\alpha_{\parallel1}$", r"$\alpha_{\parallel2}$"], rotation='vertical')
    plt.yticks(range(9), [r"$\Omega_{c}h^2_{0}$", r"$\Omega_{c}h^2_{1}$", r"$\Omega_{c}h^2_{2}$", r"$\alpha_{\perp 0}$", r"$\alpha_{\perp 1}$", r"$\alpha_{\perp 2}$", r"$\alpha_{\parallel0}$", r"$\alpha_{\parallel1}$", r"$\alpha_{\parallel2}$"])
    #plt.xticks(range(9), [r"$\Omega_{c}h^2_{0}$", r"$\alpha_{\perp 0}$",  r"$\alpha_{\parallel0}$",  r"$\Omega_{c}h^2_{1}$",  r"$\alpha_{\perp 1}$", r"$\alpha_{\parallel1}$",r"$\Omega_{c}h^2_{2}$",r"$\alpha_{\perp 2}$",  r"$\alpha_{\parallel2}$"], rotation='vertical')
    #plt.yticks(range(9), [r"$\Omega_{c}h^2_{0}$", r"$\alpha_{\perp 0}$",  r"$\alpha_{\parallel0}$",  r"$\Omega_{c}h^2_{1}$",  r"$\alpha_{\perp 1}$", r"$\alpha_{\parallel1}$",r"$\Omega_{c}h^2_{2}$",r"$\alpha_{\perp 2}$",  r"$\alpha_{\parallel2}$"])
    cbar = fig.colorbar(h2, fraction=0.0458, pad=0.04)
    cbar.ax.tick_params(labelsize=12) 
    fig.subplots_adjust(wspace=.4)


    fig.savefig("correlations.pdf", bbox_inches='tight', dpi=600, transparent=True)
    

    """
    else:
        doFit(covs, finals[1])
        d = 2
        uids = ["single_%d_z%d" % (d, z) for z in range(3)]
        for u in uids:
            cambMCMCManager.consolidateData(uid=u)
            cambMCMCManager.testConvergence(uid=u)
            cambMCMCManager.getTamParameterBounds(numBins=25, uid=u)
        cambMCMCManager.consolidateData(uid="ztesting123", fitter=CovarianceFitter())
        uids.append("ztesting123")
        cambMCMCManager.getTamParameterBounds(numBins=25, uid="ztesting123")
        cambMCMCManager.plotResults(uids=uids, parameters=["omch2", "alpha", "epsilon"], labels=["$0.2<z<0.6$", "$0.4<z<0.8$", "$0.6<z<1.0$", r"$\rm{Combined}$"], filename="corCombined_%d" % d)"""

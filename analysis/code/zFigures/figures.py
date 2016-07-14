from __future__ import print_function
import numpy as np
import sys
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
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import os
from scipy.integrate import simps

figsize=(6,6)



def plotLinear():
    fig = plt.figure(figsize=figsize, dpi=300)
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams['axes.labelsize'] = 20
    rc('text', usetex=False)
    matplotlib.rcParams['xtick.labelsize'] = 14
    matplotlib.rcParams['ytick.labelsize'] = 14

    ax = fig.add_subplot(111)
    ax.set_title("$P_{\mathrm{lin}}(k)$", fontsize=24, y=1.015)

    
    axins = zoomed_inset_axes(ax, 2.5, loc=3)
    axins.set_xlim(0.03,0.4)
    axins.set_ylim(900, 3.5e4)
    plt.setp(axins.get_xticklabels(), visible=False)
    plt.setp(axins.get_yticklabels(), visible=False)
    
    ax.plot(ks, pklin, linewidth=2)
    axins.plot(ks, pklin, linewidth=2)


    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax.set_xscale('log')
    ax.set_yscale('log')
    axins.set_xscale('log')
    axins.set_yscale('log')
    ax.set_xlabel('k/h')
    ax.set_ylabel('Power')
    
    #fig.savefig("linear.png", bbox_inches='tight', dpi=300, transparent=True)
    
def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

def plotNW():
    fig = plt.figure(figsize=figsize, dpi=300)
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams['axes.labelsize'] = 20
    rc('text', usetex=False)
    matplotlib.rcParams['xtick.labelsize'] = 14
    matplotlib.rcParams['ytick.labelsize'] = 14


    ax0 = fig.add_subplot(211)  
    ax1 = fig.add_subplot(212)  
    
    rect = [0.2,0.35,0.4,0.4]

    axins = add_subplot_axes(ax0, rect)
    axins.set_xlim(0.04,1)
    axins.set_ylim(0.95, 1.05)
    


    
    
    #plt.setp(axins.get_xticklabels(), visible=False)
    #plt.setp(axins.get_yticklabels(), visible=False)
    axins.set_title(r"$\mathrm{Normalised\ by\ }P_{\mathrm{nw}}$")


    ax0.plot(ks, pklin, 'b--', linewidth=1)
    ax0.plot(ks, pkdw, 'r-', linewidth=1)
    axins.plot(ks, np.ones(ks.size), 'k:', linewidth=1)
    axins.plot(ks, pklin/pknw, 'b--', linewidth=1)
    axins.plot(ks, pkdw/pknw, 'r-', linewidth=1)

    #mark_inset(ax0, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    
    ax1.plot(ss, ss * ss * datapointsLin, 'b--', linewidth=1, label="$P_{\mathrm{lin}}$")
    ax1.plot(ss, ss * ss * datapointsDW, 'r-', linewidth=1, label="$P_{\mathrm{dw}}$")
    ax1.plot(ss, ss * ss * datapointsNW, 'k:', linewidth=1, label="$P_{\mathrm{nw}}$")

    ax1.legend(frameon=False)


    ax1.set_xlabel('$s \ \ [\mathrm{Mpc/h}]$')
    ax1.set_ylabel(r'$s^2 \xi(s)$')


    ax0.set_xscale('log')
    ax0.set_yscale('log')
    
    ax0.set_xlim(0.001, 10)    
    ax0.set_ylim(1, 70000)
    
    ax1.set_xlim(5,160)
    axins.set_xscale('log')
    #axins.set_yscale('log')
    ax0.set_xlabel('$\mathrm{k/h}\ \  [\mathrm{Mpc}^{-1}]$')
    ax0.set_ylabel('$\mathrm{Power}$')
    #ax0.yaxis.set_major_locator(plt.MaxNLocator(15))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
    plt.tight_layout()
    fig.savefig("dwExample.png", bbox_inches='tight', dpi=100, transparent=True)
    fig.savefig("dwExample.pdf", bbox_inches='tight', transparent=True)

    
def plotNL():
    
    fig = plt.figure(figsize=figsize, dpi=300)
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams['axes.labelsize'] = 20
    rc('text', usetex=False)
    matplotlib.rcParams['xtick.labelsize'] = 14
    matplotlib.rcParams['ytick.labelsize'] = 14


    ax0 = fig.add_subplot(211)  
    ax1 = fig.add_subplot(212)  
    
    

    ax0.plot(ks, pkdw, 'r-', linewidth=1)
    ax0.plot(ks, pknl, 'g--', linewidth=2)

    #mark_inset(ax0, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    
    ax1.plot(ss, ss * ss * datapointsDW, 'r-', linewidth=1, label="$P_{\mathrm{dw}}$")
    ax1.plot(ss, ss * ss * datapointsNL, 'g--', linewidth=2, label="$P_{\mathrm{nl}}$")

    ax1.legend(frameon=False)


    ax1.set_xlabel('$s \ \ [\mathrm{Mpc/h}]$')
    ax1.set_ylabel(r'$s^2 \xi(s)$')

    ax1.set_xlim(5,160)
    ax0.set_xlim(0.001, 10)    
    ax0.set_ylim(1, 70000)
    
    ax0.set_xscale('log')
    ax0.set_yscale('log')
    #axins.set_yscale('log')
    ax0.set_xlabel('$\mathrm{k/h}\ \  [\mathrm{Mpc}^{-1}]$')
    ax0.set_ylabel('$\mathrm{Power}$')
    plt.tight_layout()
    
    fig.savefig("nlExample.png", bbox_inches='tight', dpi=100, transparent=True)
    fig.savefig("nlExample.pdf", bbox_inches='tight', transparent=True)    
    
    

def plotMono():
    fig = plt.figure(figsize=figsize, dpi=300)
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams['axes.labelsize'] = 20
    rc('text', usetex=False)
    matplotlib.rcParams['xtick.labelsize'] = 14
    matplotlib.rcParams['ytick.labelsize'] = 14

    ax = fig.add_subplot(111)  
    ax.set_title("$P_{\mathrm{mp}}(k)$", fontsize=24, y=1.015)
    axins = zoomed_inset_axes(ax, 2.5, loc=3)
    axins.set_xlim(0.03,0.4)
    axins.set_ylim(900, 3.5e4)
    plt.setp(axins.get_xticklabels(), visible=False)
    plt.setp(axins.get_yticklabels(), visible=False)



    ax.plot(ks, pklin, linewidth=2, alpha=0.1)
    ax.plot(ks, pkdw, linewidth=2, alpha=0.1)
    axins.plot(ks, pklin, linewidth=2, alpha=0.1)
    axins.plot(ks, pkdw, linewidth=2, alpha=0.1)
    ax.plot(ks, pknl, linewidth=2, alpha=0.2)
    axins.plot(ks, pknl, linewidth=2, alpha=0.2)
    ax.plot(ks, monopole, linewidth=2)
    axins.plot(ks, monopole, linewidth=2)

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax.set_xscale('log')
    ax.set_yscale('log')
    axins.set_xscale('log')
    axins.set_yscale('log')
    ax.set_xlabel('k/h')
    ax.set_ylabel('Power')

    #fig.savefig("mp.png", bbox_inches='tight', dpi=300, transparent=True)

def plotSS():
    fig = plt.figure(figsize=figsize, dpi=300)
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams['axes.labelsize'] = 20
    rc('text', usetex=False)
    matplotlib.rcParams['xtick.labelsize'] = 14
    matplotlib.rcParams['ytick.labelsize'] = 14

    ax = fig.add_subplot(111)  
    ax.set_title(r"$\xi_{\mathrm{mp}}(s)$", fontsize=24, y=1.015)



    ax.plot(ss, ss * ss * datapointsLin, linewidth=2, alpha=0.1)
    ax.plot(ss, ss * ss * datapointsDW, linewidth=2, alpha=0.1)
    ax.plot(ss, ss * ss * datapointsNL, linewidth=2, alpha=0.1)
    ax.plot(ss, ss * ss * datapoints, linewidth=2)




    ax.set_xlabel('$s$')
    ax.set_ylabel(r'$s^2 \xi(s)$')

    #fig.savefig("ss.png", bbox_inches='tight', dpi=300, transparent=True)



if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from mcmc import *
    import methods
    
    omch2 = 0.2

    if True:
        import methods
        generator = methods.SlowGenerator(debug=True)
        (ks, pklin, pkratio) = generator.getOmch2AndZ(omch2, 0.6)    

        fb = 0.1666
        ombh2 = fb * omch2
        h = 0.7
        oc = omch2 / h / h
        ns = 0.96
        sig8 = 0.8
        zz = 0.6
        om = oc / (1 - fb)
        sigmav = 8
        b2 = 0.8
        beta = 1
        loren = 3
        
        mu = np.linspace(0,1,200)
        mu2 = np.power(mu, 2)
        ss = np.linspace(1, 150, 300)
        
        
        pknw = methods.dewiggle(ks, pklin)    
        weights = methods.getLinearNoWiggleWeightSigmaV(ks, sigmav)
        pkdw = pklin * weights + pknw * (1 - weights)
        pknl = pkdw * pkratio
        mpknl = b2 * pknl
        ar = mpknl[np.newaxis].T.dot(np.power((1 + beta * mu2[np.newaxis]), 2)) # square mu, and then square entire brackets
        ksmu = ks[np.newaxis].T.dot(mu[np.newaxis])
        lor = (1 + np.power((loren * ksmu), 2))
        ar /= lor
        monopole = simps(ar, mu) * 1
        datapoints = methods.pk2xiGauss(ks, monopole, ss) 
        datapointsLin = methods.pk2xiGauss(ks, pklin, ss) 
        datapointsNW = methods.pk2xiGauss(ks, pknw, ss) 
        datapointsDW = methods.pk2xiGauss(ks, pkdw, ss) 
        datapointsNL = methods.pk2xiGauss(ks, pknl, ss) 
    
    #plotLinear()
    #plotNW()
    plotNL()
    #plotMono()
    #plotSS()
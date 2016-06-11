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
import wizcola
import wigglezold
import wiggleznew
from mcmc import *
import methods

class Fitted(object):
    def __init__(self, debug=False):
        self._debug = debug
    def getIndex(self, param):
        for i, x in enumerate(self.getParams()):
            if x[0] == param:
                return i + 3
        return None
    def debug(self, string):
        if self._debug:
            print(string)
            sys.stdout.flush()
    def getNumParams(self):
        raise NotImplementedError
    def getChi2(self, params):
        raise NotImplementedError
    def getParams(self):
        raise NotImplementedError


class CovarianceFitter(Fitted):
    def __init__(self, debug=False, cambFast=False, pk2xiFast=False):
        self._debug = debug
        self.params = [('omch2', 0.05, 2, '$\\Omega_c h^2$'),('alpha', 0.7, 1.3, r'$\alpha$'),('epsilon', -0.4, 0.4, r'$\epsilon$') ]
    def getParams(self):
        return self.params
    def getNumParams(self):
        return len(self.params)
    def setCovariance(self, cov):
        self.cov = cov
        self.invCov = np.linalg.inv(cov)
    def setData(self, data):
        self.data = data
        
    def getChi2(self, params):
        allParams = self.getParams()
        for i, p in enumerate(params):
            p = np.round(p, 5)
            if p <= allParams[i][1] or p >= allParams[i][2]:
                self.debug("Outside %s: %0.2f" % (allParams[i][0], p))
                return None
        model = np.array([[a]*3 for a in params]).flatten()
        chi2 = np.dot((self.data - model).T, np.dot(self.invCov, self.data - model))

        return chi2
        
        
class CosmoMonopoleFitter(Fitted):
    def __init__(self, debug=False, cambFast=False, pk2xiFast=False):
        self._debug = debug
        self.cambFast = cambFast
        self.pk2xiFast = pk2xiFast
        self.cambDefaults = {}
        self.cambDefaults['get_scalar_cls'] = False
        if cambFast:
            self.cambDefaults['do_lensing'] = False
            self.cambDefaults['accurate_polarization'] = False
            self.cambDefaults['high_accuracy_default'] = False
            
        self.cambParams = []
        self.fitParams = []
        
        self.ss = np.linspace(10, 235, 100)
        self.generator = None
        
    def addCambParameter(self, name, minValue, maxValue, label):
        self.cambParams.append((name, minValue, maxValue, label))
        
    def getNumParams(self):
        return len(self.cambParams) + len(self.fitParams)
        
    def addDefaultCambParams(self, **args):
        self.cambDefaults.update(args)
        
    def getParams(self):
        return self.cambParams + self.fitParams
    def generateMus(self, n=100):
        self.mu = np.linspace(0,1,n)
        self.mu2 = np.power(self.mu, 2)
        self.muNA = self.mu[np.newaxis]
        self.mu2NA = self.mu2[np.newaxis]
        self.p2 = 0.5 * (3 * self.mu2 - 1)
        self.p4 = (35*self.mu2*self.mu2 - 30*self.mu2 + 3) / 8.0
        
    def setData(self, datax, monopole, quadrupole, totalCovariance, zs, poles=True, matchQuad=True, minS=25, maxS=180, angular=True, sigmav=5.0, fast=True):
        selection = (datax > minS) & (datax < maxS)
        selection2 = np.concatenate((selection, selection))
        self.rawX = datax
        self.dataZs = zs
        self.dataX = datax[selection]
        self.monopole = monopole[selection]
        if quadrupole is not None:
            self.quadrupole = quadrupole[selection]
        self.rawMonopole = monopole
        self.rawQuadrupole = quadrupole
        self.matchQuad = matchQuad
        self.poles = poles
        self.angular = angular
        if sigmav is not None:
            self.sigmav = sigmav
        else:
            self.sigmav = None
            self.fitParams.append(("sigmav", 0.1, 10, r"$\sigma_v$"))
        self.fast = fast
        for i,z in enumerate(zs):
            self.fitParams.append(('b2%d'%i, 0.2, 2, '$b^2$'))

        if angular:
            self.fitParams.append(('beta', 0.1, 2, '$\\beta$'))
            self.fitParams.append(('lorentzian', 0.1, 10.0, '$\\sigma H(z)$'))
        if poles:
            self.generateMus()
            self.fitParams.append(('alpha', 0.7, 1.3, r'$\alpha$'))
            if matchQuad:
                self.fitParams.append(('epsilon', -0.4, 0.4, r'$\epsilon$'))
        else:
            self.generateMus(n=50)
            self.fitParams.append(('alphaPerp', 0.7, 1.3, r'$\alpha_\perp$'))
            self.fitParams.append(('alphaParallel', 0.7, 1.3, r'$\alpha_\parallel$'))
        
        if poles and not matchQuad:
            self.totalData = self.monopole
            self.dataCov = totalCovariance[:,selection][selection,:] #(monopoleCovariance[:,:,1])[:,selection][selection,:]
        else:
            self.totalData = np.concatenate((self.monopole, self.quadrupole))
            self.dataCov = totalCovariance[:,selection2][selection2,:]
            
        icov = np.sqrt(np.diag(np.linalg.inv(totalCovariance)))
        if matchQuad:
            self.rawE = icov[:icov.size/2] #np.sqrt(monopoleCovariance[np.arange(monopoleCovariance.shape[0]), np.arange(monopoleCovariance.shape[1]), 0])
            self.dataE = self.rawE[selection]
        else:
            self.rawE = icov
            self.dataE = self.rawE[selection]
            
        if quadrupole is not None:
            self.rawQE = icov[icov.size/2:] #np.sqrt(quadrupoleCovariance[np.arange(monopoleCovariance.shape[0]), np.arange(quadrupoleCovariance.shape[1]), 0])
            self.dataQE = self.rawQE[selection] #self.rawQE[selection]
        else:
            self.quadrupole = None
            self.rawQE = None
            self.dataQE = None
        
        
        
            
        
    def getData(self):
        return (self.dataX, self.monopole, self.dataE, self.quadrupole, self.dataQE, self.dataZ)
        
    def getRawData(self):
        return (self.rawX, self.rawMonopole, self.rawE, self.rawQuadrupole, self.rawQE, self.dataZ)
        
    
    def getModel(self, params, modelss):
        modelss = modelss[:modelss.size/len(self.dataZs)]
        allParams = self.getParams()
        for i, p in enumerate(params):
            p = np.round(p, 5)
            if p <= allParams[i][1] or p >= allParams[i][2]:
                self.debug("Outside %s: %0.2f" % (allParams[i][0], p))
                return None
        #cambParams = {k[0]: np.round(params[i],5) for (i,k) in enumerate(self.cambParams)}
        #fitDict = {k[0]:params[i + len(cambParams)]  for (i,k) in enumerate(self.fitParams)}
        cambParams = dict((k[0], np.round(params[i],5)) for (i,k) in enumerate(self.cambParams))
        #cambParams = {k[0]: np.round(params[i],5) for (i,k) in enumerate(self.cambParams)}
        fitDict = dict((k[0], params[i + len(cambParams)])  for (i,k) in enumerate(self.fitParams))
        #fitDict = {k[0]:params[i + len(cambParams)]  for (i,k) in enumerate(self.fitParams)}
        omch2 = cambParams.get('omch2')    
        
        
        betas = [fitDict['beta'] for i in range(len(self.dataZs))]
        lorens = [fitDict['lorentzian'] for i in range(len(self.dataZs))]
        sigmavs = [self.sigmav if self.sigmav is not None else fitDict['sigmav'] for i in range(len(self.dataZs))]
        b2s = [fitDict['b2%d'%i] for i in range(len(self.dataZs))]
        
        firsts = [] #mp / trans
        seconds = [] #qp / long
        
        for b2, beta, loren, sigmav, z in zip(b2s, betas, lorens, sigmavs, self.dataZs):         
        
            
            if self.generator is None:
                self.generator = methods.SlowGenerator(debug=True)
            (ks, pklin, pkratio) = self.generator.getOmch2AndZ(omch2, z)
            
            pknw = methods.dewiggle(ks, pklin)
    
            
            weights = methods.getLinearNoWiggleWeightSigmaV(ks, sigmav)
            pkdw = pklin * weights + pknw * (1 - weights)
            
            pknl = pkdw * pkratio
            mpknl = b2 * pknl
            
            if self.angular:
                
                ksmu = ks[np.newaxis].T.dot(self.muNA )
                ar = mpknl[np.newaxis].T.dot(np.power((1 + beta * self.mu2NA), 2)) / (1 + (loren * loren * ksmu * ksmu))
            else:
                ar = mpknl
            
            s0 = 0.32
            gamma = -1.36
            
            
            if self.poles:
                alpha = fitDict['alpha']
                if self.matchQuad:
                    epsilon = fitDict['epsilon']
                else:
                    epsilon = 0
                if self.angular:
                    monopole = simps(ar, self.mu)
                else:
                    monopole = mpknl
                quadrupole = simps(ar * self.p2 * 5.0, self.mu)       
                hexadecapole = simps(ar * self.p4 * 9.0, self.mu)
                d = 5  
                
                ximpa = methods.pk2xiGauss(ks, monopole, modelss * alpha, interpolateDetail=d)
                xiqpa = methods.pk2xiGaussQuad(ks, quadrupole, modelss * alpha, interpolateDetail=d)
                if not self.fast:
                    xihpa = methods.pk2xiGaussHex(ks, hexadecapole, modelss * alpha, interpolateDetail=d)
    
                
                ds = 0.5
                dss = np.array([i+j for i in modelss for j in [-ds, ds]])
                dlogs = np.diff(np.log(dss))[::2]
                dxi0as = np.diff(methods.pk2xiGauss(ks, monopole, dss*alpha, interpolateDetail=d))[::2]
                dxi2as = np.diff(methods.pk2xiGaussQuad(ks, quadrupole, dss*alpha, interpolateDetail=d))[::2]
                if not self.fast:
                    dxi4as = np.diff(methods.pk2xiGaussHex(ks, hexadecapole, dss*alpha, interpolateDetail=d))[::2]
                    dxi4asdlogs = dxi4as / dlogs
                dxi0asdlogs = dxi0as / dlogs
                dxi2asdlogs = dxi2as / dlogs
               
    
                datapointsM = ximpa + 0.4 * epsilon * (3 * xiqpa + dxi2asdlogs)
                datapointsQ = 2 * epsilon * dxi0asdlogs  + (1 + (6.0 * epsilon / 7.0)) * xiqpa
                if not self.fast:
                    datapointsQ += (4.0 * epsilon / 7.0) * ( dxi2asdlogs   + 5 * xihpa + dxi4asdlogs )
    
                growth = 1 + np.power(((modelss * alpha)/s0), gamma)
                datapointsM = datapointsM * growth
                datapointsQ = datapointsQ * growth
    
                firsts.append(datapointsM)
    
                if self.matchQuad:        
                    seconds.append(datapointsQ)
                    
            else:
                alphaPerp = fitDict['alphaPerp']
                alphaParallel = fitDict['alphaParallel']
    
                monopole = simps(ar, self.mu)
                quadrupole = simps(ar * self.p2 * 5, self.mu)
                hexapole = simps(ar * self.p4 * 9, self.mu)
                
                datapointsM = methods.pk2xiGauss(ks, monopole, self.ss) 
                datapointsQ = methods.pk2xiGaussQuad(ks, quadrupole, self.ss)
                datapointsH = methods.pk2xiGaussHex(ks, hexapole, self.ss)
                
    
                growth = 1 + np.power((self.ss/s0), gamma)
    
                datapointsM = datapointsM * growth
                datapointsQ = datapointsQ * growth
                datapointsH = datapointsH * growth
                
                sprime = modelss
                
                
                xi2dm = np.ones(self.mu.size)[np.newaxis].T.dot(datapointsM[np.newaxis])
                xi2dq = self.p2[np.newaxis].T.dot(datapointsQ[np.newaxis])
                xi2dh = self.p4[np.newaxis].T.dot(datapointsH[np.newaxis])
                xi2d = xi2dm + xi2dq + xi2dh
    
                mugrid = self.muNA.T.dot(np.ones((1, datapointsM.size)))
                ssgrid = self.ss[np.newaxis].T.dot(np.ones((1, self.mu.size))).T
                
                flatmu = mugrid.flatten()
                flatss = ssgrid.flatten()
                flatxi2d = xi2d.flatten()
                
                
                sqrtt = np.sqrt(alphaParallel * alphaParallel * self.mu2 + alphaPerp * alphaPerp * (1 - self.mu2))
                mus = alphaParallel * self.mu / sqrtt
                mu1 = self.mu[:self.mu.size/2]
                mu2 = self.mu[self.mu.size/2:]
                xiT = []
                xiL = []
                svals = np.array([])
                mvals = np.array([])
                for sp in sprime:
                    svals = np.concatenate((svals, sp * sqrtt))
                    mvals = np.concatenate((mvals, mus))
                
                xis = scipy.interpolate.griddata((flatmu, flatss), flatxi2d, (mvals, svals))
                for i, sp in enumerate(sprime):
                    sz = sqrtt.size/2
                    ii = 2 * i
                    xis1 = xis[ii*sz : (ii+1)*sz]
                    xis2 = xis[(ii+1)*sz : (ii+2)*sz]
                    xiT.append(2 * simps(xis1, mu1))
                    xiL.append(2 * simps(xis2, mu2))
                firsts.append(np.array(xiT))
                seconds.append(np.array(xiL))
                    
        return np.concatenate((np.array(firsts).flatten(), np.array(seconds).flatten()))

    def getChi2(self, params):
        datapoints = self.getModel(params, self.dataX)
        if datapoints is None:
            return None
        chi2 = np.dot((self.totalData - datapoints).T, np.dot(self.dataCov, self.totalData - datapoints))
        return chi2

class WizcolaWedgeAllMean(CosmoMonopoleFitter):
    def __init__(self, debug=True):
        super(self.__class__, self).__init__(debug=debug)
        bins = [0,1,2]
        wizMP = wizcola.WizColaLoader.getWedges()
        datas = [wizMP.getAllTransverse(b) for b in bins] + [wizMP.getAllLongitudinal(b) for b in bins]
        datays = np.concatenate(tuple([d[:, 1:] for d in datas]))
        cov = np.cov(datays, bias=1)
        covinv = np.linalg.inv(cov)
        dataymeans = np.mean(datays, axis=1)
        dataxs = np.concatenate(tuple([d[:, 0] for d in datas]))
        zs = [wizMP.getZ(bin) for bin in bins]
        datax = dataxs[:dataxs.size/2]
        monopole = dataymeans[:dataymeans.size/2]
        quadrupole = dataymeans[dataymeans.size/2:]
        self.addCambParameter('omch2', 0.05, 0.25, '$\\Omega_c h^2$')
        print("Wizcola wedge mean all bins")
        self.setData(datax, monopole, quadrupole, covinv, zs, poles=False)
     
class WizcolaMultipoleAllMean(CosmoMonopoleFitter):
    def __init__(self, debug=True):
        super(self.__class__, self).__init__(debug=debug)
        bins = [0,1,2]
        wizMP = wizcola.WizColaLoader.getMultipoles()
        datas = [wizMP.getAllMonopoles(b) for b in bins] + [wizMP.getAllQuadrupoles(b) for b in bins]
        dataxs = np.concatenate(tuple([d[:, 0] for d in datas]))
        datays = np.concatenate(tuple([d[:, 1:] for d in datas]))
        dataymeans = np.mean(datays, axis=1)
        cov = np.cov(datays, bias=1)
        covinv = np.linalg.inv(cov)
        zs = [wizMP.getZ(bin) for bin in bins]
        datax = dataxs[:dataxs.size/2]
        monopole = dataymeans[:dataymeans.size/2]
        quadrupole = dataymeans[dataymeans.size/2:]
        self.addCambParameter('omch2', 0.05, 0.25, '$\\Omega_c h^2$')
        print("Wizcola multipole mean all bins")
        self.setData(datax, monopole, quadrupole, covinv, zs)
        
class WigglezMultipoleAll(CosmoMonopoleFitter):
    def __init__(self, debug=True):
        super(self.__class__, self).__init__(debug=debug)
        bins = [0,1,2]
        wizMP = wizcola.WizColaLoader.getMultipoles()
        datas = [wizMP.getAllMonopoles(b) for b in bins] + [wizMP.getAllQuadrupoles(b) for b in bins]
        datays = np.concatenate(tuple([d[:, 1:] for d in datas]))
        cov = np.cov(datays, bias=1)
        covinv = np.linalg.inv(cov)
        wiz = wiggleznew.WigglezLoader.getMultipoles()
        monopoles = np.concatenate(tuple([wiz.getMonopoles(bin) for bin in bins]))
        quadrupoles = np.concatenate(tuple([wiz.getQuadrupoles(bin) for bin in bins]))
        zs = [wiz.getZ(bin) for bin in bins]
        datax = monopoles[:, 0]
        monopole = monopoles[:, 1]
        quadrupole = quadrupoles[:, 1]
        self.addCambParameter('omch2', 0.05, 0.25, '$\\Omega_c h^2$')
        print("Wigglez unreconstructed multipole all bins")
        self.setData(datax, monopole, quadrupole, covinv, zs)

class WigglezWedgeAll(CosmoMonopoleFitter):
    def __init__(self, debug=True):
        super(self.__class__, self).__init__(debug=debug)
        bins = [0,1,2]
        wizMP = wizcola.WizColaLoader.getWedges()
        datas = [wizMP.getAllTransverse(b) for b in bins] + [wizMP.getAllLongitudinal(b) for b in bins]
        datays = np.concatenate(tuple([d[:, 1:] for d in datas]))
        cov = np.cov(datays, bias=1)
        covinv = np.linalg.inv(cov)
        wiz = wiggleznew.WigglezLoader.getWedges()
        monopoles = np.concatenate(tuple([wiz.getTransverse(bin) for bin in bins]))
        quadrupoles = np.concatenate(tuple([wiz.getLongitudinal(bin) for bin in bins]))
        zs = [wiz.getZ(bin) for bin in bins]
        datax = monopoles[:, 0]
        monopole = monopoles[:, 1]
        quadrupole = quadrupoles[:, 1]       
        self.addCambParameter('omch2', 0.05, 0.25, '$\\Omega_c h^2$')
        print("Wigglez unreconstructed wedge all bins")
        self.setData(datax, monopole, quadrupole, covinv, zs, poles=False)
     
class WigglezWedgeBin(CosmoMonopoleFitter):
    def __init__(self, bin, debug=True):
        super(self.__class__, self).__init__(debug=debug)
        wiz = wiggleznew.WigglezLoader.getWedges()
        monopoles = wiz.getTransverse(bin)
        quadrupoles = wiz.getLongitudinal(bin)
        zs = [wiz.getZ(bin)]
        datax = monopoles[:, 0]
        monopole = monopoles[:, 1]
        quadrupole = quadrupoles[:, 1]
        cor = wiz.getCovariance(bin)[:,:,3]
        self.addCambParameter('omch2', 0.05, 0.25, '$\\Omega_c h^2$')
        print("Wigglez unreconstructed wedge bin %d" % (bin))
        self.setData(datax, monopole, quadrupole, cor, zs, poles=False)
        
class WigglezMultipoleBin(CosmoMonopoleFitter):
    def __init__(self, bin, debug=True):
        super(self.__class__, self).__init__(debug=debug)
        wiz = wiggleznew.WigglezLoader.getMultipoles()
        monopoles = wiz.getMonopoles(bin)
        quadrupoles = wiz.getQuadrupoles(bin)
        zs = [wiz.getZ(bin)]
        datax = monopoles[:, 0]
        monopole = monopoles[:, 1]
        quadrupole = quadrupoles[:, 1]
        cor = wiz.getCovariance(bin)[:,:,3]
        self.addCambParameter('omch2', 0.05, 0.25, '$\\Omega_c h^2$')
        print("Wigglez unreconstructed multipole bin %d" % (bin))
        self.setData(datax, monopole, quadrupole, cor, zs)

class WizcolaMultipoleBin(CosmoMonopoleFitter):
    def __init__(self, bin, realisation, debug=True):
        super(self.__class__, self).__init__(debug=debug)
        wiz = wizcola.WizColaLoader.getMultipoles()
        monopoles = wiz.getAllMonopoles(bin)
        quadrupoles = wiz.getAllQuadrupoles(bin)
        zs = [wiz.getZ(bin)]
        datax = monopoles[:, 0]
        monopole = monopoles[:, 1+realisation]
        quadrupole = quadrupoles[:, 1+realisation]
        cor = wiz.getCovariance(bin)[:,:,3]
        self.addCambParameter('omch2', 0.05, 0.25, '$\\Omega_c h^2$')
        print("Wizcola multipole for bin %d and realisation %d" % (bin, realisation))
        self.setData(datax, monopole, quadrupole, cor, zs)
        
class WizcolaWedgeBin(CosmoMonopoleFitter):
    def __init__(self, bin, realisation, debug=True):
        super(self.__class__, self).__init__(debug=debug)
        wiz = wizcola.WizColaLoader.getWedges()
        monopoles = wiz.getAllTransverse(bin)
        quadrupoles = wiz.getAllLongitudinal(bin)
        zs = [wiz.getZ(bin)]
        datax = monopoles[:, 0]
        monopole = monopoles[:, 1+realisation]
        quadrupole = quadrupoles[:, 1+realisation]
        cor = wiz.getCovariance(bin)[:,:,3]
        self.addCambParameter('omch2', 0.05, 0.25, '$\\Omega_c h^2$')
        print("Wizcola wedge for bin %d and realisation %d" % (bin, realisation))
        self.setData(datax, monopole, quadrupole, cor, zs, poles=False)
        
class WizcolaMultipoleBinMean(CosmoMonopoleFitter):
    def __init__(self, bin, debug=True):
        super(self.__class__, self).__init__(debug=debug)
        wiz = wizcola.WizColaLoader.getMultipoles()
        monopoles = wiz.getAllMonopoles(bin)
        quadrupoles = wiz.getAllQuadrupoles(bin)
        zs = [wiz.getZ(bin)]
        datax = monopoles[:, 0]
        monopole = np.mean(monopoles[:, 1:], axis=1)
        quadrupole = np.mean(quadrupoles[:, 1:], axis=1)
        cor = wiz.getCovariance(bin)[:,:,3]
        self.addCambParameter('omch2', 0.05, 0.25, '$\\Omega_c h^2$')
        print("Wizcola multipole mean for bin %d" % (bin))
        self.setData(datax, monopole, quadrupole, cor, zs)
        
class WizcolaWedgeBinMean(CosmoMonopoleFitter):
    def __init__(self, bin, debug=True):
        super(self.__class__, self).__init__(debug=debug)
        wiz = wizcola.WizColaLoader.getWedges()
        monopoles = wiz.getAllTransverse(bin)
        quadrupoles = wiz.getAllLongitudinal(bin)
        zs = [wiz.getZ(bin)]
        datax = monopoles[:, 0]
        monopole = np.mean(monopoles[:, 1:], axis=1)
        quadrupole = np.mean(quadrupoles[:, 1:], axis=1)
        cor = wiz.getCovariance(bin)[:,:,3]
        self.addCambParameter('omch2', 0.05, 0.25, '$\\Omega_c h^2$')
        print("Wizcola wedge mean for bin %d" % (bin))
        self.setData(datax, monopole, quadrupole, cor, zs, poles=False)

        
class WigglezNewMonopoleFitter(CosmoMonopoleFitter):
    def __init__(self, cambFast=False, pk2xiFast=False, debug=True, bin=0, minS=10, maxS=180, angular=True, log=False):
        super(self.__class__, self).__init__(cambFast=cambFast, pk2xiFast=pk2xiFast, debug=debug)
        
        wiz = wiggleznew.WigglezLoader.getMultipoles()
        monopoles = wiz.getMonopoles(bin)
        quadrupoles = None
        zs = [wiz.getZ(bin)]
        datax = monopoles[:, 0]
        monopole = monopoles[:, 1]
        quadrupole = None
        cor = wiz.getMonopoleCovariance(bin)[:,:,3]
        self.addCambParameter('omch2', 0.05, 0.25, '$\\Omega_c h^2$')
        
        self.setData(datax, monopole, None, cor, zs, minS=minS, maxS=maxS, matchQuad=False, angular=angular)

class WigglezNewMonopoleFitterFreeSigmav(CosmoMonopoleFitter):
    def __init__(self, cambFast=False, pk2xiFast=False, debug=True, bin=0, minS=10, maxS=180, angular=True, log=False):
        super(self.__class__, self).__init__(cambFast=cambFast, pk2xiFast=pk2xiFast, debug=debug)
        
        wiz = wiggleznew.WigglezLoader.getMultipoles()
        monopoles = wiz.getMonopoles(bin)
        quadrupoles = None
        zs = [wiz.getZ(bin)]
        datax = monopoles[:, 0]
        monopole = monopoles[:, 1]
        quadrupole = None
        cor = wiz.getMonopoleCovariance(bin)[:,:,3]
        self.addCambParameter('omch2', 0.05, 0.25, '$\\Omega_c h^2$')
        
        self.setData(datax, monopole, None, cor, zs, minS=minS, maxS=maxS, matchQuad=False, angular=angular, sigmav=None)        
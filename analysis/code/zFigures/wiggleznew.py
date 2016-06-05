import numpy as np
import os
import re

class WigglezLoader(object):
    @staticmethod
    def getMultipoles():
        return WigglezMultipoles()
        
    @staticmethod
    def getWedges():
        return WigglezWedges()

class Wigglez(object):
    def __init__(self, path):
        thisDir, thisFilename = os.path.split(__file__)
        self.path = thisDir + os.sep + path
        self.z1Pattern = re.compile('_z([0-9]+pt[0-9]+)_')
        self.z2Pattern = re.compile('_z.*_([0-9]+pt[0-9]+)\.')
        self.zs = [0.44, 0.6, 0.73]
        
        self.files = []
        self.bins = []
        self.covs = []
        self.binz1 = []
        self.binz2 = []
                
        for f in self.getDatFilesInDir(self.path):
            print(f)
            self.loadFile(self.path, f)
        
        for f in self.getDatFilesInDir(self.path + os.sep + 'cov'):
            print(f)
            self.loadCovariance(self.path + os.sep + 'cov', f)
        
    def loadCovariance(self, path, filename):
        a = np.loadtxt(path + os.sep + filename)
        self.covs.append(a.reshape((60,60,5)))
        
    def getCorner(self, bin, corner=0):
        a = self.covs[bin]
        tmp = a[a[:,:,4] == corner]
        if corner == 1:
            tmp = tmp[:tmp[:,0].size / 2, :]
        return tmp[:,:4].reshape((30,30,4))

    def getCovariance(self, bin):
        return self.covs[bin]
        
    def getCrossCovariance(self, bin):
        return self.getCorner(bin, corner=1)
        
    def getZ(self, bin):
        return self.zs[bin]
        
    def getZ1FromName(self, filename):
        z1String = self.z1Pattern.search(filename)
        if z1String:
            return float(z1String.group(1).replace('pt', '.'))
        else:
            return None
                
    def getZ2FromName(self, filename):
        z2String = self.z2Pattern.search(filename)
        if z2String:
            return float(z2String.group(1).replace('pt', '.'))
        else:
            return None
         
    def getDatFilesInDir(self, dir):
        return sorted([i for i in os.listdir(dir) if i.endswith('.dat')])
    
    def loadFile(self, path, filename):
        a = np.loadtxt(path + os.sep + filename)
        self.files.append(filename)
        self.bins.append(a)
        self.binz1.append(self.getZ1FromName(filename))
        self.binz2.append(self.getZ2FromName(filename))
        

class WigglezMultipoles(Wigglez):
    def __init__(self):
        super(WigglezMultipoles, self).__init__('wiggleznew' + os.sep + 'multipoles')
        
    def getMonopoles(self, bin, recon=False):
        i = 0 if not recon else 4
        return self.bins[bin][:, [2,3+i]]
    
        
    def getQuadrupoles(self, bin, recon=False):
        i = 2 if not recon else 6
        return self.bins[bin][:, [2,3+i]]
		
    def getMonopoleCovariance(self, bin):
        return self.getCorner(bin, 0)
        
    def getQuadrupoleCovariance(self, bin):
        return self.getCorner(bin, 2)
    
class WigglezWedges(Wigglez):
    def __init__(self):
        super(WigglezWedges, self).__init__('wiggleznew' + os.sep + 'wedges')
    
    def getTransverse(self, bin, recon=False):
        i = 0 if not recon else 4
        return self.bins[bin][:, [2,3+i]]
        
    def getLongitudinal(self, bin, recon=False):
        i = 2 if not recon else 6
        return self.bins[bin][:, [2,3+i]]

    def getTransverseCovariance(self, bin):
        return self.getCorner(bin, 0)
        
    def getLongitudinalCovariance(self, bin):
        return self.getCorner(bin, 2)
        
    
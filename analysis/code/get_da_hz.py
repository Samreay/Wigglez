import numpy as np
import os
from chainconsumer import ChainConsumer
from scipy.integrate import simps, cumtrapz
from scipy.interpolate import interp1d

def get_r_s(oms):
    # Fiducial
    h = 0.705
    ob = 0.0456
    ol = 0.727
    og = 2.469e-5 / (h * h)
    orr = og * (1 + (7.0/8.0)*np.power(4.0/11.0, 4.0/3.0)*3.04)
    
    c = 3e5
    z = 1020
    ae = 1.0 / (1.0 + z)
    a = np.linspace(ae/100000, ae, 100)
    
    rs = []
    for om in oms:
        ez = np.sqrt(om / (a*a*a) + ol + orr/(a*a*a*a))
        hs = h * ez
        fac = 1/np.sqrt(3 * (1 + (3 * ob / (4 * og))*a))
        integrand = (c/100) * fac / (a*a*hs)
        integral = simps(integrand, x=a)
        rs.append(integral)
    return rs
    
def load_directory(directory):
    
    num_burnin = 20000
    res = None
    for f in os.listdir(directory):
        filename = directory + os.sep + f
        if not os.path.isfile(filename):
            continue
        matrix = np.load(filename)[num_burnin:, [3,-2,-1]]
        if res is None:
            res = matrix
        else:
            res = np.vstack((res, matrix))
    return res
    
def convert_directory(directory, z):
    data = load_directory(directory)
    omch2 = data[:,0]
    alpha = data[:,1]
    epsilon = data[:,2]
    
    # Fiducial
    h = 0.705
    om = 0.273
    ob = 0.0456
    ol = 0.727
    H = 100
    c = 3e5
    #omall = np.linspace(0.05, 0.3, 1000)
    #rsall = get_r_s(omall)
    
    #oms = omch2/h/h + ob
    #rss = interp1d(omall, rsall)(oms)
    
    zs = np.linspace(0,z,1000)
    ez = np.sqrt(om * (1 + zs)**3 + ol)
    
    iez = 1 / ez
    
    chi = simps(iez, zs)
    da = (c/(H * h)) * chi / (1 + z)
    hs = h * H * ez[-1]
    
    rs_fid = get_r_s([0.273])[0]
    
    daval = (alpha/(1+epsilon)) * da / rs_fid
    
    hrc = hs * rs_fid / (alpha * (1 + epsilon) * (1 + epsilon)) / c
    res = np.vstack((omch2, daval, z/hrc)).T
    return res
    
p1 = [r"$\Omega_c h^2$", r"$\alpha$", r"$\epsilon$"]
p2 = [r"$\Omega_c h^2$", r"$D_A(z)/r_s$", r"$cz/H(z)/r_s $"]


if False:
    consumer = ChainConsumer()
    consumer.configure_contour(sigmas=[0,1.3])
    consumer.add_chain(load_directory("../bWigMpBin/bWigMpBin_z0"), parameters=p1, name="$0.2<z<0.6$")
    consumer.add_chain(load_directory("../bWigMpBin/bWigMpBin_z1"), parameters=p1, name="$0.4<z<0.8$")
    consumer.add_chain(load_directory("../bWigMpBin/bWigMpBin_z2"), parameters=p1, name="$0.6<z<1.0$")
    consumer.plot(figsize="column", filename="wigglez_multipole_alphaepsilon.pdf", truth=[0.113, 1.0, 0.0])
    print(consumer.get_latex_table())

if True:
    c = ChainConsumer()
    c.configure_contour(sigmas=[0,1,2])
    c.add_chain(convert_directory("../bWigMpBin/bWigMpBin_z0", 0.44), parameters=p2, name="$0.2<z<0.6$")
    c.add_chain(convert_directory("../bWigMpBin/bWigMpBin_z1", 0.60), parameters=p2, name="$0.4<z<0.8$")
    c.add_chain(convert_directory("../bWigMpBin/bWigMpBin_z2", 0.73), parameters=p2, name="$0.6<z<1.0$")
    print(c.get_latex_table())
    #c.plot(figsize="column", filename="wigglez_multipole_dah.pdf")

if False:
    c = ChainConsumer()
    c.configure_contour(sigmas=[0,1,2])
    c.add_chain(convert_directory("../bWizMpMeanBin/bWizMpMeanBin_z0", 0.44), parameters=p2, name="$0.2<z<0.6$")
    c.add_chain(convert_directory("../bWizMpMeanBin/bWizMpMeanBin_z1", 0.60), parameters=p2, name="$0.4<z<0.8$")
    c.add_chain(convert_directory("../bWizMpMeanBin/bWizMpMeanBin_z2", 0.73), parameters=p2, name="$0.6<z<1.0$")
    print(c.get_latex_table())
    c.plot(figsize="column", filename="wizcola_multipole_dah.pdf")
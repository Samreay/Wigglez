import numpy as np

cor = np.loadtxt("cor.txt")
sigma = np.array([0.03, 0.03, 0.03, 0.1, 0.1, 0.08, 0.08, 0.08, 0.06])
ssigma = np.sqrt(sigma)
sigma2 = np.dot(ssigma[:,None], ssigma[None,:])

cov = sigma2 * cor
mean = [0.117, 0.156, 0.143, 1.07, 0.98, 1.00, -0.03, 0.05, 0.12]


samples = np.random.multivariate_normal(mean, cov, 2000000)

diff = np.abs(samples - mean) / sigma
print(diff.sum(axis=1))
mask = np.abs(diff).sum(axis=1) < 30
print(mask.sum())
#samples = samples[mask, :]

m1,m2,m3,a1,a2,a3,e1,e2,e3 = tuple(samples[:,i] for i in range(samples.shape[1]))
ac = 0.4
ec = 0.3
mask = (np.abs(e1) < ec) & (np.abs(e2) < ec) & (np.abs(e3) < ec)  & (np.abs(a1-1) < ac)& (np.abs(a2-1) < ac)& (np.abs(a3-1) < ac) # & (m1 > 0) & (m2 > 0) & (m3 > 0)
m1,m2,m3,a1,a2,a3,e1,e2,e3 = tuple(samples[mask,i] for i in range(samples.shape[1]))
h1 = 87.4/(a1 * (1 + e1)**2)
h2 = 95.5/(a2 * (1 + e2)**2)
h3 = 102.8/(a3 * (1 + e3)**2)
d1 = 1175.5 * a1 / (1 + e1)
d2 = 1386.2 * a2 / (1 + e2)
d3 = 1509.4 * a3 / (1 + e3)


#plt.hist(e3,20)
'''
from chainconsumer import ChainConsumer
c = ChainConsumer()
c.add_chain(samples, parameters=['m1','m2','m3','a1','a2','a3','e1','h2','h3'])
c.plot()
'''

combined = np.vstack((m1,m2,m3,d1,d2,d3,h1,h2,h3))
newcor = np.corrcoef(combined)
combined = combined.T
np.savetxt("newcor.txt", newcor, fmt="%6.2f")
import matplotlib.pyplot as plt
plt.imshow(newcor, cmap='viridis', interpolation='none', vmax=1)

'''
from chainconsumer import ChainConsumer
c = ChainConsumer()
c.add_chain(combined, parameters=['m1','m2','m3','d1','d2','d3','h1','h2','h3'])
c.plot()
'''

from tabulate import tabulate
headers = [r"aaaaaaaaa"]*9
print(tabulate(np.round(newcor, decimals=2), headers, tablefmt="latex"))
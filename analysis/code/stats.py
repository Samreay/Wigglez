import numpy as np
import matplotlib, matplotlib.pyplot as plt


from scipy import linspace
from scipy import pi,sqrt,exp
from scipy.special import erf

from pylab import plot,show

def getBounds(xs,ys, desiredArea=0.6827):
    yscumsum = ys.cumsum() #interp1d(centers, dist)(xs)
    yscumsum /= yscumsum.max()
    indexes = np.arange(xs.size)
    
    startIndex = ys.argmax()
    maxVal = ys[startIndex]
    minVal = 0
    threshold = 0.001
    
    x1 = None
    x2 = None
    count = 0
    while x1 is None:
        mid = (maxVal + minVal) / 2.0
        count += 1                    
        good = 1
        try:
            if count > 100:
                raise Exception("no")
            i1 = np.where(ys[:startIndex] > mid)[0][0]
            i2 = startIndex + np.where(ys[startIndex:] < mid)[0][0]
        except:
            x1 = 0
            x2 = 0
            good = 0
            print("Parameter %s is not constrained" % param)
        area = yscumsum[i2] - yscumsum[i1]
        a = np.abs(area - desiredArea)
        #print(maxVal, minVal, area)
        if a < threshold:
            x1 = xs[i1]
            x2 = xs[i2]
        elif area < desiredArea:
            maxVal = mid
        elif area > desiredArea:
            minVal = mid
    return [x1, xs[startIndex], x2, i1, startIndex, i2, good]
    
    
    
    
def pdf(x):
    return 1/sqrt(2*pi) * exp(-x**2/2)

def cdf(x):
    return (1 + erf(x/sqrt(2))) / 2

def skew(x,e=0,w=1,a=0):
    t = (x-e) / w
    return 2 / w * pdf(t) * cdf(a*t)
    # You can of course use the scipy.stats.norm versions
    # return 2 * norm.pdf(t) * norm.cdf(a*t)


n = 2**10
e = 1.0 # location
w = 2.0 # scale
x = linspace(0,7,n) 
p = skew(x,e,w,9)
c = p.cumsum()
c /= c.max()

b0 = np.where(c > 0.15865)[0][0]
bm = np.where(c > 0.5)[0][0]
b1 = np.where(c > 0.84135)[0][0]
x0 = x[b0]
x1 = x[b1]


s=30

fig, ax = plt.subplots(figsize=(5,5), nrows=2, sharex=True)
ax[0].plot(x,p, color='k')
ax[1].plot(x,c, color='k')

ax[0].plot([x[b0], x[b0]], [0, p[b0]], color='r', ls='--')
ax[1].plot([0, x[b0]], [c[b0], c[b0]], color='r', ls='--')


ax[0].plot([x[b1], x[b1]], [0, p[b1]], color='r', ls='--')
ax[1].plot([0, x[b1]], [c[b1], c[b1]], color='r', ls='--')

midi = int(0.5 * (b0 + b1))
mid = x[midi]
ax[0].scatter([mid], [p[midi]], color='r', s=s, marker="o", label="Mean")
ax[1].scatter([mid], [c[midi]], color='r', s=s, marker="o")
ax[0].scatter([x[bm]], [p[bm]], color='r', s=s, marker="^", label="Cumulative")
ax[1].scatter([x[bm]], [c[bm]], color='r', s=s, marker="^")

bounds = getBounds(x,p)
ax[0].scatter([bounds[1]], [p[bounds[4]]], color='b', s=s, marker="s", label="Max Likelihood")
ax[1].scatter([bounds[1]], [c[bounds[4]]], color='b', s=s, marker="s")
ax[0].plot([bounds[0], bounds[0]], [0, p[bounds[3]]], color='b')
ax[0].plot([bounds[2], bounds[2]], [0, p[bounds[5]]], color='b')
ax[1].plot([0, bounds[0]], [c[bounds[3]], c[bounds[3]]], color='b', ls='-')
ax[1].plot([0, bounds[2]], [c[bounds[5]], c[bounds[5]]], color='b', ls='-')


ax[0].yaxis.set_major_locator(plt.MaxNLocator(5))
ax[1].set_xlabel("$x$", fontsize=14)
ax[0].set_ylabel("$P(x)$", fontsize=14)
ax[1].set_ylabel("$C(x)$", fontsize=14)

fig.tight_layout()

ax[0].set_xlim(0,7)
ax[1].set_xlim(0,7)
ax[1].set_ylim(0,1)
ax[0].set_ylim(0, 0.4)

ax[0].legend(frameon=False, scatterpoints = 1)
fig.savefig("stats.pdf", bbox_inches="tight")


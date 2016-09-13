import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

d = r"C:\Users\shint1\Desktop\shifted"

fs = []
fs.append(r"Xi1D_integrated_CLiggleZ_m00001-03600%s_region_01_r001-100_nsep60_nmu30_ds05_mpch-1_min000max200_mpch-1_mod1_shifted.pickle")
fs.append(r"Xi1D_integrated_CLiggleZ_m00001-03600%s_region_03_r001-100_nsep60_nmu30_ds05_mpch-1_min000max200_mpch-1_mod2_shifted.pickle")
fs.append(r"Xi1D_integrated_CLiggleZ_m00001-03600%s_region_09_r001-100_nsep60_nmu30_ds05_mpch-1_min000max200_mpch-1_mod3_shifted.pickle")
fs.append(r"Xi1D_integrated_CLiggleZ_m00001-03600%s_region_11_r001-100_nsep60_nmu30_ds05_mpch-1_min000max200_mpch-1_mod4_shifted.pickle")
fs.append(r"Xi1D_integrated_CLiggleZ_m00001-03600%s_region_15_r001-100_nsep60_nmu30_ds05_mpch-1_min000max200_mpch-1_mod5_shifted.pickle")
fs.append(r"Xi1D_integrated_CLiggleZ_m00001-03600%s_region_22_r001-100_nsep60_nmu30_ds05_mpch-1_min000max200_mpch-1_mod6_shifted.pickle")

bins = ['a', 'b', 'c']
results = []
for b in bins:
    ts = []
    tsd = []
    ls = []
    lsd = []
    for f in fs:
        data = pickle.load(open(d + os.sep + f % b, 'rb'), encoding='latin1')
        s = data['sep']
        ts.append(data['T mean'])
        tsd.append(np.std(data['T mocks'], axis=1))
        ls.append(data['L mean'])
        lsd.append(np.std(data['L mocks'], axis=1))
    ts = np.array(ts)
    ls = np.array(ls)
    tsd = np.array(tsd)
    lsd = np.array(lsd)

    tw = 1.0 / (tsd**2)
    lw = 1.0 / (lsd**2)

    tsd = np.sqrt((tsd*tsd).sum(axis=0)) / len(fs)
    lsd = np.sqrt((lsd*lsd).sum(axis=0)) / len(fs)

    ts = np.average(ts, axis=0, weights=tw)
    ls = np.average(ls, axis=0, weights=lw)
    results.append({"s": s, "t": ts, "te": tsd, "l": ls, "le": lsd})

    plt.errorbar(s, s*s*ts, yerr=s*s*tsd, fmt='.')
    plt.errorbar(s, s*s*ls, yerr=s*s*lsd, fmt='.')

pickle.dump(results, open("consolidated.pickle", 'wb'))
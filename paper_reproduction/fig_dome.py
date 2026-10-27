"""Figure S6: projected charge density across a neutral spherical Pt particle (phase object) for the two trackers
and the CoM. Run inside results_v1.1/ (reads dome_linescan.json)."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

K_DET = 0.18644e-3; P = 8244.5; PIX = 0.86
d = json.load(open('dome_linescan.json'))
R = d['R'] / 10.0
r_scan = np.array(d['res']['mip']['r']) / 10
prof_all = np.array(d['res']['mip']['prof'])
names = ['ellipse', 'bin_xcorr', 'com']
pcol = {'bin_xcorr': 0, 'ellipse': 1, 'com': 2}
STY = {'ellipse': dict(c='#D55E00', ls='-', label='Ellipse fit'),
       'bin_xcorr': dict(c='#0072B2', ls='-', label='Edge cross-correlation'),
       'com': dict(c='#CC79A7', ls='--', label='Center of mass (reference)')}

N = 241; c = (N - 1) / 2
yy, xx = np.mgrid[0:N, 0:N]; rr = np.hypot(xx - c, yy - c) * PIX / 10
ux = (xx - c) / np.maximum(np.hypot(xx - c, yy - c), 1e-9); uy = (yy - c) / np.maximum(np.hypot(xx - c, yy - c), 1e-9)


def sigma_profile(n):
    a_r = np.interp(rr, r_scan, prof_all[:, pcol[n]], right=0.0) * K_DET
    s = P * (np.gradient(a_r * ux, PIX / 10, axis=1) + np.gradient(a_r * uy, PIX / 10, axis=0))
    sb = gaussian_filter(s, 3.0)
    rb = np.arange(0, 6.0, 0.086)
    return rb + 0.043, np.array([sb[(rr >= a) & (rr < a + 0.086)].mean() for a in rb])


plt.rcParams.update({'font.size': 8.5, 'axes.linewidth': 0.8, 'font.family': 'DejaVu Sans'})
fig, ax = plt.subplots(figsize=(3.5, 3.0), dpi=300)
fig.subplots_adjust(left=0.2, right=0.97, top=0.95, bottom=0.33)
for n in names:
    rb, pb = sigma_profile(n)
    ax.plot(rb, pb, color=STY[n]['c'], ls=STY[n]['ls'], lw=1.3, label=STY[n]['label'])
    inner = pb[rb < 0.5].mean(); i = np.argmax(pb); k = np.where((pb[:-1] < 0) & (pb[1:] >= 0))[0][0]
    zc = rb[k] - pb[k] * (rb[k + 1] - rb[k]) / (pb[k + 1] - pb[k])
    print(f"{n:10s} interior {inner:+.2f}  rim {pb[i]:+.2f} at {rb[i]:.2f} nm  sign change {zc:.2f} nm")
ax.axvline(R, color='0.5', lw=0.7, ls=':'); ax.axhline(0, color='0.6', lw=0.6)
ax.set_xlim(0, 5); ax.set_xlabel('r (nm)'); ax.set_ylabel(r'$\sigma$ ($\sigma_g$ = 3 px) (e nm$^{-2}$)')
ax.tick_params(width=0.8, length=3)
fig.legend(loc='lower center', ncol=1, frameon=False, fontsize=7.6, bbox_to_anchor=(0.55, 0.0))
fig.savefig('FigS6_spatial_transfer.png', dpi=300)

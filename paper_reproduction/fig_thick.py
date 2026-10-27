"""Figure S5 from thick_thresholds.json (run inside results_v1.1/)."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
D = json.load(open('thick_thresholds.json'))
C = {'ellipse': '#D55E00', 'bin_xcorr': '#0072B2', 'com': '#CC79A7'}
TI = {'bin_xcorr': 0, 'ellipse': 1, 'com': 2}
cols = [('Carbon supports', [('aC', 'a-C', '-'), ('graphite_001', 'graphite [001]', '--')], 'aC'),
        ('TiO$_2$', [('TiO2r_001', '[001]', '-'), ('TiO2r_t5', '5° off', '--'), ('TiO2r_t10', '10° off', ':')], 'TiO2r_t5'),
        ('TiN (TiON proxy)', [('TiN_001', '[001]', '-'), ('TiN_t5', '5° off', '--'), ('TiN_t10', '10° off', ':')], 'TiN_t5'),
        ('Pt (particle)', [('Pt_011', '[011]', '-'), ('Pt_t5', '5° off', '--'), ('Pt_t10', '10° off', ':')], 'Pt_t5')]
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.8, 'font.family': 'DejaVu Sans'})
fig, axs = plt.subplots(2, 4, figsize=(7.2, 4.6), dpi=300)
fig.subplots_adjust(left=0.105, right=0.99, top=0.90, bottom=0.22, hspace=0.45, wspace=0.34)
for j, (title, cases, rep) in enumerate(cols):
    ax = axs[0, j]
    for name, lab, ls in cases:
        s = D[name]['series']; t = np.array(s['t']); eta = np.array(s['eta'])
        ok = np.array(s['applied']) >= 0.5 - 1e-9
        for k in ('ellipse', 'bin_xcorr'):
            tf = D[name]['contam'][k]
            okk = ok & ((t < tf) if tf is not None else np.ones(len(t), bool))
            ax.plot(t[ok], np.where(okk, eta[:, TI[k]], np.nan)[ok], color=C[k], ls=ls, lw=1.2)
        ax.plot(t[ok], eta[ok, TI['com']], color=C['com'], ls=ls, lw=1.2)
    ax.axhline(1, color='0.6', lw=0.6); ax.axhspan(0.9, 1.1, color='0.92', zorder=0)
    ax.set_ylim(0, 1.5); ax.set_xlim(0, max(D[n]['series']['t'][-1] for n, _, _ in cases))
    ax.set_title(title, fontsize=8.5)
    if j == 0: ax.set_ylabel('Response to uniform\nfield $\\eta$')
    ax = axs[1, j]
    for name, lab, ls in cases:
        s = D[name]['series']; ax.plot(s['t'], s['redge'], color='0.15', ls=ls, lw=1.1, label=lab)
    ax.axhline(0.1, color='#E69F00', lw=0.8, ls='-'); ax.axhline(0.3, color='#D55E00', lw=0.8, ls='-')
    ax.set_ylim(0, 2.5); ax.set_xlim(0, max(D[n]['series']['t'][-1] for n, _, _ in cases))
    ax.set_xlabel('Thickness (nm)')
    if j == 0: ax.set_ylabel('Edge contamination\n$r_{edge}$')
for i, ax in enumerate(axs.ravel()):
    ax.text(-0.36 if i % 4 == 0 else -0.27, 1.16, 'abcdefgh'[i], transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')
    ax.tick_params(width=0.8, length=2.5)
H = [plt.Line2D([], [], color=C['ellipse'], lw=1.2, label='Ellipse fit'),
     plt.Line2D([], [], color=C['bin_xcorr'], lw=1.2, label='Edge cross-correlation'),
     plt.Line2D([], [], color=C['com'], lw=1.2, label='Center of mass'),
     plt.Line2D([], [], color='#E69F00', lw=0.8, label='$r_{edge}$ = 0.1'),
     plt.Line2D([], [], color='#D55E00', lw=0.8, label='$r_{edge}$ = 0.3'),
     plt.Line2D([], [], color='0.15', lw=1.1, ls='-', label='zone axis (carbon: a-C)'),
     plt.Line2D([], [], color='0.15', lw=1.1, ls='--', label='5° off axis (carbon: graphite [001])'),
     plt.Line2D([], [], color='0.15', lw=1.1, ls=':', label='10° off axis')]
fig.legend(handles=H, loc='lower center', ncol=3, frameon=False, fontsize=6.8, bbox_to_anchor=(0.53, 0.0))
fig.savefig('FigS5_thickness.png', dpi=300)
print('ok')

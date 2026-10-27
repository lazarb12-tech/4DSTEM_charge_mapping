"""Thickness thresholds of SI Table S4 and the data of Figure S5, from thickness_out/*.npz.
Contamination: cell-averaged field-free shift > 0.5 px (edge trackers also: r.m.s. within the cell > 1 px).
Nonlinear field response: |eta - 1| > 10 % at two consecutive thicknesses, where the applied deflection is >= 0.5 px."""
import numpy as np, json, os, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else 'thickness_out'
CASES = [('aC', 'a-C (amorphous)', 'vacuum_t300'),
         ('graphite_001', 'graphite [001]', 'vacuum_t300_f100'),
         ('TiO2r_001', 'TiO2 [001]', 'vacuum_t300'),
         ('TiO2r_t5', 'TiO2, 5 deg off [001]', 'vacuum_t300'),
         ('TiO2r_t10', 'TiO2, 10 deg off [001]', 'vacuum_t300'),
         ('TiN_001', 'TiN [001]', 'vacuum_t300'),
         ('TiN_t5', 'TiN, 5 deg off [001]', 'vacuum_t300'),
         ('TiN_t10', 'TiN, 10 deg off [001]', 'vacuum_t300'),
         ('Pt_011', 'Pt [011]', 'vacuum_t300_f120'),
         ('Pt_t5', 'Pt, 5 deg off [011]', 'vacuum_t300_f120'),
         ('Pt_t10', 'Pt, 10 deg off [011]', 'vacuum_t300_f120')]
TR = ['bin_xcorr', 'ellipse', 'com']
FALSE_TOL = 0.5       # px, cell-averaged field-free (spurious) shift, all trackers
RMS_TOL = 1.0         # px, r.m.s. of the field-free shift over the cell (edge trackers only)
ETA_TOL = 0.10        # fractional deviation of the response to a uniform field
MIN_DEFL = 0.5        # px, applied deflection from which eta is evaluated
DEFL_DESIGN = 5.0     # px, applied deflection at thickness tf (thickness_series.py)


def metrics(name, vac):
    d = np.load(f'{OUT}/{name}.npz'); v = np.load(f'{OUT}/{vac}.npz')
    th = d['thick'] / 10.0; n = min(len(th), len(v['thick'])); th = th[:n]
    sh = d['shift'][:n]; vs = v['shift'][:n]
    vresp = np.nanmean(vs[:, :, :, 1, :, 0] - vs[:, :, :, 0, :, 0], axis=(1, 2))
    eta = np.nanmean(sh[:, :, :, 1, :, 0] - sh[:, :, :, 0, :, 0], axis=(1, 2)) / vresp
    false = np.nanmean(sh[:, :, :, 0, :, :], axis=2)
    fmag = np.nanmean(np.hypot(false[..., 0], false[..., 1]), axis=1)
    fmean = np.nanmean(false, axis=1); foff = np.hypot(fmean[..., 0], fmean[..., 1])
    dev = false - fmean[:, None]; frms = np.sqrt(np.nanmean(np.sum(dev**2, axis=-1), axis=1))
    tf = float(vac.split('_f')[1]) if '_f' in vac else float(vac.split('_t')[1])
    applied = DEFL_DESIGN * d['thick'][:n] / tf
    redge = np.nanmean(d['qc'][:n, :, :, 0, 0], axis=(1, 2))
    return th, eta, fmag, redge, float(d['mip']), foff, applied, frms


def first(th, bad):
    i = np.where(bad)[0]
    return th[i[0]] if len(i) else None


def fmt(t, tmax, t0=1.0):
    if t is None:
        return f'>{tmax:.0f}'
    return f'<={t0:.0f}' if t <= t0 else f'{t:.0f}'


rows = {}
for name, lab, vac in CASES:
    if not (os.path.exists(f'{OUT}/{name}.npz') and os.path.exists(f'{OUT}/{vac}.npz')):
        continue
    th, eta, fmag, redge, mip, foff, applied, frms = metrics(name, vac)
    tmax = th[-1]
    ev = applied >= MIN_DEFL - 1e-9
    t_eval = float(th[ev][0])
    r = dict(label=lab, mip=mip, tmax=tmax, t_eval=t_eval, contam={}, nonlin={})
    for k, tr in enumerate(TR):
        bad = foff[:, k] > FALSE_TOL
        if tr != 'com':
            bad = bad | (frms[:, k] > RMS_TOL)
        r['contam'][tr] = first(th, bad)
        dev = (np.abs(eta[:, k] - 1) > ETA_TOL) & ev
        r['nonlin'][tr] = first(th, dev & np.r_[dev[1:], False])
    r['redge01'] = first(th, redge > 0.1); r['redge03'] = first(th, redge > 0.3)
    rows[name] = r
    c = ' / '.join(fmt(r['contam'][t], tmax) for t in ('ellipse', 'bin_xcorr', 'com'))
    nl = ' / '.join(fmt(r['nonlin'][t], tmax, t_eval) for t in ('ellipse', 'bin_xcorr', 'com'))
    print(f"{lab:26s} MIP {mip:5.2f} V | r_edge>0.1 {fmt(r['redge01'], tmax):>4s} | contamination (ell / xc / CoM) {c:16s} | "
          f"nonlinear response {nl}  (nm)")
    rows[name]['series'] = dict(t=th.tolist(), eta=eta.tolist(), false=fmag.tolist(), offset=foff.tolist(),
                                applied=applied.tolist(), redge=redge.tolist(), rms=frms.tolist())
json.dump(rows, open('thick_thresholds.json', 'w'))

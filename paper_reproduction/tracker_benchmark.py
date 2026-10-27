"""Synthetic benchmark of the disc trackers (SI Figure S4, Table S3): patterns with the experimental geometry, known
rigid shifts and controlled contamination (overlapping discs, dark bands, dose, template mismatch, non-rigid patterns).
Usage: python tracker_benchmark.py [repetitions]  ->  tracker_benchmark_results.json"""
import os
import sys, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trackers import (get_disk_edge, build_template_edge, track_xcorr, fit_ellipse, com, qc_metrics,
                      edge_contamination, edge_radius)

N = 256
R0 = 18.7e-3 / (55e-6 / 0.295)          # 100.3 px
C0 = np.array([128.3, 127.6])          # vacuum disc centre (x, y)
G_LIST = [69.0, 90.0, 97.5, 102.4, 132.8]  # TiO2 reflection distances (px)
YY, XX = np.mgrid[0:N, 0:N].astype(float)
EDGE_W = 1.0                           # soft edge width (px)
BKG = 0.01


def disc(cx, cy, R=R0):
    return 0.5 * (1.0 - np.tanh((np.hypot(XX - cx, YY - cy) - R) / EDGE_W))


def make_pattern(rng, shift, r=0.0, ndisc=0, D=0.0, nband=0, Ne=1.5e5, dR=0.0,
                 nonrigid=None, poisson=True, coh=False):
    cx, cy = C0 + shift
    if nonrigid is not None:
        w, delta = nonrigid
        img = (1 - w) * disc(C0[0], C0[1], R0 + dR) + w * disc(C0[0] + delta, C0[1], R0 + dR)
    else:
        img = disc(cx, cy, R0 + dR)
    if ndisc > 0:
        phis = []
        if ndisc == 1:
            phis = [rng.uniform(0, 2*np.pi)]
            gs = [rng.choice(G_LIST)]
        elif ndisc == 2:
            p = rng.uniform(0, 2*np.pi); phis = [p, p + np.pi]
            g = rng.choice(G_LIST); gs = [g, g]
        else:
            p1 = rng.uniform(0, 2*np.pi); p2 = p1 + rng.uniform(np.pi/4, 3*np.pi/4)
            g1, g2 = rng.choice(G_LIST, 2)
            phis = [p1, p1 + np.pi, p2, p2 + np.pi]; gs = [g1, g1, g2, g2]
        bf = img.copy()
        for p, g in zip(phis, gs):
            dg = disc(cx + g*np.cos(p), cy + g*np.sin(p), R0 + dR)
            img = img + r * dg
            if coh:
                kf = rng.uniform(0.0, 1.0/8.0); ph = rng.uniform(0, 2*np.pi)
                u = (XX - cx)*np.cos(p) + (YY - cy)*np.sin(p)
                img = img + 2*np.sqrt(r)*np.cos(2*np.pi*kf*u + ph) * np.sqrt(bf*dg)
    if nband > 0 and D > 0:
        mod = np.ones_like(img)
        for _ in range(nband):
            th = rng.uniform(0, np.pi)
            off = rng.uniform(-0.6*R0, 0.6*R0)
            dist = (XX - cx)*np.cos(th) + (YY - cy)*np.sin(th) - off
            mod *= 1.0 - D*np.exp(-0.5*(dist/6.0)**2)
        img = img * mod
    img = np.clip(img, 0, None) + BKG
    img = img / img.sum() * Ne
    return rng.poisson(img).astype(float) if poisson else img


def vacuum_reference(Ne, seed=0):
    rng = np.random.default_rng(seed)
    vac = [make_pattern(rng, np.zeros(2), Ne=Ne) for _ in range(25)]
    tmpl = build_template_edge(vac)
    ell = np.array([fit_ellipse(p)[:2] for p in vac])
    cm = np.array([com(p) for p in vac])
    return tmpl, np.nanmean(ell, 0), cm.mean(0)


_REF = {}


def get_ref(Ne):
    if Ne not in _REF:
        _REF[Ne] = vacuum_reference(Ne)
    return _REF[Ne]


def run_one(args):
    cond, seed = args
    rng = np.random.default_rng(seed)
    Ne = cond.get('Ne', 1.5e5)
    tmpl, ell0, com0 = get_ref(Ne)
    ang = rng.uniform(0, 2*np.pi); mag = rng.uniform(0.0, 3.0)
    shift = np.array([mag*np.cos(ang), mag*np.sin(ang)])
    kw = {k: v for k, v in cond.items() if k != 'name'}
    p = make_pattern(rng, shift, **kw)
    edge = get_disk_edge(p)
    xc_dx, xc_dy, xerr = track_xcorr(p, tmpl, edge=edge)
    ex, ey, ea, eb = fit_ellipse(p, edge=edge)
    cmx, cmy = com(p)
    rmax, rmin = edge_contamination(p, C0[0] + xc_dx, C0[1] + xc_dy, R0)
    rad = edge_radius(edge, C0[0] + xc_dx, C0[1] + xc_dy, R0)
    true = shift if cond.get('nonrigid') is None else np.array([cond['nonrigid'][0]*cond['nonrigid'][1], 0.0])
    q = qc_metrics(edge, C0[0] + xc_dx, C0[1] + xc_dy, R0)
    return dict(name=cond['name'], seed=seed,
                true=true.tolist(),
                xcorr=[xc_dx - true[0], xc_dy - true[1]],
                ellipse=[ex - ell0[0] - true[0], ey - ell0[1] - true[1]],
                com=[cmx - com0[0] - true[0], cmy - com0[1] - true[1]],
                ell_rad=[ea, eb], r_edge_max=rmax, r_edge_min=rmin, edge_rad=rad,
                counts=float(p.sum()), **q)


def conditions():
    C = []
    for nd in (1, 2, 4):
        for r in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5):
            C.append(dict(name=f'disc_n{nd}_r{r}', r=r, ndisc=nd))
    for nd in (1, 4):
        for r in (0.1, 0.2, 0.3, 0.5, 0.8):
            C.append(dict(name=f'coh_n{nd}_r{r}', r=r, ndisc=nd, coh=True))
    for nb in (1, 2):
        for D in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            C.append(dict(name=f'band_n{nb}_D{D}', D=D, nband=nb))
    for Ne in (3e2, 1e3, 2e3, 3e3, 5e3, 1e4, 3e4, 1e5, 1.5e5):
        C.append(dict(name=f'dose_{int(Ne)}', Ne=Ne))
    for dR in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        C.append(dict(name=f'dR_{dR}', dR=dR))
    for w in (0.25, 0.5, 0.75):
        for d in (1.0, 2.0, 4.0, 8.0):
            C.append(dict(name=f'nonrigid_w{w}_d{d}', nonrigid=(w, d)))
    for r in (0.2, 0.4, 0.6):
        for D in (0.5, 0.8):
            C.append(dict(name=f'combo_r{r}_D{D}', r=r, ndisc=1, D=D, nband=1))
    return C


if __name__ == '__main__':
    nrep = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    jobs = []
    for i, c in enumerate(conditions()):
        for k in range(nrep):
            jobs.append((c, 1000*i + k))
    t0 = time.time()
    with Pool(2) as pool:
        res = pool.map(run_one, jobs, chunksize=8)
    json.dump(res, open('tracker_benchmark_results.json', 'w'))
    print(f'{len(res)} patterns in {time.time()-t0:.0f} s')

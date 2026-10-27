"""Run this first on new data, before CentreFitting_xcorr.py or CentreFitting_physical.py, to find out which disc
tracker to use. Output: results/<FILE_STEM>/qc/."""
import os
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
from joblib import Parallel, delayed
import disc_tracking as dt

# ----------------------------------------------------------------- SETTINGS
FILE_STEM = "FILENAME"
VAC_H5_STEM = "FILENAME_vacuum"
PROJECT_ROOT = Path(os.environ.get("FOURDSTEM_ROOT", Path(__file__).resolve().parent))  # or edit this path
TRACKERS = ["edge_binary", "ellipse", "com_bf"]
JUDGED = ["edge_binary", "ellipse"]   # trackers judged by the impact test
IMPACT_TOL = 0.05             # px, allowed contamination shift after smoothing
REL_TOL = 0.10                # ... or this fraction of the local deflection, whichever is larger
PREFERENCE = ["edge_binary", "ellipse"]   # the first acceptable tracker is recommended
MAP_SIGMA = 3.0               # scan px, Gaussian smoothing of the charge maps
MAX_FRACTION = 0.01           # fraction of positions allowed above IMPACT_TOL
SPREAD_TOL = 0.3              # px, disagreement of the trackers on the cleaned patterns that flags a position
ELLIPSE_MEDIAN3 = True        # 3x3 median filter of the ellipse maps, as CentreFitting_physical.py
COM_WINDOW = "follow"         # CoM window follows the edge-tracked disc ("fixed": vacuum position)
REUSE_SAVED_SHIFTS = True     # reuse saved shift maps of an earlier run
RERUN = []                    # trackers to recompute even if saved
UPSAMPLE = 20                 # as in CentreFitting_xcorr.py
SMART_REF_SIZE = 5
N_BASELINE = 25               # vacuum patterns used for the self-check
ELECTRONS_PER_PATTERN = None  # None: estimated from the vacuum counts
STRIDE = 1                    # > 1 for a quick look on a sub-sampled scan
N_JOBS = -1
# --------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "results" / FILE_STEM / "qc"
DESCRIPTORS = ('r_edge', 'overlap', 'modulation', 'd_radius')


def smart_vacuum_block(vac, size=SMART_REF_SIZE):
    Rx, Ry = vac.shape[:2]
    half = size // 2
    means = np.asarray(vac).reshape(Rx, Ry, -1).mean(axis=2)
    best, best_rc = np.inf, (Rx // 2, Ry // 2)
    for r in range(half + 1, Rx - half - 1):
        for c in range(half + 1, Ry - half - 1):
            s = np.var(means[r - half:r + half + 1, c - half:c + half + 1])
            if s < best:
                best, best_rc = s, (r, c)
    r, c = best_rc
    pats = [np.asarray(vac[i, j], float) for i in range(r - half, r + half + 1) for j in range(c - half, c + half + 1)]
    return pats, best_rc


def smart_vacuum_patterns(vac, size=SMART_REF_SIZE):
    return smart_vacuum_block(vac, size)[0]


def baseline_vacuum_patterns(vac, block_rc, n=N_BASELINE, size=SMART_REF_SIZE, seed=0):
    Rx, Ry = vac.shape[:2]; half = size // 2
    cand = [(i, j) for i in range(Rx) for j in range(Ry)
            if not (abs(i - block_rc[0]) <= half and abs(j - block_rc[1]) <= half)]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(cand), size=min(n, len(cand)), replace=False)
    return [np.asarray(vac[cand[k][0], cand[k][1]], float) for k in pick]


def analyse_pattern(dp, ref, todo, upsample, known, com_window, judged):
    out = dict(known)
    for t in sorted(todo, key=lambda t: t == 'com_bf'):
        try:
            if t == 'com_bf' and com_window == 'follow':
                c = out.get('edge_binary')
                if c is None or not np.all(np.isfinite(c)):
                    c = dt.track(dp, 'edge_binary', ref, upsample=upsample)
                out[t] = dt.track(dp, 'com_bf', ref, centre_shift=c)
            else:
                out[t] = dt.track(dp, t, ref, upsample=upsample)
        except Exception:
            out[t] = (np.nan, np.nan)
    est = {t: np.asarray(out[t], float) for t in judged if t in out}
    ci = dt.contamination_impact(dp, ref, est, tuple(judged), upsample)
    base = out.get('edge_binary')
    q = dt.qc(dp, base[0], base[1], ref) if base is not None and np.all(np.isfinite(base)) else dict(r_edge=np.nan, d_radius=np.nan)
    if ci is None:
        ci = dict(clean={t: np.array([np.nan, np.nan]) for t in judged}, spread=np.nan, overlap=np.nan, modulation=np.nan)
    desc = dict(r_edge=q['r_edge'], d_radius=q['d_radius'], overlap=ci['overlap'], modulation=ci['modulation'])
    return {t: out[t] for t in out}, ci['clean'], ci['spread'], desc, ci.get('time_ms', {})


def load_saved_shifts(trackers, shape):
    saved = {}
    for t in trackers:
        fx, fy = OUT_DIR / f'{FILE_STEM}_shift_{t}_x.npy', OUT_DIR / f'{FILE_STEM}_shift_{t}_y.npy'
        if t in RERUN or not (fx.exists() and fy.exists()):
            continue
        s = np.stack([np.load(fx), np.load(fy)], axis=-1)
        if s.shape[:2] == shape:
            saved[t] = s
        else:
            print(f'saved shifts for {t} have shape {s.shape[:2]}, expected {shape}: recomputing')
    return saved


def deflection_signal(s):
    ny, nx = s.shape[:2]; yy, xx = np.mgrid[0:ny, 0:nx]
    comp = []
    for k in (0, 1):
        d = s[..., k]; ok = np.isfinite(d)
        A = np.stack([xx[ok], yy[ok], np.ones(ok.sum())], 1)
        c, *_ = np.linalg.lstsq(A, d[ok], rcond=None)
        r = np.where(ok, d - (c[0] * xx + c[1] * yy + c[2]), 0.0)
        comp.append(gaussian_filter(r, MAP_SIGMA))
    return np.hypot(*comp)


def tracker_differences(shifts, ref_tracker='edge_binary'):
    out = {}
    for t, s in shifts.items():
        if t == ref_tracker or ref_tracker not in shifts:
            continue
        d = s - shifts[ref_tracker]
        out[t] = np.hypot(*(d[..., k] - np.nanmedian(d[..., k]) for k in (0, 1)))
    return out


def run(data, vac_patterns, trackers=TRACKERS, judged=JUDGED, upsample=UPSAMPLE, stride=STRIDE, n_jobs=N_JOBS,
        electrons_per_pattern=ELECTRONS_PER_PATTERN, baseline_patterns=None, reuse=REUSE_SAVED_SHIFTS,
        com_window=COM_WINDOW):
    judged = [t for t in judged if t in trackers]
    ref = dt.build_reference(vac_patterns)
    Rx, Ry = data.shape[:2]
    idx = [(i, j) for i in range(0, Rx, stride) for j in range(0, Ry, stride)]
    shp = (len(range(0, Rx, stride)), len(range(0, Ry, stride)))
    saved = load_saved_shifts(trackers, shp) if reuse else {}
    todo = [t for t in trackers if t not in saved]
    print(f'trackers loaded from file: {sorted(saved) or "none"}; computing: {todo or "none"}; '
          f'cleaning + impact for {judged}')
    res = Parallel(n_jobs=n_jobs)(
        delayed(analyse_pattern)(np.asarray(data[i, j], float), ref, todo, upsample,
                                 {t: tuple(saved[t][i // stride, j // stride]) for t in saved}, com_window, judged)
        for i, j in idx)
    shifts = {t: np.full(shp + (2,), np.nan) for t in trackers}
    clean = {t: np.full(shp + (2,), np.nan) for t in judged}
    spread = np.full(shp, np.nan)
    desc = {k: np.full(shp, np.nan) for k in DESCRIPTORS}
    times = {t: [] for t in judged}
    for (i, j), (o, c, sp, d, tm) in zip(idx, res):
        for t, v in tm.items():
            times[t].append(v)
        a, b = i // stride, j // stride
        for t in trackers:
            shifts[t][a, b] = o[t]
        for t in judged:
            clean[t][a, b] = c[t]
        spread[a, b] = sp
        for k in desc:
            desc[k][a, b] = d[k]
    def proc(t, m):
        if t == 'ellipse' and ELLIPSE_MEDIAN3:
            return np.stack([median_filter(m[..., k], size=3) for k in (0, 1)], -1)
        return m
    impact = {t: proc(t, shifts[t]) - proc(t, clean[t]) for t in judged}
    if electrons_per_pattern is None:
        electrons_per_pattern = float(np.mean([p.sum() for p in vac_patterns]))

    msgs = []
    bs = dt.boundary_summary(ref)
    msgs.append(f"vacuum disc: semi-axes {bs['a']:.1f} / {bs['b']:.1f} px (ellipticity {100*bs['ellipticity']:.1f} %), "
                f"{electrons_per_pattern:.3g} counts per pattern")
    if baseline_patterns:
        bl = dt.impact_baseline(baseline_patterns, ref, tuple(judged), upsample)
        msgs.append(f'self-check on {len(baseline_patterns)} vacuum patterns (no contamination, impact should be ~0): '
                    + '; '.join(f'{t} {v[0]:.3f} / {v[1]:.3f} px' for t, v in bl.items()) + '  (median / 95th pct)')
    signal = deflection_signal(proc(*next((t, shifts[t]) for t in ('edge_binary', judged[0]) if t in shifts)))
    timing = {t: float(np.median(v)) for t, v in times.items() if v}
    choice, m, smoothed = dt.recommend(impact, spread, electrons_per_pattern, desc['d_radius'], tol=IMPACT_TOL,
                                       sigma=MAP_SIGMA, max_fraction=MAX_FRACTION, spread_tol=SPREAD_TOL,
                                       signal=signal, rel_tol=REL_TOL, preference=PREFERENCE, timing=timing)
    msgs += m
    for t, d in tracker_differences({t: proc(t, v) for t, v in shifts.items()}).items():
        msgs.append(f'{t} vs edge_binary |difference| (constant offset removed): median {np.nanmedian(d):.3f}, '
                    f'95th pct {np.nanpercentile(d, 95):.3f} px')
    msgs.append(f'descriptors: overlap I_out/I_BF > 0.1 at {100*np.nanmean(desc["overlap"] > 0.1):.1f} % of positions '
                f'(r_edge > 0.1: {100*np.nanmean(desc["r_edge"] > 0.1):.1f} %), modulation > 0.3 at '
                f'{100*np.nanmean(desc["modulation"] > 0.3):.1f} %, median d_radius {np.nanmedian(desc["d_radius"]):+.2f} px')
    return dict(shifts=shifts, clean=clean, impact=impact, smoothed=smoothed, spread=spread, desc=desc,
                choice=choice, msgs=msgs, ref=ref)


def save(R):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for t, s in R['shifts'].items():
        np.save(OUT_DIR / f'{FILE_STEM}_shift_{t}_x.npy', s[..., 0]); np.save(OUT_DIR / f'{FILE_STEM}_shift_{t}_y.npy', s[..., 1])
    for t, s in R['clean'].items():
        np.save(OUT_DIR / f'{FILE_STEM}_clean_{t}_x.npy', s[..., 0]); np.save(OUT_DIR / f'{FILE_STEM}_clean_{t}_y.npy', s[..., 1])
    for t, s in R['smoothed'].items():
        np.save(OUT_DIR / f'{FILE_STEM}_impact_{t}.npy', s)
    np.save(OUT_DIR / f'{FILE_STEM}_spread.npy', R['spread'])
    for k, v in R['desc'].items():
        np.save(OUT_DIR / f'{FILE_STEM}_qc_{k}.npy', v)
    judged = list(R['smoothed'])
    ncol = max(len(judged) + 1, 4)
    fig, axs = plt.subplots(2, ncol, figsize=(4.6 * ncol, 8.4), gridspec_kw=dict(wspace=0.45))
    vmax = max(2 * IMPACT_TOL, max(np.nanpercentile(v, 99.5) for v in R['smoothed'].values()))
    for ax, t in zip(axs[0], judged):
        im = ax.imshow(R['smoothed'][t], cmap='magma', vmin=0, vmax=vmax)
        frac = np.mean(R['smoothed'][t] > IMPACT_TOL)
        ax.contour(R['smoothed'][t], [IMPACT_TOL], colors='c', linewidths=0.6)
        ax.set_title(f'{t}: contamination impact (px)\nsmoothed sigma={MAP_SIGMA:g}; cyan = {IMPACT_TOL:g} px '
                     f'({100*frac:.1f} % above)', fontsize=9)
        ax.axis('off'); plt.colorbar(im, ax=ax, fraction=0.046)
    ax = axs[0, len(judged)]
    im = ax.imshow(R['spread'], cmap='gray_r', vmin=0, vmax=max(2 * SPREAD_TOL, np.nanpercentile(R['spread'], 99)))
    ax.contour(R['spread'] > SPREAD_TOL, [0.5], colors='r', linewidths=0.6)
    ax.set_title(f'tracker disagreement on cleaned patterns (px)\nred: > {SPREAD_TOL:g} px (not removable)', fontsize=9)
    ax.axis('off'); plt.colorbar(im, ax=ax, fraction=0.046)
    for ax in axs[0, len(judged) + 1:]:
        ax.axis('off')
    spec = [('overlap', 0, 0.5, 'viridis', 'secondary-disc intensity at the edge\n(I_out / I_BF)'),
            ('modulation', 0, 1, 'viridis', 'band / contrast depth inside the disc'),
            ('d_radius', -1, 1, 'RdBu_r', 'edge radius - template (px)')]
    for ax, (k, lo, hi, cm, title) in zip(axs[1], spec):
        im = ax.imshow(R['desc'][k], vmin=lo, vmax=hi, cmap=cm); ax.set_title(title, fontsize=9); ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    ax = axs[1, 3]
    for t in judged:
        v = np.sort(np.hypot(R['impact'][t][..., 0], R['impact'][t][..., 1]).ravel()); v = v[np.isfinite(v)]
        ax.plot(v, 1 - np.arange(v.size) / max(v.size, 1), label=t)
    ax.set_xscale('symlog', linthresh=0.05); ax.set_yscale('log'); ax.set_ylim(1e-3, 1)
    ax.axvline(IMPACT_TOL, color='0.5', lw=0.7, ls='--')
    ax.set_xlabel('per-pattern |impact| (px)'); ax.set_ylabel('fraction above'); ax.legend(fontsize=8)
    ax.set_title('per-pattern impact (unsmoothed)', fontsize=9)
    for ax in axs[1, 4:]:
        ax.axis('off')
    fig.suptitle(R['choice'], fontsize=10)
    plt.savefig(OUT_DIR / f'{FILE_STEM}_qc_report.png', dpi=140, bbox_inches='tight'); plt.close()
    (OUT_DIR / f'{FILE_STEM}_qc_report.txt').write_text(dt.RULES + '\n' + '\n'.join(R['msgs']) + '\n')


if __name__ == '__main__':
    import py4DSTEM
    data = py4DSTEM.read(str(DATA_DIR / f'{FILE_STEM}.h5')).data
    vac = py4DSTEM.read(str(DATA_DIR / f'{VAC_H5_STEM}.h5')).data
    vac_ref, block_rc = smart_vacuum_block(vac)
    vac_check = baseline_vacuum_patterns(vac, block_rc)
    R = run(data, vac_ref, baseline_patterns=vac_check)
    save(R)
    print(dt.RULES)
    print('\n'.join(R['msgs']))

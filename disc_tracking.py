"""Helper module with the disc trackers and the contamination test. Other scripts import it; no need to run it."""
import time
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage import filters, feature, morphology
from skimage.registration import phase_cross_correlation

_DISK3 = morphology.disk(3)
_DISK5 = morphology.disk(5)

def get_disk_edge(dp):
    dp_med = filters.median(dp.astype(np.float32), _DISK3)
    thresh = filters.threshold_otsu(dp_med)
    dp_closed = morphology.closing(dp_med > thresh, _DISK5)
    return feature.canny(dp_closed.astype(float), sigma=1).astype(float)


def _ellipse_hf(xy):
    data = xy.astype(float)
    if len(data) < 5:
        return None
    origin = data.mean(axis=0); data = data - origin
    scale = data.std()
    if scale < np.finfo(float).tiny:
        return None
    data = data / scale
    x, y = data[:, 0], data[:, 1]
    D1 = np.vstack([x**2, x*y, y**2]).T; D2 = np.vstack([x, y, np.ones_like(x)]).T
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    C1 = np.array([[0., 0., 2.], [0., -1., 0.], [2., 0., 0.]])
    try:
        M = np.linalg.inv(C1) @ (S1 - S2 @ np.linalg.inv(S3) @ S2.T)
    except np.linalg.LinAlgError:
        return None
    ev, evec = np.linalg.eig(M); evec = np.real(evec)
    cond = 4*evec[0, :]*evec[2, :] - evec[1, :]**2
    a1 = evec[:, cond > 0]
    if a1.size != 3:
        return None
    a, b, c = a1.ravel()
    d, f, g = (-np.linalg.inv(S3) @ S2.T @ a1).ravel()
    b, d, f = b/2., d/2., f/2.
    den = b**2 - a*c
    x0 = (c*d - b*f)/den; y0 = (a*f - b*d)/den
    num = a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g
    term = np.sqrt((a - c)**2 + 4*b**2)
    with np.errstate(invalid='ignore'):
        w = np.sqrt(2*num/(den*(term - (a + c)))); h = np.sqrt(2*num/(den*(-term - (a + c))))
    if not (np.isfinite(w) and np.isfinite(h)):
        return None
    return x0*scale + origin[0], y0*scale + origin[1], max(w, h)*scale, min(w, h)*scale


def fit_ellipse(dp, edge=None):
    if edge is None:
        edge = get_disk_edge(dp)
    pts = np.column_stack(np.where(edge > 0))
    out = _ellipse_hf(pts[:, ::-1].astype(float)) if len(pts) >= 5 else None
    return (np.nan,)*4 if out is None else out


def build_reference(vac_patterns, bf_radius_px=None, nbins=180):
    vac = [np.asarray(p, float) for p in vac_patterns]
    mean = np.mean(vac, axis=0)
    edges = [get_disk_edge(p) for p in vac]
    ref = dict(edge_binary=np.mean(edges, axis=0))
    ell = np.array([fit_ellipse(p, e)[:4] for p, e in zip(vac, edges)])
    ref['ellipse_c0'] = np.nanmean(ell[:, :2], axis=0)
    r0 = np.nanmean(ell[:, 2:4]) if bf_radius_px is None else bf_radius_px
    ref['r0'] = float(r0)
    ref['vac_ellipse'] = np.nanmean(ell, axis=0)
    rr, aa = [], []
    for e, (x0, y0, _, _) in zip(edges, ell):
        if not np.isfinite(x0):
            continue
        ys, xs = np.nonzero(e > 0)
        r = np.hypot(xs - x0, ys - y0); keep = np.abs(r - r0) < 0.15 * r0
        rr.append(r[keep]); aa.append(np.arctan2(ys - y0, xs - x0)[keep])
    rr, aa = np.concatenate(rr), np.concatenate(aa)
    b = np.floor((aa + np.pi) / (2 * np.pi) * nbins).astype(int) % nbins
    prof = np.array([np.median(rr[b == k]) if np.count_nonzero(b == k) >= 3 else np.nan for k in range(nbins)])
    ok = np.isfinite(prof)
    if ok.sum() >= 3:
        ext = np.concatenate([prof[-1:], prof, prof[:1]])
        prof = np.where(ok, np.nanmedian(np.stack([ext[:-2], ext[1:-1], ext[2:]]), axis=0), np.nan)
    ref['bnd_theta'] = -np.pi + (np.arange(nbins) + 0.5) * 2 * np.pi / nbins
    ref['bnd_r'] = prof
    yy, xx = np.indices(mean.shape)
    ref['bf_window'] = (_rel_to_boundary(xx, yy, ref['ellipse_c0'][0], ref['ellipse_c0'][1], ref) <= 6).astype(float)
    ref['com_c0'] = np.array(_com(mean, ref['bf_window']))
    rel0 = _rel_to_boundary(xx, yy, ref['ellipse_c0'][0], ref['ellipse_c0'][1], ref)
    ref['lvl_in'] = float(np.median(mean[rel0 < -3]))
    ref['lvl_out'] = float(np.median(mean[(rel0 > 3) & (rel0 < 8)]))
    return ref


def template_radius(ref, theta):
    c, p = ref['bnd_theta'], ref['bnd_r']; ok = np.isfinite(p)
    if ok.sum() < 3:
        return np.full_like(np.asarray(theta, float), ref['r0'])
    return np.interp(theta, c[ok], p[ok], period=2 * np.pi)


def _rel_to_boundary(x, y, xc, yc, ref):
    return np.hypot(x - xc, y - yc) - template_radius(ref, np.arctan2(y - yc, x - xc))


def boundary_summary(ref):
    x0, y0, a, b = ref['vac_ellipse']
    p = ref['bnd_r'][np.isfinite(ref['bnd_r'])]
    return dict(centre=(float(x0), float(y0)), a=float(a), b=float(b), ellipticity=float(a / b - 1),
                boundary_dev_px=float(np.max(np.abs(p - np.mean(p)))) if p.size else np.nan,
                traced_fraction=float(np.isfinite(ref['bnd_r']).mean()))


def _com(dp, w=None):
    a = dp.astype(float) if w is None else dp.astype(float) * w
    yy, xx = np.indices(dp.shape); s = a.sum()
    return (xx*a).sum()/s, (yy*a).sum()/s


def track(dp, method, ref, upsample=20, centre_shift=None):
    dp = np.asarray(dp, float)
    if method == 'edge_binary':
        s, _, _ = phase_cross_correlation(ref['edge_binary'], get_disk_edge(dp), upsample_factor=upsample)
        return -s[1], -s[0]
    if method == 'ellipse':
        x, y, _, _ = fit_ellipse(dp)
        return x - ref['ellipse_c0'][0], y - ref['ellipse_c0'][1]
    if method == 'com_bf':
        if centre_shift is None or not np.all(np.isfinite(centre_shift)):
            w = ref['bf_window']
        else:
            yy, xx = np.indices(dp.shape)
            w = (_rel_to_boundary(xx, yy, ref['ellipse_c0'][0] + centre_shift[0],
                                  ref['ellipse_c0'][1] + centre_shift[1], ref) <= 6).astype(float)
        x, y = _com(dp, w)
        return x - ref['com_c0'][0], y - ref['com_c0'][1]
    raise ValueError(method)


def qc(dp, dx, dy, ref, tol=2.0, nbins=180, band=(3.0, 8.0), nsect=36):
    dp = np.asarray(dp, float)
    xc, yc = ref['ellipse_c0'][0] + dx, ref['ellipse_c0'][1] + dy
    edge = get_disk_edge(dp)
    ys, xs = np.nonzero(edge > 0)
    out = dict(inlier_frac=0.0, coverage=0.0, d_radius=np.nan, r_edge=np.nan)
    if xs.size:
        d = _rel_to_boundary(xs, ys, xc, yc, ref); near = np.abs(d) <= tol
        out['inlier_frac'] = float(near.mean())
        ang = np.arctan2(ys[near] - yc, xs[near] - xc)
        hit = np.zeros(nbins, bool)
        hit[np.floor((ang + np.pi)/(2*np.pi)*nbins).astype(int) % nbins] = True
        traced = np.isfinite(ref['bnd_r']) if ref['bnd_r'].size == nbins else np.ones(nbins, bool)
        out['coverage'] = float((hit & traced).sum() / max(traced.sum(), 1))
        m = np.abs(d) <= 6
        out['d_radius'] = float(np.median(d[m])) if m.any() else np.nan
    yy, xx = np.indices(dp.shape)
    rel = _rel_to_boundary(xx, yy, xc, yc, ref); aa = np.arctan2(yy - yc, xx - xc)
    inner = (rel > -band[1]) & (rel < -band[0]); outer = (rel > band[0]) & (rel < band[1])
    lvl = np.median(dp[inner]) if inner.any() else np.nan
    sect = np.floor((aa + np.pi)/(2*np.pi)*nsect).astype(int) % nsect
    vals = [dp[outer & (sect == s)].mean() for s in range(nsect) if (outer & (sect == s)).sum() > 5]
    if vals and np.isfinite(lvl) and lvl > 0:
        out['r_edge'] = float(np.max(vals) / lvl)
    return out


def _sample_along(L, xc, yc, ref, theta, offset):
    r = template_radius(ref, theta) + offset
    return map_coordinates(L, [yc + r * np.sin(theta), xc + r * np.cos(theta)], order=1, mode='nearest')


def suppress_contamination(dp, ref, dx, dy, sigma=1.5, band=(2.5, 5.0), min_contrast=0.2):
    dp = np.asarray(dp, float)
    xc, yc = ref['ellipse_c0'][0] + dx, ref['ellipse_c0'][1] + dy
    yy, xx = np.indices(dp.shape)
    rel = _rel_to_boundary(xx, yy, xc, yc, ref); th = np.arctan2(yy - yc, xx - xc)
    mid = 0.5 * (band[0] + band[1])

    def level(mask):
        num = gaussian_filter(dp * mask, sigma); den = gaussian_filter(mask.astype(float), sigma)
        return np.where(den > 0.05, num / np.maximum(den, 0.05), 0.0)
    m_in, m_out = rel < -band[0], rel > band[0]
    Lin, Lout = level(m_in), level(m_out)
    Lin_x, Lout_x = Lin.copy(), Lout.copy()
    Lin_x[~m_in] = _sample_along(Lin, xc, yc, ref, th[~m_in], -mid)
    Lout_x[~m_out] = _sample_along(Lout, xc, yc, ref, th[~m_out], +mid)
    dv = ref['lvl_in'] - ref['lvl_out']
    contrast = Lin_x - Lout_x
    clean = ref['lvl_out'] + (dp - Lout_x) * dv / np.maximum(contrast, min_contrast * dv)
    m_in3, m_out3 = rel < -3.0, rel > 3.0
    num = gaussian_filter(dp * m_in3, 3.0); den = gaussian_filter(m_in3.astype(float), 3.0)
    Lin3 = np.where(den > 0.05, num / np.maximum(den, 0.05), 0.0)
    num = gaussian_filter(dp * m_out3, 3.0); den = gaussian_filter(m_out3.astype(float), 3.0)
    Lout3 = np.where(den > 0.05, num / np.maximum(den, 0.05), 0.0)
    ang = np.linspace(-np.pi, np.pi, 180, endpoint=False)
    li, lo = _sample_along(Lin3, xc, yc, ref, ang, -5.5), _sample_along(Lout3, xc, yc, ref, ang, +5.5)
    deep = rel < -8.0
    med = np.median(Lin3[deep]) if deep.any() else np.nan
    desc = dict(overlap=float(np.max((lo - ref['lvl_out']) / np.maximum(li, 1e-9))),
                edge_contrast=float(np.min((li - lo) / dv)),
                modulation=float(max(1 - np.percentile(Lin3[deep], 1) / med, np.percentile(Lin3[deep], 99) / med - 1))
                if deep.any() and med > 0 else np.nan)
    return clean, desc


def contamination_impact(dp, ref, est, trackers=('edge_binary', 'ellipse'), upsample=20,
                         centre=None, **suppress_kw):
    if centre is None:
        centre = next((est[t] for t in ('edge_binary',) + tuple(trackers)
                       if t in est and np.all(np.isfinite(est[t]))), None)
    if centre is None:
        return None
    clean, desc = suppress_contamination(dp, ref, centre[0], centre[1], **suppress_kw)
    cl, imp, ms = {}, {}, {}
    for t in trackers:
        t0 = time.perf_counter()
        try:
            cl[t] = np.array(track(clean, t, ref, upsample=upsample), float)
        except Exception:
            cl[t] = np.array([np.nan, np.nan])
        ms[t] = 1e3 * (time.perf_counter() - t0)
        imp[t] = np.asarray(est[t], float) - cl[t] if t in est else np.array([np.nan, np.nan])
    vals = [v for v in cl.values() if np.all(np.isfinite(v))]
    spread = max((np.hypot(*(a - b)) for i, a in enumerate(vals) for b in vals[i + 1:]), default=0.0)
    return dict(impact=imp, clean=cl, spread=float(spread), time_ms=ms, **desc)


def impact_baseline(vac_patterns, ref, trackers=('edge_binary', 'ellipse'), upsample=20):
    rows = {t: [] for t in trackers}
    for p in vac_patterns:
        est = {t: np.array(track(p, t, ref, upsample=upsample), float) for t in trackers}
        ci = contamination_impact(p, ref, est, trackers, upsample)
        for t in trackers:
            rows[t].append(np.hypot(*ci['impact'][t]))
    return {t: (float(np.nanmedian(v)), float(np.nanpercentile(v, 95))) for t, v in rows.items()}


RULES = """
How the tracker is chosen (v1.1).
  1. Contamination impact.  For every pattern, suppress_contamination() builds a copy in which
     overlapping secondary (diffracted) discs [additive intensity at the edge] and Kikuchi / dark
     bands [smooth multiplicative modulation] are removed without moving the edge.  Each tracker
     is run on both; impact = estimate(original) - estimate(cleaned).
  2. Acceptance.  The impact is smoothed like the charge maps (MAP_SIGMA scan px, scan median
     removed).  A position passes if it stays below IMPACT_TOL px or below REL_TOL x the local
     (smoothed, plane-removed) deflection, whichever is larger: small errors on a strong signal
     are tolerated, on a weak signal only small absolute errors.  A tracker is ACCEPTABLE if no
     more than MAX_FRACTION of the positions fail.
  3. Choice.  Among the acceptable trackers, the first in PREFERENCE is recommended.  Default
     order: edge_binary (the default of the pipeline; tolerates a fragmented edge, needs a
     vacuum template recorded at the same optics), then ellipse (no template; fails abruptly
     once diffracted intensity or dark bands reach the edge).  Both follow nanometre-scale
     fields about equally (SI Figure S6).  Computing times are reported for information.
  * spread = disagreement between the two trackers on the CLEANED patterns.  Large spread means
    something the model cannot remove (coherent zone-axis overlaps, too low dose): all trackers
    are uncertain there and those positions are flagged.
  * electrons per pattern < 3e4: edge_binary / ellipse unreliable (benchmark).
  * |median d_radius| > 0.5 px: vacuum template does not match the specimen optics; rebuild it.
"""

PREFERENCE = ('edge_binary', 'ellipse')


def recommend(impact_maps, spread=None, electrons_per_pattern=None, d_radius=None, tol=0.05, sigma=3.0,
              max_fraction=0.01, spread_tol=0.3, signal=None, rel_tol=0.10, preference=PREFERENCE, timing=None):
    msgs, smoothed, stats = [], {}, {}
    if electrons_per_pattern is not None and electrons_per_pattern < 3e4:
        msgs.append(f'dose {electrons_per_pattern:.2g} e-/pattern < 3e4: edge_binary / ellipse unreliable, increase the dose (dwell time)')
    if d_radius is not None and np.isfinite(np.nanmedian(d_radius)) and abs(np.nanmedian(d_radius)) > 0.5:
        msgs.append(f'median edge-radius mismatch {np.nanmedian(d_radius):+.2f} px: rebuild the vacuum template')
    allowed = tol if signal is None else np.maximum(tol, rel_tol * np.nan_to_num(signal))
    for t, im in impact_maps.items():
        comp = []
        for k in (0, 1):
            c = im[..., k] - np.nanmedian(im[..., k])
            c = np.where(np.isfinite(c), c, 0.0)
            comp.append(gaussian_filter(c, sigma) if sigma else c)
        sm = np.hypot(*comp); raw = np.hypot(im[..., 0], im[..., 1])
        smoothed[t] = sm
        stats[t] = dict(fail=float(np.mean(sm > allowed)), abs_fail=float(np.mean(sm > tol)),
                        p99=float(np.nanpercentile(sm, 99)), med=float(np.nanmedian(raw)), p95=float(np.nanpercentile(raw, 95)))
        tm = f'; {timing[t]:.0f} ms/pattern' if timing and t in timing else ''
        msgs.append(f'{t:14s} contamination impact (smoothed): fails at {100*stats[t]["fail"]:.1f} % of positions '
                    f'(> {tol:g} px and > {100*rel_tol:.0f} % of the local deflection; > {tol:g} px alone: '
                    f'{100*stats[t]["abs_fail"]:.1f} %), 99th pct {stats[t]["p99"]:.3f} px; per pattern median '
                    f'{stats[t]["med"]:.3f}, 95th pct {stats[t]["p95"]:.3f} px{tm}')
    order = [t for t in preference if t in stats] + [t for t in stats if t not in preference]
    ok = [t for t in order if stats[t]['fail'] <= max_fraction]
    if ok:
        others = [t for t in ok[1:]]
        choice = (f'recommended: {ok[0]}' + (f' (also acceptable: {", ".join(others)})' if others else '')
                  + f'; acceptable = contamination impact within tolerance at >= {100*(1-max_fraction):.0f} % of positions, '
                  'order = PREFERENCE')
    else:
        best = min(stats, key=lambda t: stats[t]['fail'])
        choice = (f'no tracker is within tolerance everywhere; least affected: {best} (fails at '
                  f'{100*stats[best]["fail"]:.1f} % of positions; mask them, maps saved as *_impact_{best}.npy)')
    if spread is not None:
        f = float(np.nanmean(spread > spread_tol))
        msgs.append(f'trackers disagree by > {spread_tol:g} px on the cleaned patterns at {100*f:.1f} % of positions '
                    + ('(contamination the model cannot remove, e.g. coherent overlaps: all trackers uncertain there)'
                       if f > max_fraction else '(none of note)'))
    msgs.append(f'-> {choice}')
    return choice, msgs, smoothed

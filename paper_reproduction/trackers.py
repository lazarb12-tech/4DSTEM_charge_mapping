"""The two disc trackers as released in v1.0 (CentreFitting_xcorr.py, CentreFitting_physical.py) and the per-pattern
QC metrics, for the simulations. Shifts are the disc motion (dx, dy) in detector pixels."""
import warnings
import numpy as np
from skimage import filters, feature, morphology
from skimage.registration import phase_cross_correlation

_DISK3 = morphology.disk(3)
_DISK5 = morphology.disk(5)


def get_disk_edge(dp):
    dp_med = filters.median(dp.astype(np.float32), _DISK3)
    thresh = filters.threshold_otsu(dp_med)
    dp_bin = dp_med > thresh
    dp_closed = morphology.closing(dp_bin, _DISK5)
    return feature.canny(dp_closed.astype(float), sigma=1).astype(float)


def build_template_edge(vac_patterns):
    return np.mean([get_disk_edge(p) for p in vac_patterns], axis=0)


def track_xcorr(dp, template_edge, edge=None):
    if edge is None:
        edge = get_disk_edge(dp)
    shift, err, _ = phase_cross_correlation(template_edge, edge, upsample_factor=20)
    dy, dx = shift
    return -dx, -dy, err


def _ellipse_hf(xy):
    data = xy.astype(float)
    if len(data) < 5:
        return None
    origin = data.mean(axis=0)
    data = data - origin
    scale = data.std()
    if scale < np.finfo(float).tiny:
        return None
    data = data / scale
    x, y = data[:, 0], data[:, 1]
    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones_like(x)]).T
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    C1 = np.array([[0., 0., 2.], [0., -1., 0.], [2., 0., 0.]])
    try:
        M = np.linalg.inv(C1) @ (S1 - S2 @ np.linalg.inv(S3) @ S2.T)
    except np.linalg.LinAlgError:
        return None
    ev, evec = np.linalg.eig(M)
    evec = np.real(evec)
    cond = 4*evec[0, :]*evec[2, :] - evec[1, :]**2
    a1 = evec[:, cond > 0]
    if 0 in a1.shape or a1.size != 3:
        return None
    a, b, c = a1.ravel()
    d, f, g = (-np.linalg.inv(S3) @ S2.T @ a1).ravel()
    b, d, f = b/2., d/2., f/2.
    den = b**2 - a*c
    x0 = (c*d - b*f)/den
    y0 = (a*f - b*d)/den
    num = a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g
    term = np.sqrt((a - c)**2 + 4*b**2)
    with np.errstate(invalid="ignore"):
        w = np.sqrt(2*num/(den*(term - (a + c))))
        h = np.sqrt(2*num/(den*(-term - (a + c))))
    if not (np.isfinite(w) and np.isfinite(h)):
        return None
    if w < h:
        w, h = h, w
    return x0*scale + origin[0], y0*scale + origin[1], w*scale, h*scale


def fit_ellipse(dp, edge=None):
    if edge is None:
        edge = get_disk_edge(dp)
    pts = np.column_stack(np.where(edge > 0))
    if len(pts) < 5:
        return np.nan, np.nan, np.nan, np.nan
    out = _ellipse_hf(pts[:, ::-1].astype(float))
    if out is None:
        return np.nan, np.nan, np.nan, np.nan
    return out


def com(dp, window=None):
    w = dp.astype(float) if window is None else dp.astype(float) * window
    s = w.sum()
    yy, xx = np.indices(dp.shape)
    return (xx * w).sum() / s, (yy * w).sum() / s


def qc_metrics(edge, xc, yc, r0, tol=2.0, nbins=180):
    ys, xs = np.nonzero(edge > 0)
    if xs.size == 0:
        return dict(inlier_frac=0.0, coverage=0.0, n_edge_rel=0.0)
    r = np.hypot(xs - xc, ys - yc)
    near = np.abs(r - r0) <= tol
    inl = near.mean()
    ang = np.arctan2(ys[near] - yc, xs[near] - xc)
    bins = np.floor((ang + np.pi) / (2 * np.pi) * nbins).astype(int) % nbins
    cov = np.unique(bins).size / nbins
    return dict(inlier_frac=float(inl), coverage=float(cov),
                n_edge_rel=float(xs.size / (2 * np.pi * r0)))


from scipy.ndimage import gaussian_filter


def edge_contamination(dp, xc, yc, r0, band=(3.0, 8.0), nbins=36):
    yy, xx = np.indices(dp.shape)
    r = np.hypot(xx - xc, yy - yc)
    ang = np.arctan2(yy - yc, xx - xc)
    inner = (r > r0 - band[1]) & (r < r0 - band[0])
    outer = (r > r0 + band[0]) & (r < r0 + band[1])
    lvl = np.median(dp[inner]) if inner.any() else np.nan
    bins = np.floor((ang + np.pi) / (2*np.pi) * nbins).astype(int) % nbins
    vals = []
    for b in range(nbins):
        m = outer & (bins == b)
        if m.sum() > 5:
            vals.append(dp[m].mean())
    vals = np.array(vals) / lvl if lvl and np.isfinite(lvl) and lvl > 0 else np.array([np.nan])
    return float(np.nanmax(vals)), float(np.nanmin(vals))


def edge_radius(edge, xc, yc, r0, win=6.0):
    ys, xs = np.nonzero(edge > 0)
    if xs.size == 0:
        return np.nan
    r = np.hypot(xs - xc, ys - yc)
    m = np.abs(r - r0) <= win
    return float(np.median(r[m])) if m.any() else np.nan

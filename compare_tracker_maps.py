"""Optional, run after tracker_qc_report.py: the charge map from each tracker (processed as in CentreFitting_xcorr.py),
their differences and radial profiles. Output: results/<FILE_STEM>/tracker_comparison/."""
import os
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

# ----------------------------------------------------------------- SETTINGS
FILE_STEM = "FILENAME"
PROJECT_ROOT = Path(os.environ.get("FOURDSTEM_ROOT", Path(__file__).resolve().parent))  # or edit this path
TRACKERS = ["edge_binary", "ellipse", "com_bf"]   # those run in tracker_qc_report.py
REFERENCE = "edge_binary"           # differences are shown relative to this tracker
ROTATION_ANGLE_DEG = 186.0          # same value as in CentreFitting_xcorr.py
PLANE_SUBTRACT = True               # apply the stored descan plane (results/descan_plane_coeffs.npz)
SMART_REF_SIZE = 5
SIGMA = 3.0                         # display smoothing (scan px), as in the paper
BEAM_ENERGY_KV = 80
CAMERA_LENGTH_MM = 295
DETECTOR_PIXEL_UM = 55
PIXEL_SIZE_ANG = None               # scan step (A); None = read from data/<FILE_STEM>.h5 as in CentreFitting_xcorr.py
PIXEL_SIZE_ANG_MANUAL = 1.0         # fallback if the .h5 carries no real-space calibration
PARTICLE_CENTRE = None              # (row, col) in scan px, or None = centroid of the most negative 10 %
ELLIPSE_MEDIAN3 = True              # 3x3 median filter of the ellipse maps, as CentreFitting_physical.py
SHOW_CONTAMINATION_IMPACT = True    # use the *_clean_* maps of tracker_qc_report.py if present
# --------------------------------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "results"
QC_DIR = RESULTS_DIR / FILE_STEM / "qc"
OUT_DIR = RESULTS_DIR / FILE_STEM / "tracker_comparison"


def prefactor_e_nm2_per_rad_nm(kv=BEAM_ENERGY_KV):
    c, m0, e, eps0 = 2.998e8, 9.109e-31, 1.602e-19, 8.854e-12
    gamma = 1 + kv * 1e3 * e / (m0 * c ** 2)
    v = np.sqrt(1 - 1 / gamma ** 2) * c
    return eps0 * (gamma * m0 * v ** 2 / e) * 1e9 * (1e-18 / e)


def subtract_plane(data, coeffs):
    ny, nx = data.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    A = np.stack([xx.ravel(), yy.ravel(), np.ones(nx * ny)], axis=1)
    return data - (A @ coeffs).reshape(ny, nx)


def smart_reference(dx, dy, size=SMART_REF_SIZE):
    half = size // 2; best = np.inf; rc = None
    for r in range(half + 1, dx.shape[0] - half - 1):
        for c in range(half + 1, dx.shape[1] - half - 1):
            px = dx[r - half:r + half + 1, c - half:c + half + 1]; py = dy[r - half:r + half + 1, c - half:c + half + 1]
            if np.isnan(px).any() or np.isnan(py).any():
                continue
            s = np.var(px) + np.var(py)
            if s < best:
                best, rc = s, (r, c)
    r, c = rc
    return np.nanmean(dx[r - half:r + half + 1, c - half:c + half + 1]), np.nanmean(dy[r - half:r + half + 1, c - half:c + half + 1])


def charge_map(sx, sy, coeffs=None):
    th = np.deg2rad(ROTATION_ANGLE_DEG)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    d = R @ np.stack([sx.ravel(), sy.ravel()])
    dx, dy = d[0].reshape(sx.shape), d[1].reshape(sx.shape)
    if coeffs is not None:
        dx = subtract_plane(dx, coeffs['coeffs_x']); dy = subtract_plane(dy, coeffs['coeffs_y'])
    rx, ry = smart_reference(dx, dy)
    dx, dy = dx - rx, dy - ry
    dx = np.where(np.isfinite(dx), dx, np.nanmedian(dx)); dy = np.where(np.isfinite(dy), dy, np.nanmedian(dy))
    k = DETECTOR_PIXEL_UM * 1e-6 / (CAMERA_LENGTH_MM * 1e-3)
    pix_nm = PIXEL_SIZE_ANG * 0.1
    _, dax_dx = np.gradient(dx * k, pix_nm)
    day_dy, _ = np.gradient(dy * k, pix_nm)
    s = (dax_dx + day_dy) * prefactor_e_nm2_per_rad_nm()
    return s, gaussian_filter(s, SIGMA), dx * k, dy * k


def radial_profile(m, centre, pix_nm, rmax_nm=6.0, dr_nm=0.1):
    yy, xx = np.indices(m.shape)
    r = np.hypot(yy - centre[0], xx - centre[1]) * pix_nm
    edges = np.arange(0, rmax_nm + dr_nm, dr_nm)
    prof = np.array([np.nanmean(m[(r >= a) & (r < b)]) if ((r >= a) & (r < b)).any() else np.nan
                     for a, b in zip(edges[:-1], edges[1:])])
    return 0.5 * (edges[:-1] + edges[1:]), prof


def read_pixel_size_ang():
    import py4DSTEM
    dc = py4DSTEM.read(str(PROJECT_ROOT / "data" / f"{FILE_STEM}.h5"))
    size, units = dc.calibration.get_R_pixel_size(), dc.calibration.get_R_pixel_units()
    if units in ('nm', 'NM'):
        return size * 10
    if units in ('A', 'Å', 'angstrom', 'Angstrom'):
        return size
    print(f"No usable R-pixel calibration in the .h5 (units '{units}'): using PIXEL_SIZE_ANG_MANUAL")
    return PIXEL_SIZE_ANG_MANUAL


def main():
    global PIXEL_SIZE_ANG
    if PIXEL_SIZE_ANG is None:
        PIXEL_SIZE_ANG = read_pixel_size_ang()
    print(f"Scan pixel size: {PIXEL_SIZE_ANG} A")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coeffs = None
    cf = RESULTS_DIR / "descan_plane_coeffs.npz"
    if PLANE_SUBTRACT:
        if not cf.exists():
            raise FileNotFoundError(f"{cf} not found: run CentreFitting_xcorr.py in 'measure' mode on the vacuum first, "
                                    f"or set PLANE_SUBTRACT = False")
        coeffs = np.load(cf)
    def load(kind, t):
        fx, fy = QC_DIR / f"{FILE_STEM}_{kind}_{t}_x.npy", QC_DIR / f"{FILE_STEM}_{kind}_{t}_y.npy"
        if not (fx.exists() and fy.exists()):
            return None
        sx, sy = np.load(fx), np.load(fy)
        if t == 'ellipse' and ELLIPSE_MEDIAN3:
            sx, sy = median_filter(sx, size=3), median_filter(sy, size=3)
        return sx, sy

    maps, cleaned = {}, {}
    for t in TRACKERS:
        sx, sy = load("shift", t)
        maps[t] = charge_map(sx, sy, coeffs)
        np.save(OUT_DIR / f"{FILE_STEM}_sigma_{t}.npy", maps[t][0])
        c = load("clean", t) if SHOW_CONTAMINATION_IMPACT else None
        if c is not None:
            cleaned[t] = charge_map(c[0], c[1], coeffs)
            np.save(OUT_DIR / f"{FILE_STEM}_sigma_{t}_cleaned.npy", cleaned[t][0])
    pix_nm = PIXEL_SIZE_ANG * 0.1
    ref = maps[REFERENCE][1]
    if PARTICLE_CENTRE is None:
        neg = ref < np.nanpercentile(ref, 10)
        yy, xx = np.indices(ref.shape)
        centre = ((yy * neg).sum() / neg.sum(), (xx * neg).sum() / neg.sum())
    else:
        centre = PARTICLE_CENTRE
    scale_from = [t for t in maps if t != 'com_bf'] or list(maps)
    vlim = np.nanpercentile(np.abs(np.concatenate([maps[t][1].ravel() for t in scale_from])), 99.5)
    n = len(TRACKERS)
    nrow = 3 if cleaned else 2
    fig, axs = plt.subplots(nrow, n + 1, figsize=(4.4 * (n + 1), 3.8 * nrow), gridspec_kw=dict(wspace=0.45))
    lines = []
    for i, t in enumerate(TRACKERS):
        sb = maps[t][1]
        im = axs[0, i].imshow(sb, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
        axs[0, i].set_title(f"{t}  (sigma_code, e nm$^{{-2}}$)"); axs[0, i].axis('off')
        axs[0, i].plot(centre[1], centre[0], 'k+', ms=8)
        if t != REFERENCE:
            dlim = vlim / 2
            axs[1, i].imshow(sb - ref, cmap='PuOr_r', vmin=-dlim, vmax=dlim)
            axs[1, i].set_title(f"{t} - {REFERENCE}"); axs[1, i].axis('off')
        else:
            axs[1, i].axis('off')
        r, p = radial_profile(sb, centre, pix_nm)
        axs[0, n].plot(r, p, label=t)
        sign_change = r[np.where((p[:-1] < 0) & (p[1:] >= 0))[0][0]] if np.any((p[:-1] < 0) & (p[1:] >= 0)) else np.nan
        interior = np.nanmean(p[r < 0.5]); rim = np.nanmax(p[(r > 0.5)])
        lines.append(f"{t:14s} interior (r<0.5 nm) {interior:+.2f}  rim max {rim:+.2f} e/nm2  "
                     f"sign change {sign_change:.2f} nm")
        if t in cleaned:
            rc, pc = radial_profile(cleaned[t][1], centre, pix_nm)
            axs[0, n].plot(rc, pc, ls='--', lw=1, color=axs[0, n].lines[-1].get_color())
            dsig = sb - cleaned[t][1]
            axs[2, i].imshow(dsig, cmap='RdBu_r', vmin=-vlim / 2, vmax=vlim / 2)
            axs[2, i].set_title(f"{t}: sigma - sigma(cleaned)\n= effect of the contamination"); axs[2, i].axis('off')
            sc = r[np.where((pc[:-1] < 0) & (pc[1:] >= 0))[0][0]] if np.any((pc[:-1] < 0) & (pc[1:] >= 0)) else np.nan
            imf = QC_DIR / f"{FILE_STEM}_impact_{t}.npy"
            quiet = (np.load(imf) < 0.05) if imf.exists() else np.ones(dsig.shape, bool)
            quiet = quiet if (quiet.shape == dsig.shape and quiet.sum() > 20) else np.ones(dsig.shape, bool)
            lines.append(f"{'  cleaned':14s} interior (r<0.5 nm) {np.nanmean(pc[rc < 0.5]):+.2f}  rim max "
                         f"{np.nanmax(pc[rc > 0.5]):+.2f} e/nm2  sign change {sc:.2f} nm  -> contamination changes "
                         f"interior by {interior - np.nanmean(pc[rc < 0.5]):+.2f}, rim by {rim - np.nanmax(pc[rc > 0.5]):+.2f} e/nm2 "
                         f"(noise of this difference {np.nanstd(dsig[quiet]):.2f})")
        elif cleaned:
            axs[2, i].axis('off')
    fig.colorbar(im, ax=axs[0, n - 1], fraction=0.046, pad=0.02)
    axs[0, n].axhline(0, color='0.6', lw=0.7); axs[0, n].set_xlabel('r from particle centre (nm)')
    axs[0, n].set_ylabel('azimuthal mean sigma_code (e nm$^{-2}$)'); axs[0, n].legend(fontsize=8)
    axs[0, n].set_title('Radial profiles' + ('  (dashed: contamination removed)' if cleaned else ''))
    if cleaned:
        axs[2, n].axis('off')
    axs[1, n].axis('off')
    axs[1, n].text(0, 1, '\n'.join(lines), va='top', family='monospace', fontsize=7, transform=axs[1, n].transAxes)
    plt.savefig(OUT_DIR / f"{FILE_STEM}_tracker_comparison.png", dpi=150, bbox_inches='tight'); plt.close()
    (OUT_DIR / f"{FILE_STEM}_tracker_comparison.txt").write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f"Saved to {OUT_DIR}")


if __name__ == '__main__':
    main()

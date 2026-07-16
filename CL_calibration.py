import os
from pathlib import Path
import py4DSTEM
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import median_filter, gaussian_filter, maximum_filter
from skimage.feature import peak_local_max

FILE_STEM        = "Au_C_1"          # h5 file stem in data dir
PROJECT_ROOT     = Path(os.environ.get("FOURDSTEM_ROOT", Path(__file__).resolve().parent))  # set the FOURDSTEM_ROOT env var, or edit this path, to point at your project folder

BEAM_ENERGY_KV   = 80           # accelerating voltage
DETECTOR_PIXEL_UM = 55          # Medipix pixel size, microns
CAMERA_LENGTH_NOMINAL_MM = 200  # alleged value, for comparison only

D_SPACING_ANG    = 2.3545       # Au d111 = 2.3545 A  (d200=2.0391, d220=1.4419)
DISK_RADIUS_PX   = None         # central disk radius in px; None = auto-estimate

MEAN_REGION      = None         # None = average whole scan; or (rs,re,cs,ce)
ENHANCE          = "log"        # "log" / "sqrt" / "none" — boost weak diffracted disks

# The diffracted disks are diluted by averaging the whole scan, because they
# only appear at scan positions where a grain diffracts. Two ways to recover them:
DIFF_PATTERN_METHOD = "selective" # "variance" = std/variance projection (flickering disks)
                                  # "selective" = sum only the frames that diffract
SELECTIVE_PERCENTILE = 95         # for "selective": keep top (100-this)% diffracting frames

N_DISKS_EXPECTED = 6           # detect up to this many diffracted disks (no symmetry assumed)
# Candidate d-spacings to test each disk against (Angstrom)
D_CANDIDATES = {"111": 2.3545, "200": 2.0391, "220": 1.4419}

DATA_DIR = PROJECT_ROOT / "data"
h5_path  = DATA_DIR / f"{FILE_STEM}.h5"

h_planck = 6.62607015e-34
m0       = 9.1093837015e-31
e_C      = 1.602176634e-19
c_light  = 2.99792458e8
V        = BEAM_ENERGY_KV * 1e3
lam_m    = h_planck / np.sqrt(2*m0*e_C*V*(1 + e_C*V/(2*m0*c_light**2)))
lam_ang  = lam_m * 1e10
print(f"Relativistic wavelength at {BEAM_ENERGY_KV} kV: {lam_m*1e12:.4f} pm")

dc      = py4DSTEM.read(str(h5_path))
Rx, Ry  = dc.Rshape
Qx, Qy  = dc.Qshape
print(f"Scan: {Rx}x{Ry}, detector: {Qx}x{Qy}")

if MEAN_REGION is None:
    mean_dp = dc.data.mean(axis=(0, 1)).astype(float)
    print("Mean pattern: averaged over full scan")
else:
    rs, re, cs, ce = MEAN_REGION
    mean_dp = dc.data[rs:re, cs:ce].mean(axis=(0, 1)).astype(float)
    print(f"Mean pattern: averaged over [{rs}:{re}, {cs}:{ce}]")

mean_dp = median_filter(mean_dp, size=3)
if ENHANCE == "log":
    enh = np.log1p(mean_dp - mean_dp.min())
elif ENHANCE == "sqrt":
    enh = np.sqrt(mean_dp - mean_dp.min())
else:
    enh = mean_dp.copy()
enh_blur = gaussian_filter(enh, sigma=2)

if DISK_RADIUS_PX is None:
    bright   = mean_dp > (0.5 * mean_dp.max())
    area_px  = bright.sum()
    DISK_RADIUS_PX = np.sqrt(area_px / np.pi)
    print(f"Auto disk radius estimate: {DISK_RADIUS_PX:.1f} px")
else:
    print(f"Disk radius (set): {DISK_RADIUS_PX:.1f} px")

_t = int(np.ceil(DISK_RADIUS_PX * 2.5))
_yy, _xx = np.mgrid[-_t:_t+1, -_t:_t+1]
template = (np.sqrt(_xx**2 + _yy**2) <= DISK_RADIUS_PX).astype(float)
template -= template.mean()

from scipy.signal import fftconvolve

cc_central = fftconvolve(enh_blur - enh_blur.mean(),
                         template[::-1, ::-1], mode='same')
cy0, cx0 = np.unravel_index(np.argmax(cc_central), cc_central.shape)
print(f"Central (brightest) disk at: x={cx0}, y={cy0}")

data_f = dc.data.astype(float)

if DIFF_PATTERN_METHOD == "variance":
    diff_dp = data_f.var(axis=(0, 1))
    print("Diffracted-disk pattern: variance projection")

elif DIFF_PATTERN_METHOD == "selective":
    yy, xx = np.mgrid[0:Qx, 0:Qy]
    outside = np.sqrt((xx - cx0)**2 + (yy - cy0)**2) > (DISK_RADIUS_PX * 1.5)
    frame_scores = (data_f * outside).sum(axis=(2, 3))
    thr  = np.percentile(frame_scores, SELECTIVE_PERCENTILE)
    sel  = frame_scores > thr
    diff_dp = data_f[sel].sum(axis=0)
    print(f"Diffracted-disk pattern: summed {sel.sum()} / {Rx*Ry} "
          f"diffracting frames (top {100-SELECTIVE_PERCENTILE}%)")
else:
    raise ValueError(f"DIFF_PATTERN_METHOD must be 'variance' or 'selective'")

diff_dp = median_filter(diff_dp, size=3)

yy_r, xx_r = np.mgrid[0:Qx, 0:Qy]
r_int       = np.sqrt((xx_r - cx0)**2 + (yy_r - cy0)**2).astype(int)
radial_mean = np.bincount(r_int.ravel(), diff_dp.ravel()) / np.bincount(r_int.ravel())
background   = radial_mean[r_int]
diff_dp      = diff_dp - background
diff_dp[diff_dp < 0] = 0
print("Subtracted radial background (amorphous ring suppression)")

yy, xx = np.mgrid[0:Qx, 0:Qy]
central_mask = np.sqrt((xx - cx0)**2 + (yy - cy0)**2) < (DISK_RADIUS_PX * 1.3)
diff_dp_dimmed = diff_dp.copy()
diff_dp_dimmed[central_mask] = np.median(diff_dp)

if ENHANCE == "log":
    diff_enh = np.log1p(diff_dp_dimmed - diff_dp_dimmed.min())
elif ENHANCE == "sqrt":
    diff_enh = np.sqrt(diff_dp_dimmed - diff_dp_dimmed.min())
else:
    diff_enh = diff_dp_dimmed.copy()
diff_enh_blur = gaussian_filter(diff_enh, sigma=2)

cc = fftconvolve(diff_enh_blur - diff_enh_blur.mean(),
                 template[::-1, ::-1], mode='same')

exclude_central = np.sqrt((xx - cx0)**2 + (yy - cy0)**2) < (DISK_RADIUS_PX * 2.0)
cc_search = cc.copy()
cc_search[exclude_central] = cc.min()

min_dist = max(int(DISK_RADIUS_PX * 1.5), 5)
peaks = peak_local_max(
    cc_search,
    min_distance=min_dist,
    threshold_rel=0.05,
    num_peaks=N_DISKS_EXPECTED + 5,
    exclude_border=False
)


dists_to_centre = np.sqrt((peaks[:,1]-cx0)**2 + (peaks[:,0]-cy0)**2)
diff_peaks = peaks[dists_to_centre > min_dist]
diff_dists = dists_to_centre[dists_to_centre > min_dist]

order = np.argsort(diff_dists)
diff_peaks = diff_peaks[order]
diff_dists = diff_dists[order]

print(f"\nFound {len(diff_peaks)} candidate diffracted disks")
print(f"{'#':<4}{'x':>6}{'y':>6}{'R (px)':>10}")
for i, ((py, px), R) in enumerate(zip(diff_peaks, diff_dists)):
    print(f"{i:<4}{px:>6}{py:>6}{R:>10.2f}")

R_refined = None
if len(diff_peaks) > 0:
    py0, px0 = diff_peaks[0]
    win = int(DISK_RADIUS_PX)
    y_lo, y_hi = max(0, py0-win), min(Qx, py0+win+1)
    x_lo, x_hi = max(0, px0-win), min(Qy, px0+win+1)
    sub = diff_dp[y_lo:y_hi, x_lo:x_hi]
    _sy, _sx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    w = sub - sub.min()
    if w.sum() > 0:
        cy_ref = y_lo + np.average(_sy, weights=w)
        cx_ref = x_lo + np.average(_sx, weights=w)
        R_refined = np.sqrt((cx_ref - cx0)**2 + (cy_ref - cy0)**2)
        print(f"\nDisk 0 sub-pixel centre: ({cx_ref:.2f}, {cy_ref:.2f})")
        print(f"Disk 0 R: {diff_dists[0]:.2f} px (peak) -> "
              f"{R_refined:.3f} px (refined)")

det_px_m = DETECTOR_PIXEL_UM * 1e-6

def CL_from(R_px, d_ang):
    return (R_px * det_px_m) * (d_ang * 1e-10) / lam_m * 1e3

if len(diff_dists) > 0:
    print(f"\n=== Per-disk CL under each candidate d-spacing ===")
    header = f"{'disk':<6}{'R(px)':>9}" + "".join(f"{'CL_'+k+'(mm)':>13}" for k in D_CANDIDATES)
    print(header)
    print("-" * len(header))
    for i, R in enumerate(diff_dists):
        row = f"{i:<6}{R:>9.2f}"
        for k, d in D_CANDIDATES.items():
            row += f"{CL_from(R, d):>13.1f}"
        print(row)

    print(f"\n  (Each column assumes that disk is the given reflection.)")
    print(f"  Identify the correct d by which CL matches your nominal "
          f"~{CAMERA_LENGTH_NOMINAL_MM} mm,")
    print(f"  or by consistency across disks if you have more than one.")

    R_primary = R_refined if R_refined is not None else diff_dists[0]
    CL_primary = CL_from(R_primary, D_SPACING_ANG)
    print(f"\n=== Primary result (closest disk as d={D_SPACING_ANG} A) ===")
    print(f"  R = {R_primary:.3f} px = {R_primary*det_px_m*1e3:.4f} mm on detector"
          f"{' (sub-pixel refined)' if R_refined is not None else ''}")
    print(f"  lambda: {lam_m*1e12:.4f} pm")
    print(f"  >>> Calibrated CL = {CL_primary:.1f} mm")
    print(f"  >>> Nominal CL    = {CAMERA_LENGTH_NOMINAL_MM} mm")
    print(f"  >>> Ratio         = {CL_primary/CAMERA_LENGTH_NOMINAL_MM:.3f}")

    if len(diff_dists) >= 2:
        print(f"\n=== All-disk consistency check ===")
        CL_TOL_FRAC = 0.05

        per_disk_CL = [
            {k: CL_from(R, d) for k, d in D_CANDIDATES.items()}
            for R in diff_dists
        ]

        all_CLs = np.array([cl for d in per_disk_CL for cl in d.values()])
        trial_CLs = np.linspace(all_CLs.min(), all_CLs.max(), 400)

        best = None
        for trial in trial_CLs:
            assignment = {}
            inlier_CLs = []
            for i, cl_dict in enumerate(per_disk_CL):
                k_best, cl_best = min(cl_dict.items(),
                                      key=lambda kv: abs(kv[1] - trial))
                if abs(cl_best - trial) / trial < CL_TOL_FRAC:
                    assignment[i] = (k_best, cl_best)
                    inlier_CLs.append(cl_best)
            n_in = len(inlier_CLs)
            if n_in >= 2:
                spread = np.std(inlier_CLs)
                score  = (n_in, -spread)
                if best is None or score > best[0]:
                    best = (score, np.mean(inlier_CLs), dict(assignment), n_in)

        if best is not None:
            _, CL_consensus, assignment, n_in = best
            print(f"  Best consensus: {n_in}/{len(diff_dists)} disks agree")
            print(f"  >>> Consensus CL = {CL_consensus:.1f} mm")
            print(f"  Disk assignments:")
            print(f"    {'disk':<6}{'R(px)':>9}{'reflection':>12}{'CL(mm)':>10}")
            for i in sorted(assignment):
                k, cl = assignment[i]
                print(f"    {i:<6}{diff_dists[i]:>9.2f}{('d'+k):>12}{cl:>10.1f}")
            outliers = [i for i in range(len(diff_dists)) if i not in assignment]
            if outliers:
                print(f"  Outlier disks (not consistent): {outliers} "
                      f"(likely noise, carbon, or higher orders)")
        else:
            print(f"  No consistent CL found across disks at {CL_TOL_FRAC*100:.0f}% "
                  f"tolerance — disks may be noise or need manual inspection.")
else:
    print("\nNo diffracted disks found — check ENHANCE, threshold_rel, or DISK_RADIUS_PX")

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

axes[0].imshow(enh, cmap='inferno')
axes[0].set_title(f'Mean pattern ({ENHANCE}) — central disk')
axes[0].plot(cx0, cy0, 'c+', markersize=15, markeredgewidth=2, label='central disk')
axes[0].legend(loc='upper right')

axes[1].imshow(diff_enh, cmap='inferno')
axes[1].set_title(f'Diffracted-disk pattern ({DIFF_PATTERN_METHOD}, {ENHANCE})')
axes[1].plot(cx0, cy0, 'c+', markersize=12, markeredgewidth=2)
for i, (py, px) in enumerate(diff_peaks):
    color = 'lime' if i == 0 else 'orange'
    axes[1].add_patch(Circle((px, py), DISK_RADIUS_PX,
                             fill=False, edgecolor=color, linewidth=1.5))
    axes[1].annotate(f"{i}", (px, py), color='white', fontsize=12,
                     ha='center', va='center')

axes[2].imshow(cc, cmap='viridis')
axes[2].set_title('Diffracted-disk cross-correlation')
axes[2].plot(cx0, cy0, 'r+', markersize=15, markeredgewidth=2)
for i, (py, px) in enumerate(diff_peaks):
    color = 'lime' if i == 0 else 'orange'
    axes[2].add_patch(Circle((px, py), DISK_RADIUS_PX,
                             fill=False, edgecolor=color, linewidth=1.5))
    axes[2].annotate(f"{i}", (px, py), color='white', fontsize=12,
                     ha='center', va='center')
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "results" / f"{FILE_STEM}_CL_calibration.png", dpi=150)
plt.show()
print(f"\nSaved diagnostic figure.")

fig_dir = PROJECT_ROOT / "results" / f"{FILE_STEM}_figdata"
fig_dir.mkdir(exist_ok=True)
np.save(fig_dir / "mean_pattern.npy",      enh)
np.save(fig_dir / "selective_pattern.npy", diff_enh)
np.save(fig_dir / "diff_crosscorr.npy",    cc)
np.save(fig_dir / "central_disk_xy.npy",   np.array([cx0, cy0]))
np.save(fig_dir / "disks_table.npy",       diff_peaks[:, ::-1])
print(f"Saved figure arrays to {fig_dir}")

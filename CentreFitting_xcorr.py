from pathlib import Path
import py4DSTEM
import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation
from joblib import Parallel, delayed
from skimage import filters, feature, morphology
import io
import os
from datetime import datetime
from PIL import Image
import tifffile
import math

FILE_STEM           = "FILENAME"
PROJECT_ROOT        = Path(os.environ.get("FOURDSTEM_ROOT", Path(__file__).resolve().parent))  # set the FOURDSTEM_ROOT env var, or edit this path, to point at your project folder
HAADF_PATH          = PROJECT_ROOT / "data" / f"{FILE_STEM}_haadf.tif"
VAC_H5_STEM         = "FILENAME_vacuum"   # h5 stem of the vacuum reference dataset (in data dir)
VAC_H5_PATH         = PROJECT_ROOT / "data" / f"{VAC_H5_STEM}.h5"

PIXEL_SIZE_ANG_MANUAL       = 1.0    # real space pixel size in Angstroms, fallback if not stored in h5
ROTATION_ANGLE_DEG          = 186.0
BEAM_ENERGY_KV              = 80     # accelerating voltage in kV
CAMERA_LENGTH_MM            = 295     # effective camera length in mm
DETECTOR_PIXEL_UM           = 55      # 4D detector pixel size in µm

SMART_REF_SIZE         = 5       # side length of reference patch in pixels
VAC_REF_SIZE           = 2 # half-size of square reference region in "vacuum mode" (e.g. VAC_REF_SIZE=2 → 5×5 region)
PLANE_SUBTRACT         = True   # set to True to perform plane subtraction
PLANE_MODE             = "apply"  # "measure" to fit and save plane coefficients from current data; "apply" to load and apply previously saved coefficients
USE_ARCSIN_CORRECTION  = True # True = exact arcsin chain rule; False = small angle approx

SKIP_FITTING          = False   # set to True to skip fitting and load saved centres from .npy files
SIGMA                 = 3.0     # Gaussian blur sigma for divergence map (in pixels)
CONTRAST_ENHANCE      = False   # set to True to use ±2σ limits for color scaling instead of min/max
N_QUIVER              = 4
OVERLAY_HAADF         = False   # set to True if you have a HAADF image to overlay
ALPHA_OVERLAY         = 0.25    # transparency of divergence overlay on HAADF

DATA_DIR    = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATASET_DIR = RESULTS_DIR / FILE_STEM
DATASET_DIR.mkdir(parents=True, exist_ok=True)
print(f"Results directory: {DATASET_DIR}")

h5_path         = DATA_DIR / f"{FILE_STEM}.h5"
VAC_COEFFS_FILE = RESULTS_DIR / "descan_plane_coeffs.npz"
datacube        = py4DSTEM.read(str(h5_path))
Rx, Ry          = datacube.Rshape

_r_size  = datacube.calibration.get_R_pixel_size()
_r_units = datacube.calibration.get_R_pixel_units()

if (_r_units is not None and _r_units not in ('pixels', 'PIXELS', '')):
    if _r_units in ('nm', 'NM'):
        PIXEL_SIZE_ANG = _r_size * 10
        print(f"Pixel size from h5: {_r_size} nm = {PIXEL_SIZE_ANG} Å")
    elif _r_units in ('A', 'Å', 'angstrom', 'Angstrom'):
        PIXEL_SIZE_ANG = _r_size
        print(f"Pixel size from h5: {PIXEL_SIZE_ANG} Å")
    else:
        print(f"Unknown units '{_r_units}' in h5 — using manual value")
        PIXEL_SIZE_ANG = PIXEL_SIZE_ANG_MANUAL
else:
    PIXEL_SIZE_ANG = PIXEL_SIZE_ANG_MANUAL
    print(f"No calibration in h5 — using manual value: {PIXEL_SIZE_ANG} Å")

print(f"PIXEL_SIZE_ANG = {PIXEL_SIZE_ANG} Å")

def get_clim(data):
    if CONTRAST_ENHANCE:
        v = 2 * data.std()
        return -v, v
    return data.min(), data.max()

def add_scalebar(ax):
    scalebar = ScaleBar(
        PIXEL_SIZE_ANG * 1e-10,
        units='m',
        dimension='si-length',
        fixed_value=1,
        fixed_units='nm',
        location='lower left',
        color='k',
        box_alpha=0,
        font_properties={'size': 10}
    )
    ax.add_artist(scalebar)
    ax.set_axis_off()

def make_extent():
    return [0, Ry * PIXEL_SIZE_ANG, Rx * PIXEL_SIZE_ANG, 0]

def save_tif_colored(data, filename, cmap='RdBu_r', unit='pixels'):
    if CONTRAST_ENHANCE:
        vmin, vmax = -2 * data.std(), 2 * data.std()
    else:
        vmin, vmax = data.min(), data.max()

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=make_extent())
    plt.colorbar(im, ax=ax, label=unit, fraction=0.046, pad=0.04)
    ax.set_title(filename.replace('_', ' '))
    ax.set_xlabel('Å')
    ax.set_ylabel('Å')
    add_scalebar(ax)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    img.save(DATASET_DIR / f"{FILE_STEM}_{filename}.tif")
    print(f"Saved: {FILE_STEM}_{filename}.tif")

print("Building reference template from vacuum smart region...")
_vac_dc    = py4DSTEM.read(str(VAC_H5_PATH))
_vRx, _vRy = _vac_dc.Rshape

_half   = SMART_REF_SIZE // 2
_margin = _half + 1
_best_score = np.inf
_best_r, _best_c = _vRx // 2, _vRy // 2

for _r in range(_margin, _vRx - _margin):
    for _c in range(_margin, _vRy - _margin):
        _patch = np.array([
            _vac_dc.data[_r + dr, _c + dc].astype(float).mean()
            for dr in range(-_half, _half + 1)
            for dc in range(-_half, _half + 1)
        ])
        _score = np.var(_patch)
        if _score < _best_score:
            _best_score = _score
            _best_r, _best_c = _r, _c

_ti, _ti2 = _best_r - _half, _best_r + _half + 1
_tj, _tj2 = _best_c - _half, _best_c + _half + 1

_ref_patterns = [_vac_dc.data[_i, _j].astype(float)
                 for _i in range(_ti, _ti2)
                 for _j in range(_tj, _tj2)]
del _vac_dc
print(f"  Vacuum smart region: centre ({_best_r}, {_best_c}), "
      f"[{_ti}:{_ti2}, {_tj}:{_tj2}], score={_best_score:.6f}")
print(f"  Loaded {len(_ref_patterns)} patterns")

TEMPLATE = np.mean(_ref_patterns, axis=0)

def get_disk_edge(dp):
    dp_med    = filters.median(dp.astype(np.float32), morphology.disk(3))
    thresh    = filters.threshold_otsu(dp_med)
    dp_bin    = (dp_med > thresh)
    dp_closed = morphology.closing(dp_bin, morphology.disk(5))
    return feature.canny(dp_closed.astype(float), sigma=1).astype(float)

TEMPLATE_EDGE = np.mean([get_disk_edge(p) for p in _ref_patterns], axis=0)
print(f"  Edge template built, mean edge density: {TEMPLATE_EDGE.mean():.4f}")

_test_dp = datacube.data[Rx//2, Ry//2].astype(float)
_test_edge = get_disk_edge(_test_dp)
_test_shift, _, _ = phase_cross_correlation(
    TEMPLATE_EDGE, _test_edge,
    upsample_factor=20
)
print(f"  Template vs central sample pattern (edge): "
      f"dy={_test_shift[0]:.3f}, dx={_test_shift[1]:.3f} px")
print(f"  (>5 px shift suggests mismatched camera length or beam position)")

def fit_single_pattern(dp_ij):
    try:
        dp_edge = get_disk_edge(dp_ij)
        shift, _, _ = phase_cross_correlation(
            TEMPLATE_EDGE, dp_edge,
            upsample_factor=20
        )
        dy, dx = shift
        return -dx, -dy, np.nan, np.nan
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

if SKIP_FITTING:
    centres_x = np.load(DATASET_DIR / f"{FILE_STEM}_centres_x.npy")
    centres_y = np.load(DATASET_DIR / f"{FILE_STEM}_centres_y.npy")
    Rx, Ry    = centres_x.shape
    print(f"Loaded saved centres from {DATASET_DIR}")
    print(f"Failed fits: {np.sum(np.isnan(centres_x))} / {Rx * Ry}")
else:
    def fit_at(i, j):
        return fit_single_pattern(datacube.data[i, j])

    index_pairs = [(i, j) for i in range(Rx) for j in range(Ry)]
    print(f"Processing {Rx*Ry} patterns in parallel...")
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(fit_at)(i, j) for i, j in index_pairs
    )
    results   = np.array(results)
    centres_x = results[:, 0].reshape(Rx, Ry)
    centres_y = results[:, 1].reshape(Rx, Ry)
    print(f"Failed fits: {np.sum(np.isnan(centres_x))} / {Rx * Ry}")
    np.save(DATASET_DIR / f"{FILE_STEM}_centres_x.npy", centres_x)
    np.save(DATASET_DIR / f"{FILE_STEM}_centres_y.npy", centres_y)
    print(f"Saved raw fitting results to {DATASET_DIR}")

disp_x = np.where(np.isfinite(centres_x), centres_x, np.nan)
disp_y = np.where(np.isfinite(centres_y), centres_y, np.nan)
print(f"Displacement range: X={np.nanmin(disp_x):.4f}–{np.nanmax(disp_x):.4f} px, "
      f"Y={np.nanmin(disp_y):.4f}–{np.nanmax(disp_y):.4f} px")

theta_rad  = np.deg2rad(ROTATION_ANGLE_DEG)
R          = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                       [np.sin(theta_rad),  np.cos(theta_rad)]])
disp_stack = np.stack([disp_x.ravel(), disp_y.ravel()], axis=0)
disp_rot   = R @ disp_stack
disp_x     = disp_rot[0].reshape(Rx, Ry)
disp_y     = disp_rot[1].reshape(Rx, Ry)

ext = make_extent()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, data, title in zip(axes, [disp_x, disp_y],
                            ['Displacement X (before plane subtraction)',
                             'Displacement Y (before plane subtraction)']):
    vmin, vmax = get_clim(data)
    im = ax.imshow(data, cmap='RdBu_r', vmin=vmin, vmax=vmax, extent=ext)
    plt.colorbar(im, ax=ax, label='pixels')
    ax.set_title(title)
    ax.set_xlabel('Å')
    ax.set_ylabel('Å')
    add_scalebar(ax)
plt.tight_layout()
plt.show()

def subtract_plane(data, coeffs=None):
    ny, nx = data.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    A      = np.stack([xx.ravel(), yy.ravel(), np.ones(nx*ny)], axis=1)
    if coeffs is None:
        coeffs, _, _, _ = lstsq(A, data.ravel(), rcond=None)
    plane = (A @ coeffs).reshape(ny, nx)
    return data - plane, plane, coeffs

if PLANE_SUBTRACT:
    if PLANE_MODE == "measure":
        disp_x, plane_x, coeffs_x = subtract_plane(disp_x)
        disp_y, plane_y, coeffs_y = subtract_plane(disp_y)
        if VAC_COEFFS_FILE.exists():
            VAC_COEFFS_FILE.unlink()
            print(f"Deleted existing coefficients file")
        np.savez(VAC_COEFFS_FILE, coeffs_x=coeffs_x, coeffs_y=coeffs_y)
        mtime = os.path.getmtime(VAC_COEFFS_FILE)
        print(f"Coefficients saved to {VAC_COEFFS_FILE}")
        print(f"File timestamp: {datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"File size: {os.path.getsize(VAC_COEFFS_FILE)} bytes")
        print(f"X descan plane: {coeffs_x[0]:.6f}*x + {coeffs_x[1]:.6f}*y + {coeffs_x[2]:.6f}")
        print(f"Y descan plane: {coeffs_y[0]:.6f}*x + {coeffs_y[1]:.6f}*y + {coeffs_y[2]:.6f}")
    elif PLANE_MODE == "apply":
        if not VAC_COEFFS_FILE.exists():
            raise FileNotFoundError(f"No saved coefficients found at {VAC_COEFFS_FILE}.")
        saved    = np.load(VAC_COEFFS_FILE)
        coeffs_x = saved['coeffs_x']
        coeffs_y = saved['coeffs_y']
        print(f"Loaded coefficients from {VAC_COEFFS_FILE}")
        print(f"X descan plane: {coeffs_x[0]:.6f}*x + {coeffs_x[1]:.6f}*y + {coeffs_x[2]:.6f}")
        print(f"Y descan plane: {coeffs_y[0]:.6f}*x + {coeffs_y[1]:.6f}*y + {coeffs_y[2]:.6f}")
        disp_x, plane_x, coeffs_x = subtract_plane(disp_x, coeffs=coeffs_x)
        disp_y, plane_y, coeffs_y = subtract_plane(disp_y, coeffs=coeffs_y)
else:
    plane_x  = np.zeros_like(disp_x)
    plane_y  = np.zeros_like(disp_y)
    coeffs_x = np.zeros(3)
    coeffs_y = np.zeros(3)
    print("Plane subtraction disabled")

if PLANE_MODE == "measure":
    cx, cy = Rx // 2, Ry // 2
    ref_x = np.nanmean(disp_x[cx-VAC_REF_SIZE:cx+VAC_REF_SIZE+1,
                               cy-VAC_REF_SIZE:cy+VAC_REF_SIZE+1])
    ref_y = np.nanmean(disp_y[cx-VAC_REF_SIZE:cx+VAC_REF_SIZE+1,
                               cy-VAC_REF_SIZE:cy+VAC_REF_SIZE+1])
    rs, re, cs, ce = cx-VAC_REF_SIZE, cx+VAC_REF_SIZE+1, cy-VAC_REF_SIZE, cy+VAC_REF_SIZE+1
    print(f"Measure mode: central reference [{rs}:{re}, {cs}:{ce}]")
else:
    half   = SMART_REF_SIZE // 2
    margin = half + 1
    best_score = np.inf
    best_r, best_c = margin, margin
    for r in range(margin, Rx - margin):
        for c in range(margin, Ry - margin):
            px = disp_x[r-half:r+half+1, c-half:c+half+1]
            py = disp_y[r-half:r+half+1, c-half:c+half+1]
            if np.sum(np.isnan(px)) > 0:
                continue
            score = np.var(px) + np.var(py)
            if score < best_score:
                best_score = score
                best_r, best_c = r, c
    rs, re = best_r - half, best_r + half + 1
    cs, ce = best_c - half, best_c + half + 1
    ref_x = np.nanmean(disp_x[rs:re, cs:ce])
    ref_y = np.nanmean(disp_y[rs:re, cs:ce])
    print(f"Smart reference: [{rs}:{re}, {cs}:{ce}], score={best_score:.6f}, "
          f"ref=({ref_x:.4f}, {ref_y:.4f})")

    if np.isnan(ref_x) or np.isnan(ref_y):
        raise ValueError("Smart reference region contains only NaN fits.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title in zip(axes, [disp_x, disp_y], ['Displacement X', 'Displacement Y']):
        vmin, vmax = get_clim(data)
        im = ax.imshow(data, cmap='RdBu_r', vmin=vmin, vmax=vmax, extent=ext)
        plt.colorbar(im, ax=ax, label='pixels')
        ax.add_patch(mpatches.Rectangle(
            (cs*PIXEL_SIZE_ANG, rs*PIXEL_SIZE_ANG),
            SMART_REF_SIZE*PIXEL_SIZE_ANG, SMART_REF_SIZE*PIXEL_SIZE_ANG,
            linewidth=2, edgecolor='lime', facecolor='none'))
        ax.set_title(title)
        add_scalebar(ax)
    plt.suptitle('Smart reference region (green box)', fontsize=10)
    plt.tight_layout()
    plt.show()

disp_x = disp_x - ref_x
disp_y = disp_y - ref_y

dEx_dy, dEx_dx = np.gradient(disp_x)
dEy_dy, dEy_dx = np.gradient(disp_y)
divergence      = dEx_dx + dEy_dy
divergence_blur = gaussian_filter(divergence, sigma=SIGMA)


c_light = 2.998e8
m0      = 9.109e-31
e_C     = 1.602e-19
eps0    = 8.854e-12

gamma   = 1 + (BEAM_ENERGY_KV * 1e3 * e_C) / (m0 * c_light**2)
beta    = np.sqrt(1 - 1/gamma**2)
v_rel   = beta * c_light
m_rel   = gamma * m0

print(f"\n=== Relativistic parameters at {BEAM_ENERGY_KV} kV ===")
print(f"  γ = {gamma:.4f},  β = {beta:.4f},  v = {v_rel:.4e} m/s")
print(f"  m_rel × v² / e = {m_rel * v_rel**2 / e_C:.2f} V")

k_rad_per_px  = (DETECTOR_PIXEL_UM * 1e-6) / (CAMERA_LENGTH_MM * 1e-3)
PIXEL_SIZE_NM = PIXEL_SIZE_ANG * 0.1

print(f"\n  k (angular calibration): {k_rad_per_px*1e3:.4f} mrad/detector_px")
print(f"  Scan pixel size:         {PIXEL_SIZE_NM:.4f} nm")

rad_nm_to_rad_m = 1e9
C_m2_to_e_nm2  = 1e-18 / e_C

prefactor = eps0 * (m_rel * v_rel**2 / e_C) * rad_nm_to_rad_m * C_m2_to_e_nm2
print(f"  Physical prefactor:      {prefactor:.4f} (e/nm²) per (rad/nm)")
print(f"                           {prefactor/1e3:.6f} (e/nm²) per (mrad/nm)")

if USE_ARCSIN_CORRECTION:
    d_magnitude  = np.sqrt(disp_x**2 + disp_y**2)
    arcsin_arg   = d_magnitude * k_rad_per_px
    correction   = 1 / np.sqrt(np.clip(1 - arcsin_arg**2, 1e-10, None))

    print(f"\n  USE_ARCSIN_CORRECTION = True")
    print(f"  Max arcsin argument:   {arcsin_arg.max():.8f}  (sinα_max)")
    print(f"  Max correction factor: {correction.max():.8f}  (1.0 = no correction)")
    print(f"  Mean correction:       {correction.mean():.8f}")

    dax_dy, dax_dx = np.gradient(disp_x * k_rad_per_px, PIXEL_SIZE_NM)
    day_dy, day_dx = np.gradient(disp_y * k_rad_per_px, PIXEL_SIZE_NM)
    div_rad_per_nm = (dax_dx + day_dy) * correction

else:
    print(f"\n  USE_ARCSIN_CORRECTION = False (small angle approx, error < 0.001%)")

    dax_dy, dax_dx = np.gradient(disp_x * k_rad_per_px, PIXEL_SIZE_NM)
    day_dy, day_dx = np.gradient(disp_y * k_rad_per_px, PIXEL_SIZE_NM)
    div_rad_per_nm = dax_dx + day_dy

div_phys      = div_rad_per_nm * prefactor
div_phys_blur = gaussian_filter(div_phys, sigma=SIGMA)

div_phys_denoised = div_phys_blur

deflection_mrad = np.sqrt(disp_x**2 + disp_y**2) * k_rad_per_px * 1e3

print(f"\n=== Physical map statistics ===")
print(f"  Deflection magnitude:  {deflection_mrad.min():.4f} – {deflection_mrad.max():.4f} mrad")
print(f"  Charge density (raw):  {div_phys.min():.4f} – {div_phys.max():.4f} e/nm²  (std={div_phys.std():.4f})")
print(f"  Charge density (blur): {div_phys_denoised.min():.4f} – {div_phys_denoised.max():.4f} e/nm²  (std={div_phys_denoised.std():.4f})")


if PLANE_MODE == "measure":
    noise_rs, noise_re = cx-VAC_REF_SIZE, cx+VAC_REF_SIZE+1
    noise_cs, noise_ce = cy-VAC_REF_SIZE, cy+VAC_REF_SIZE+1
    noise_region_label = f"central reference [{noise_rs}:{noise_re}, {noise_cs}:{noise_ce}]"
else:
    noise_rs, noise_re, noise_cs, noise_ce = rs, re, cs, ce
    noise_region_label = f"smart reference [{rs}:{re}, {cs}:{ce}]"

noise_patch_x = centres_x[noise_rs:noise_re, noise_cs:noise_ce]
noise_patch_y = centres_y[noise_rs:noise_re, noise_cs:noise_ce]

if np.sum(~np.isnan(noise_patch_x)) >= 4:
    dx_diff = np.diff(noise_patch_x, axis=1)
    dy_diff = np.diff(noise_patch_y, axis=0)
    noise_x_px = np.nanstd(dx_diff) / np.sqrt(2)
    noise_y_px = np.nanstd(dy_diff) / np.sqrt(2)
else:
    noise_x_px = np.nanstd(noise_patch_x)
    noise_y_px = np.nanstd(noise_patch_y)
    print("WARNING: noise patch too small for finite difference — using std fallback")

noise_px              = max(noise_x_px, noise_y_px)
noise_mrad            = noise_px * k_rad_per_px * 1e3
div_noise_mrad_per_nm = np.sqrt(2) * noise_mrad / PIXEL_SIZE_NM

blur_noise_reduction  = 1.0 / (math.sqrt(2 * math.pi) * SIGMA)
div_noise_floor_raw   = div_noise_mrad_per_nm * 1e-3 * prefactor
div_noise_floor_blur  = div_noise_floor_raw * blur_noise_reduction

print(f"\n=== Fitting noise floor ===")
print(f"  Noise estimated from: {noise_region_label}")
print(f"  Shift noise (finite difference method):")
print(f"    X: {noise_x_px:.4f} px  →  {noise_x_px * k_rad_per_px * 1e3:.4f} mrad")
print(f"    Y: {noise_y_px:.4f} px  →  {noise_y_px * k_rad_per_px * 1e3:.4f} mrad")
print(f"  Divergence noise floor (raw):      {div_noise_mrad_per_nm:.4f} mrad/nm")
print(f"  Charge density noise floor (raw):  {div_noise_floor_raw:.4f} e/nm²")
print(f"  Charge density noise floor (blur): {div_noise_floor_blur:.4f} e/nm²  "
      f"(after σ={SIGMA} blur)")
print(f"  Charge density signal std:         {div_phys_denoised.std():.4f} e/nm²")

_snr      = div_phys_denoised.std() / div_noise_floor_blur
_snr_raw  = div_phys.std() / div_noise_floor_raw
print(f"  SNR (blurred signal / blurred noise): {_snr:.1f}×")
print(f"  SNR (raw signal / raw noise):         {_snr_raw:.1f}×")

if _snr < 2:
    print(f"  WARNING: signal std < 2× noise floor — maps may be noise-dominated")
    print(f"           Consider increasing SIGMA or checking fitting quality")
elif _snr < 5:
    print(f"  CAUTION: moderate SNR — blurred map more reliable than raw")
else:
    print(f"  Good SNR — signal is well above noise floor")

np.savez(DATASET_DIR / f"{FILE_STEM}_noise_floor.npz",
         charge_map      = div_phys,
         charge_map_blur = div_phys_denoised,
         noise_floor_raw  = div_noise_floor_raw,
         noise_floor_blur = div_noise_floor_blur,
         signal_std_raw   = div_phys.std(),
         signal_std_blur  = div_phys_denoised.std(),
         snr_raw  = _snr_raw,
         snr_blur = _snr,
         noise_region = np.array([noise_rs, noise_re, noise_cs, noise_ce]),
         pixel_nm = PIXEL_SIZE_NM)
print(f"Saved noise-floor data for supplementary figure: "
      f"{FILE_STEM}_noise_floor.npz")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, data, title in zip(axes,
                            [div_phys, div_phys_denoised],
                            ['Projected charge density (raw)',
                             f'Projected charge density (blur σ={SIGMA})']):
    vmin, vmax = get_clim(data)
    im = ax.imshow(data, cmap='RdBu_r', vmin=vmin, vmax=vmax, extent=ext)
    plt.colorbar(im, ax=ax, label='e/nm²')
    ax.set_title(title)
    add_scalebar(ax)
plt.suptitle(f'{"Exact arcsin" if USE_ARCSIN_CORRECTION else "Small angle approx"} | '
             f'{BEAM_ENERGY_KV} kV | CL={CAMERA_LENGTH_MM} mm', fontsize=10)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(deflection_mrad, cmap='viridis', extent=ext)
plt.colorbar(im, ax=ax, label='mrad')
ax.set_title('Beam deflection magnitude')
add_scalebar(ax)
plt.tight_layout()
plt.show()

suffix = "_arcsin" if USE_ARCSIN_CORRECTION else "_smallangle"
save_tif_colored(div_phys_denoised,   f"charge_density_blur_s{SIGMA}{suffix}",  unit='e/nm²')
np.save(DATASET_DIR / f"{FILE_STEM}_charge_density_raw.npy", div_phys)
print(f"Saved raw charge density array")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, data, title in zip(axes, [disp_x, disp_y],
                            [f'Displacement X (rot={ROTATION_ANGLE_DEG}°)',
                             f'Displacement Y (rot={ROTATION_ANGLE_DEG}°)']):
    vmin, vmax = get_clim(data)
    im = ax.imshow(data, cmap='RdBu_r', vmin=vmin, vmax=vmax, extent=ext)
    plt.colorbar(im, ax=ax, label='pixels')
    ax.set_title(title)
    ax.set_xlabel('Å')
    ax.set_ylabel('Å')
    add_scalebar(ax)
plt.tight_layout()
plt.show()

iy, ix   = np.mgrid[0:disp_x.shape[0]:N_QUIVER, 0:disp_x.shape[1]:N_QUIVER]
ix_ang   = ix * PIXEL_SIZE_ANG
iy_ang   = iy * PIXEL_SIZE_ANG
Ex_sub   = disp_x[::N_QUIVER, ::N_QUIVER]
Ey_sub   = disp_y[::N_QUIVER, ::N_QUIVER]

fig, ax = plt.subplots(figsize=(7, 7))
ax.quiver(ix_ang, iy_ang, Ex_sub, Ey_sub,
          color='black', alpha=0.8,
          scale=None, scale_units='xy',
          angles='xy', width=0.003)
ax.set_title(f'Vector field (rot={ROTATION_ANGLE_DEG}°)')
ax.set_aspect('equal')
ax.invert_yaxis()
ax.set_xlabel('Å')
ax.set_ylabel('Å')
add_scalebar(ax)
plt.tight_layout()
plt.show()

if PLANE_SUBTRACT:
    angle_x = np.degrees(np.arctan2(coeffs_x[1], coeffs_x[0]))
    angle_y = np.degrees(np.arctan2(coeffs_y[1], coeffs_y[0]))
    print(f"Descan angle in X map: {angle_x:.2f}°")
    print(f"Descan angle in Y map: {angle_y:.2f}°")


if OVERLAY_HAADF:
    if HAADF_PATH.exists():
        haadf_raw  = tifffile.imread(str(HAADF_PATH)).astype(float)
        haadf_norm = (haadf_raw - haadf_raw.min()) / (haadf_raw.max() - haadf_raw.min())

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, div_data, title in zip(axes,
                                        [div_phys, div_phys_denoised],
                                        ['HAADF + Divergence (raw)',
                                         f'HAADF + Divergence (blur σ={SIGMA})']):
            vmin, vmax = get_clim(div_data)
            ax.imshow(haadf_norm, cmap='gray', extent=ext)
            im = ax.imshow(div_data, cmap='RdBu_r', alpha=ALPHA_OVERLAY,
                           vmin=vmin, vmax=vmax, extent=ext)
            plt.colorbar(im, ax=ax, label='e/nm²', fraction=0.046, pad=0.04)
            ax.set_title(title)
            ax.set_xlabel('Å')
            ax.set_ylabel('Å')
            add_scalebar(ax)
        plt.tight_layout()
        plt.show()

        for div_data, name in [(div_phys,      "haadf_divergence_raw"),
                                (div_phys_denoised, f"haadf_divergence_blur_s{SIGMA}")]:
            vmin, vmax = get_clim(div_data)
            fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
            ax.imshow(haadf_norm, cmap='gray', extent=ext)
            im = ax.imshow(div_data, cmap='RdBu_r', alpha=ALPHA_OVERLAY,
                          vmin=vmin, vmax=vmax, extent=ext)
            plt.colorbar(im, ax=ax, label='e/nm²', fraction=0.046, pad=0.04)
            ax.set_title(name.replace('_', ' '))
            ax.set_xlabel('Å')
            ax.set_ylabel('Å')
            add_scalebar(ax)
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            Image.open(buf).convert('RGB').save(DATASET_DIR / f"{FILE_STEM}_{name}.tif")
            print(f"Saved: {FILE_STEM}_{name}.tif")
    else:
        print(f"HAADF not found at {HAADF_PATH} — skipping overlay plots")

DIAG_I, DIAG_J = 64, 64

dp_diag      = datacube.data[DIAG_I, DIAG_J].astype(float)
dp_edge_diag = get_disk_edge(dp_diag)

shift_diag, _, _ = phase_cross_correlation(
    TEMPLATE_EDGE, dp_edge_diag, upsample_factor=20
)
dy_diag, dx_diag = shift_diag

tcy, tcx = np.array(TEMPLATE.shape) / 2.0
disc_cx = tcx + (-dx_diag)
disc_cy = tcy + (-dy_diag)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(dp_diag, cmap='gray')
axes[0].set_title(f'Raw pattern [{DIAG_I}, {DIAG_J}]')
axes[0].axis('off')

axes[1].imshow(dp_edge_diag, cmap='gray')
axes[1].set_title('Detected edge\n(shared input to both methods)')
axes[1].axis('off')

axes[2].imshow(dp_diag, cmap='gray')
ref_overlay = np.zeros((*TEMPLATE_EDGE.shape, 4))
ref_overlay[..., 0] = 1.0
ref_overlay[..., 3] = TEMPLATE_EDGE / (TEMPLATE_EDGE.max() + 1e-9) * 0.6
axes[2].imshow(ref_overlay)
axes[2].plot(disc_cx, disc_cy, 'c+', markersize=20, markeredgewidth=2,
             label='measured disc centre')
axes[2].plot(tcx, tcy, 'y+', markersize=14, markeredgewidth=2,
             label='template centre')
axes[2].set_title(f'Cross-correlation result\n'
                  f'shift dx={-dx_diag:.2f}, dy={-dy_diag:.2f} px')
axes[2].legend(loc='lower right', fontsize=7)
axes[2].axis('off')

plt.suptitle(f'{FILE_STEM} — cross-correlation diagnostic at [{DIAG_I}, {DIAG_J}]',
             fontsize=11)
plt.tight_layout()
plt.savefig(DATASET_DIR / f"{FILE_STEM}_xcorr_diag_{DIAG_I}_{DIAG_J}.png",
            dpi=150, bbox_inches='tight')
plt.show()
print(f"Cross-correlation diagnostic at [{DIAG_I},{DIAG_J}]: "
      f"dx={-dx_diag:.4f}, dy={-dy_diag:.4f} px (disc displacement)")

np.save(DATASET_DIR / f"{FILE_STEM}_diag_{DIAG_I}_{DIAG_J}_raw.npy", dp_diag)
np.save(DATASET_DIR / f"{FILE_STEM}_diag_{DIAG_I}_{DIAG_J}_edge.npy", dp_edge_diag)
np.save(DATASET_DIR / f"{FILE_STEM}_diag_{DIAG_I}_{DIAG_J}_xcorr.npy",
        np.array([disc_cx, disc_cy, tcx, tcy, -dx_diag, -dy_diag]))
print("Saved diagnostic arrays for the figure script.")

r_mean_px = np.nan
CL_mm     = CAMERA_LENGTH_MM

print(f"\n{'='*50}")
print(f"  COMPLETED: {FILE_STEM}")
print(f"  Scan size:      {Rx} × {Ry} pixels")
print(f"  Pixel size:     {PIXEL_SIZE_ANG} Å")
print(f"NaN shifts: {np.sum(np.isnan(centres_x))} / {Rx * Ry}")
print(f"  Plane mode:     {PLANE_MODE if PLANE_SUBTRACT else 'disabled'}")
print(f"  Charge density std (blur): {div_phys_denoised.std():.4f} e/nm²")
print(f"  Calibrated CL:  "
      f"{'N/A (cross-corr mode)' if np.isnan(r_mean_px) else f'{CL_mm:.1f} mm'} "
      f"(set: {CAMERA_LENGTH_MM} mm)")
print(f"{'='*50}")

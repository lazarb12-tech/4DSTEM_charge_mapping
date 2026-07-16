import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter

FILE_STEM    = "FILENAME"  # adjust to match your dataset
PROJECT_ROOT = Path(os.environ.get("FOURDSTEM_ROOT", Path(__file__).resolve().parent))  # set the FOURDSTEM_ROOT env var, or edit this path, to point at your project folder
PIPELINE     = "crosscorr"   # "crosscorr" (disp = centres) or "ellipse" (disp = centres - mean)

BLUR_PX      = 4.0           # Gaussian blur before curl/div (literature: ~4 px helps a lot)
N_ANGLES     = 360           # angular sampling over 0–360 deg
# Optional: a region known to be high-MIP (the particle) to fix the 180 ambiguity.
# (row_start, row_end, col_start, col_end) in scan pixels, or None to skip.
PARTICLE_REGION = None       # e.g. (40, 90, 40, 90)
PARTICLE_SIGN_EXPECTED = -1  # +1 if high-MIP particle should read positive charge density

DATASET_DIR = PROJECT_ROOT / "results" / FILE_STEM
cx = np.load(DATASET_DIR / f"{FILE_STEM}_centres_x.npy")
cy = np.load(DATASET_DIR / f"{FILE_STEM}_centres_y.npy")

if PIPELINE == "ellipse":
    disp_x0 = cx - np.nanmean(cx)
    disp_y0 = cy - np.nanmean(cy)
else:
    disp_x0 = np.where(np.isfinite(cx), cx, np.nan)
    disp_y0 = np.where(np.isfinite(cy), cy, np.nan)

disp_x0 = np.nan_to_num(disp_x0 - np.nanmean(disp_x0))
disp_y0 = np.nan_to_num(disp_y0 - np.nanmean(disp_y0))

disp_x0 = gaussian_filter(disp_x0, BLUR_PX)
disp_y0 = gaussian_filter(disp_y0, BLUR_PX)

def curl_div_power(dx, dy):
    ddx_dy, ddx_dx = np.gradient(dx)
    ddy_dy, ddy_dx = np.gradient(dy)
    curl = ddy_dx - ddx_dy
    div  = ddx_dx + ddy_dy
    return np.sum(curl**2), np.sum(div**2)

angles = np.linspace(0, 2*np.pi, N_ANGLES, endpoint=False)
curl_power = np.zeros(N_ANGLES)
div_power  = np.zeros(N_ANGLES)

for i, phi in enumerate(angles):
    c, s = np.cos(phi), np.sin(phi)
    rx =  c*disp_x0 - s*disp_y0
    ry =  s*disp_x0 + c*disp_y0
    curl_power[i], div_power[i] = curl_div_power(rx, ry)

i_min = np.argmin(curl_power)
phi_best = angles[i_min]
phi_best_deg = np.degrees(phi_best)

def div_map(phi):
    c, s = np.cos(phi), np.sin(phi)
    rx =  c*disp_x0 - s*disp_y0
    ry =  s*disp_x0 + c*disp_y0
    ddx_dy, ddx_dx = np.gradient(rx)
    ddy_dy, ddy_dx = np.gradient(ry)
    return ddx_dx + ddy_dy

phi_candidates = [phi_best, (phi_best + np.pi) % (2*np.pi)]
chosen = phi_best
if PARTICLE_REGION is not None:
    rs, re, cs, ce = PARTICLE_REGION
    best_match = None
    for phi in phi_candidates:
        d = div_map(phi)
        s = np.sign(np.mean(d[rs:re, cs:ce]))
        if s == np.sign(PARTICLE_SIGN_EXPECTED):
            chosen = phi
            best_match = phi
    if best_match is None:
        print("WARNING: neither candidate matched expected particle sign; "
              "using curl-min angle.")
else:
    print("No PARTICLE_REGION given — reporting curl-min angle with its "
          "180-deg ambiguity; choose the one giving the physically correct sign.")

print(f"\n=== Rotation calibration ({FILE_STEM}) ===")
print(f"  Curl-minimising angle: {phi_best_deg:.1f} deg "
      f"(also valid: {(phi_best_deg+180)%360:.1f} deg)")
print(f"  Chosen angle:          {np.degrees(chosen):.1f} deg")
print(f"  --> set ROTATION_ANGLE_DEG = {np.degrees(chosen):.1f} in your pipeline")
print(f"  Curl power at min:  {curl_power[i_min]:.4e}")
print(f"  Curl power at 0deg: {curl_power[0]:.4e} "
      f"(ratio {curl_power[0]/curl_power[i_min]:.1f}x)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(np.degrees(angles), curl_power, label='curl power (minimise)')
axes[0].plot(np.degrees(angles), div_power, label='div power', alpha=0.6)
axes[0].axvline(phi_best_deg, color='red', ls='--',
                label=f'min curl = {phi_best_deg:.1f} deg')
axes[0].axvline(np.degrees(chosen), color='green', ls=':',
                label=f'chosen = {np.degrees(chosen):.1f} deg')
axes[0].set_xlabel('Scan–detector rotation (deg)')
axes[0].set_ylabel('Integrated power')
axes[0].set_title('Curl vs divergence power vs rotation')
axes[0].legend(); axes[0].grid(alpha=0.3)

d_chosen = div_map(chosen)
vlim = np.percentile(np.abs(d_chosen), 99)
im = axes[1].imshow(d_chosen, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
axes[1].set_title(f'Divergence (charge sign) at {np.degrees(chosen):.1f} deg')
plt.colorbar(im, ax=axes[1])
if PARTICLE_REGION is not None:
    import matplotlib.patches as mp
    rs, re, cs, ce = PARTICLE_REGION
    axes[1].add_patch(mp.Rectangle((cs, rs), ce-cs, re-rs,
                                   fill=False, edgecolor='lime', linewidth=2))
plt.tight_layout()
plt.savefig(DATASET_DIR / f"{FILE_STEM}_rotation_calibration.png", dpi=150)
plt.show()
print(f"\nSaved {FILE_STEM}_rotation_calibration.png")

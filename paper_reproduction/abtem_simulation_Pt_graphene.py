"""Independent-atom-model reference simulation of a Pt nanoparticle on graphene (SI Figure S2)."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import abtem
from ase import Atoms
from ase.build import bulk
from scipy.fft import next_fast_len
from scipy.ndimage import fourier_shift, gaussian_filter, generic_filter
from skimage import filters, feature, morphology
from skimage.registration import phase_cross_correlation

QUICK_LOOK = False               # True: coarse + fast (minutes) — use this to check
                                # the ring is closed. False: production quality.

BEAM_ENERGY_KV      = 80

CONV_SEMIANGLE_MRAD = 18.7      # convergence semi-angle (experimental value)
DETECTOR_MAX_MRAD   = 40.0      # detector cut-off

if QUICK_LOOK:
    POTENTIAL_SAMPLING = 0.15   # A
    SCAN_SAMPLING      = 2.0    # A  (coarse; for checking only)
else:
    POTENTIAL_SAMPLING = 0.10   # A (upper limit; the grid is rounded up to a fast FFT size)
    SCAN_SAMPLING      = 1.2

UPSAMPLE         = 20           # as in CentreFitting_xcorr.py
# sub-pixel offsets averaged per pattern; [(0, 0)] = off
DITHER_OFFSETS   = [(0, 0), (0.5, 0.5), (0.25, 0.75), (0.75, 0.25)]
MEASURE_BLUR_PX  = 1.5          # smoothing of the deflection FIELD before divergence
DISPLAY_BLUR_PX  = 2.0          # final display smoothing of the charge map

CACHE_DIR = f"IAM_4D_rows_{CONV_SEMIANGLE_MRAD:g}mrad" + ("_quick" if QUICK_LOOK else "")

# "graphene"    : ORTHORHOMBIC few-layer graphene.
# "amorphous_c" : similar mean inner potential, no diffraction discs from the support.
SUPPORT_TYPE    = "graphene"
GRAPHENE_LAYERS = 3
INTERFACE_GAP_A = 2.3

AMORPHOUS_THICKNESS_A = GRAPHENE_LAYERS * 3.35
AMORPHOUS_DENSITY     = 0.10    # C atoms / A^3 (~2 g/cm^3)
AMORPHOUS_SEED        = 0

PT_DIAMETER_NM    = 4.0
OFF_ZONE_TILT_DEG = (7.0, 5.0)  # well off the zone axis; unequal so no accidental
                                # symmetry is imposed on the channelling.

SCAN_MARGIN_A    = 26.0         # support visible AROUND the particle. Generous, so
                                # the round particle is NOT clipped by the square scan.
EDGE_CLEARANCE_A = 22.0         # empty support between the FOV and the cell edge.

c_light = 2.998e8; m0 = 9.109e-31; e_C = 1.602e-19; eps0 = 8.854e-12
gamma  = 1 + (BEAM_ENERGY_KV*1e3*e_C)/(m0*c_light**2)
beta   = np.sqrt(1 - 1/gamma**2)
v_rel  = beta*c_light
m_rel  = gamma*m0
prefactor = eps0*(m_rel*v_rel**2/e_C)*1e9*(1e-18/e_C)

lam_A = 12.398/np.sqrt(BEAM_ENERGY_KV*(2*511 + BEAM_ENERGY_KV))
g_graphene_mrad = 1000*lam_A/(2.46*np.sqrt(3)/2)

print(f"Pt/{SUPPORT_TYPE} IAM | lambda={lam_A:.4f} A | prefactor={prefactor:.1f} (e/nm2)/(rad/nm)")
print(f"graphene {{10-10}} at {g_graphene_mrad:.1f} mrad; BF disc edge at {CONV_SEMIANGLE_MRAD:.1f} mrad: "
      f"{'the graphene discs overlap the BF disc (as in the experiment)' if g_graphene_mrad < 2*CONV_SEMIANGLE_MRAD else 'no overlap'}")

a_pt = 3.924
R = PT_DIAMETER_NM*10/2.0
rep = int(np.ceil(PT_DIAMETER_NM*10/a_pt)) + 4
block = bulk('Pt', 'fcc', a=a_pt, cubic=True).repeat((rep, rep, rep))
d = np.linalg.norm(block.positions - block.positions.mean(0), axis=1)
pt = block[d <= R]

pt.rotate(OFF_ZONE_TILT_DEG[0], 'x', center='COM')
pt.rotate(OFF_ZONE_TILT_DEG[1], 'y', center='COM')

pt_size = pt.positions.max(0) - pt.positions.min(0)
print(f"Pt sphere: {len(pt)} atoms, {pt_size[0]/10:.2f} nm across, "
      f"projected thickness {pt_size[2]/10:.2f} nm, tilt {OFF_ZONE_TILT_DEG} deg")

half_fov  = pt_size[0]/2 + SCAN_MARGIN_A
half_cell = half_fov + EDGE_CLEARANCE_A
need = 2*half_cell

if SUPPORT_TYPE == "graphene":
    a0 = 2.46
    Lx, Ly = a0, a0*np.sqrt(3)
    basis = np.array([[0.0, 0.0],
                      [Lx/2, Ly/6],
                      [Lx/2, Ly/2],
                      [0.0, 2*Ly/3]])
    nx = int(np.ceil(need/Lx))
    ny = int(np.ceil(need/Ly))
    cell_x, cell_y = nx*Lx, ny*Ly

    layer0 = Atoms('C4', positions=np.c_[basis, np.zeros(4)],
                   cell=[Lx, Ly, 10.0], pbc=(True, True, False)).repeat((nx, ny, 1))
    zsp = 3.35
    support = layer0.copy()
    for L in range(1, GRAPHENE_LAYERS):
        lay = layer0.copy()
        lay.positions[:, 2] += L*zsp
        if L % 2 == 1:
            lay.positions[:, 1] += 1.42
        support += lay
    support_desc = (f"orthorhombic graphene {nx}x{ny} cells, {GRAPHENE_LAYERS} layers, "
                    f"{len(support)} C")

elif SUPPORT_TYPE == "amorphous_c":
    cell_x = cell_y = float(np.ceil(need))
    thick = AMORPHOUS_THICKNESS_A
    n_C = int(round(AMORPHOUS_DENSITY*cell_x*cell_y*thick))
    rng = np.random.default_rng(AMORPHOUS_SEED)
    pos = rng.uniform([0, 0, 0], [cell_x, cell_y, thick], size=(n_C, 3))
    support = Atoms(numbers=[6]*n_C, positions=pos,
                    cell=[cell_x, cell_y, thick], pbc=(True, True, False))
    support_desc = f"amorphous C, {n_C} atoms, {thick:.1f} A thick"
else:
    raise ValueError("SUPPORT_TYPE must be 'graphene' or 'amorphous_c'")

pt.positions[:, 0] += cell_x/2 - pt.positions[:, 0].mean()
pt.positions[:, 1] += cell_y/2 - pt.positions[:, 1].mean()
pt.positions[:, 2] += support.positions[:, 2].max() + INTERFACE_GAP_A - pt.positions[:, 2].min()

atoms = support + pt
zmin, zmax = atoms.positions[:, 2].min(), atoms.positions[:, 2].max()
atoms.set_cell([cell_x, cell_y, (zmax - zmin) + 16.0])
atoms.positions[:, 2] += 8.0 - zmin
atoms.pbc = (True, True, False)

gcx = atoms.positions[atoms.numbers == 78][:, 0].mean()
gcy = atoms.positions[atoms.numbers == 78][:, 1].mean()
print(f"System: {len(atoms)} atoms, cell {np.round(atoms.cell.lengths(),1)} A ({support_desc})")

gpts = tuple(next_fast_len(int(np.ceil(c/POTENTIAL_SAMPLING))) for c in (cell_x, cell_y))
potential = abtem.Potential(atoms, gpts=gpts, projection='infinite', slice_thickness=2.0)
print(f"potential grid {gpts}, sampling {cell_x/gpts[0]:.4f} x {cell_y/gpts[1]:.4f} A, "
      f"{len(potential)} slices")
probe = abtem.Probe(energy=BEAM_ENERGY_KV*1e3, semiangle_cutoff=CONV_SEMIANGLE_MRAD)
probe.grid.match(potential)

clearance = min(gcx - half_fov, gcy - half_fov,
                cell_x - (gcx + half_fov), cell_y - (gcy + half_fov))
print(f"FOV {2*half_fov/10:.1f} x {2*half_fov/10:.1f} nm "
      f"(particle is {100*pt_size[0]/(2*half_fov):.0f}% of the FOV width); "
      f"clearance to cell edge {clearance:.1f} A")
if clearance < 18.0:
    print("  *** WARNING: low clearance — raise EDGE_CLEARANCE_A. ***")

scan = abtem.GridScan(start=[gcx - half_fov, gcy - half_fov],
                      end=[gcx + half_fov, gcy + half_fov], sampling=SCAN_SAMPLING)
positions = np.asarray(scan.get_positions())
step_A = float(scan.sampling[0])
detector = abtem.PixelatedDetector(max_angle=DETECTOR_MAX_MRAD)
print(f"Scan: {positions.shape[0]} x {positions.shape[1]} probe positions at {step_A:.3f} A")

os.makedirs(CACHE_DIR, exist_ok=True)
todo = [i for i in range(positions.shape[0])
        if not os.path.exists(os.path.join(CACHE_DIR, f"row_{i:03d}.npy"))]
if todo:
    print(f"simulating {len(todo)} of {positions.shape[0]} rows")
    potential = potential.build().compute()
    for i in todo:
        meas = probe.scan(potential, scan=abtem.CustomScan(positions[i]), detectors=detector)
        meas.compute()
        np.save(os.path.join(CACHE_DIR, f"row_{i:03d}.npy"), np.asarray(meas.array, dtype=np.float32))
        print(f"  row {i + 1}/{positions.shape[0]} done", flush=True)
dp4d = np.stack([np.load(os.path.join(CACHE_DIR, f"row_{i:03d}.npy"))
                 for i in range(positions.shape[0])])
s0, s1, q0, q1 = dp4d.shape
print(f"4D shape: {dp4d.shape}")

det_mrad_per_px = np.array([1000*lam_A/cell_x, 1000*lam_A/cell_y])
bf_radius_px = CONV_SEMIANGLE_MRAD/det_mrad_per_px.mean()

vacuum = detector.detect(probe.build([(gcx, gcy)]))
vacuum = np.asarray(vacuum.compute().array if hasattr(vacuum, 'compute') else vacuum.array)
vacuum = vacuum.reshape(vacuum.shape[-2:])
assert vacuum.shape == (q0, q1), (vacuum.shape, (q0, q1))

mean_pat = dp4d.mean(axis=(0, 1))
yy, xx = np.mgrid[0:q0, 0:q1]
core = vacuum > 0.5*vacuum.max()
qc0 = (yy*core).sum()/core.sum()
qc1 = (xx*core).sum()/core.sum()
print(f"BF disc: centre ({qc1:.1f}, {qc0:.1f}), radius {bf_radius_px:.1f} px")
if bf_radius_px < 12:
    print("  *** WARNING: BF disc < 12 px. Lower DETECTOR_MAX_MRAD. ***")

plt.imsave("mean_pattern.png", np.log1p(mean_pat), cmap="inferno")


_DISK3 = morphology.disk(3)
_DISK5 = morphology.disk(5)


def get_disk_edge(dp):
    dp_med = filters.median(dp.astype(np.float32), _DISK3)
    dp_bin = dp_med > filters.threshold_otsu(dp_med)
    dp_closed = morphology.closing(dp_bin, _DISK5)
    return feature.canny(dp_closed.astype(float), sigma=1).astype(float)


template_edge = get_disk_edge(vacuum)
rr = np.hypot(xx - qc1, yy - qc0)
bf_window = rr < 1.35*bf_radius_px

shifts = np.full((s0, s1, 2), np.nan)
com    = np.full((s0, s1, 2), np.nan)
for i in range(s0):
    for j in range(s1):
        pat = dp4d[i, j]
        F = np.fft.fft2(pat)
        acc = []
        for o in DITHER_OFFSETS:
            p_o = pat if o == (0, 0) else np.real(np.fft.ifft2(fourier_shift(F, o)))
            sh, _, _ = phase_cross_correlation(template_edge, get_disk_edge(p_o),
                                               upsample_factor=UPSAMPLE)
            acc.append(sh + np.asarray(o))
        sh = np.mean(acc, axis=0)
        if np.hypot(*sh) < bf_radius_px:
            shifts[i, j] = sh
        w = pat*bf_window
        com[i, j] = [(yy*w).sum()/w.sum(), (xx*w).sum()/w.sum()]


def _nanmed(v):
    v = v[np.isfinite(v)]
    return np.median(v) if v.size else 0.0


n_failed = int((~np.isfinite(shifts[..., 0])).sum())
for k in range(2):
    c = shifts[..., k]
    bad = ~np.isfinite(c)
    if bad.any():
        c[bad] = generic_filter(np.nan_to_num(c), _nanmed, size=3)[bad]
        shifts[..., k] = c
print(f"failed registrations (median-filled): {n_failed}")


def charge_from(comp0, comp1):
    a0 = -comp0*det_mrad_per_px[0]*1e-3
    a1 = -comp1*det_mrad_per_px[1]*1e-3
    m = np.zeros(a0.shape, bool); m[:4] = m[-4:] = True; m[:, :4] = m[:, -4:] = True
    a0 -= a0[m].mean(); a1 -= a1[m].mean()
    a0 = gaussian_filter(a0, MEASURE_BLUR_PX)
    a1 = gaussian_filter(a1, MEASURE_BLUR_PX)
    d0 = np.gradient(a0, step_A*0.1, axis=0)
    d1 = np.gradient(a1, step_A*0.1, axis=1)
    return gaussian_filter((d0 + d1)*prefactor, DISPLAY_BLUR_PX), np.hypot(a0, a1)


charge, defl = charge_from(shifts[..., 0], shifts[..., 1])
charge_com, _ = charge_from(-(com[..., 0] - np.nanmean(com[..., 0])),
                            -(com[..., 1] - np.nanmean(com[..., 1])))

ic, jc = (gcx - positions[0, 0, 0])/step_A, (gcy - positions[0, 0, 1])/step_A
I, J = np.mgrid[0:s0, 0:s1]
Rr = np.hypot(I - ic, J - jc)*step_A*0.1
rbin = np.arange(0, Rr.max(), step_A*0.1)
idx = np.digitize(Rr.ravel(), rbin)
prof = np.array([charge.ravel()[idx == k].mean() if (idx == k).any() else np.nan
                 for k in range(1, len(rbin))])
rnm = 0.5*(rbin[:-1] + rbin[1:])
ok = np.isfinite(prof)
rnm, prof = rnm[ok], prof[ok]
k0 = int(np.argmax(prof[:-1]*prof[1:] < 0))
r_sign = rnm[k0] - prof[k0]*(rnm[k0 + 1] - rnm[k0])/(prof[k0 + 1] - prof[k0])
corr = float(np.corrcoef(charge.ravel(), charge_com.ravel())[0, 1])
summary = dict(
    interior_e_nm2=float(np.interp(0, rnm, prof)), centre_pixel_e_nm2=float(charge[int(round(ic)), int(round(jc))]),
    rim_max_e_nm2=float(prof.max()), r_rim_max_nm=float(rnm[np.argmax(prof)]),
    r_sign_change_nm=float(r_sign), particle_radius_nm=float(pt_size[0]/20),
    max_deflection_mrad=float(defl.max()*1e3), dither_offsets=DITHER_OFFSETS,
    corr_with_com=corr, failed_registrations=n_failed, scan_step_A=step_A,
    fov_nm=float(2*half_fov/10), potential_gpts=list(gpts), n_atoms=len(atoms), n_pt=len(pt))
json.dump(summary, open("Pt_graphene_IAM_summary.json", "w"), indent=1)
for k, v in summary.items():
    print(f"  {k}: {v}")
np.save("Pt_graphene_IAM_charge_reference.npy", charge)

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 7.5, 'axes.titlesize': 7.5})
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.505), gridspec_kw=dict(width_ratios=[1.12, 1, 1]))
vl = float(np.ceil(np.nanmax(np.abs(charge))))
ext = [-ic*step_A*0.1, (s0 - 1 - ic)*step_A*0.1, -jc*step_A*0.1, (s1 - 1 - jc)*step_A*0.1]
im = ax[0].imshow(charge.T, origin='lower', cmap='RdBu_r', vmin=-vl, vmax=vl, extent=ext)
rp = pt_size[0]/20
ax[0].add_patch(plt.Circle((0, 0), rp, fill=False, ls='--', lw=0.9, color='0.3'))
ax[0].plot([ext[0], ext[1]], [0, 0], 'k--', lw=1.0)
ax[0].plot(ext[0] + 0.12, 0, 'ko', ms=3)
ax[0].plot([ext[0] + 0.6, ext[0] + 2.6], [ext[2] + 0.7]*2, 'k-', lw=3)
ax[0].text(ext[0] + 1.6, ext[2] + 0.95, '2 nm', ha='center', va='bottom')
ax[0].set_xticks([]); ax[0].set_yticks([])
ax[0].set_title('Projected charge density (IAM)')
cb = plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04); cb.set_label('e nm$^{-2}$')
x_nm = (np.arange(s0) - ic)*step_A*0.1
line = charge[:, int(round(jc))]
ax[1].plot(x_nm, line, color='#1f5f7f', lw=1.5)
ax[2].plot(rnm, prof, color='#b22222', lw=1.5)
for a in ax[1:]:
    a.axhline(0, color='0.5', lw=0.8, ls=':')
    a.set_ylabel('Charge density (e nm$^{-2}$)')
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
for s in (-rp, rp):
    ax[1].axvline(s, color='0.7', lw=0.9, ls='--')
ax[2].axvline(rp, color='0.7', lw=0.9, ls='--')
ax[1].set_xlabel('Distance from center (nm)'); ax[1].set_title('Line profile')
ax[2].set_xlabel('Radius from center (nm)'); ax[2].set_title('Radial profile (azimuthal average)')
ax[2].set_xlim(0, None)
ax[1].set_xticks([-4, -2, 0, 2, 4])
fig.subplots_adjust(left=0.03, right=0.965, bottom=0.17, top=0.86, wspace=0.62)
for a, lab, dx in zip(ax, 'abc', (-0.08, -0.26, -0.26)):
    a.text(dx, 1.08, lab, transform=a.transAxes, fontsize=10, fontweight='bold', va='bottom')
plt.savefig("FigS2_IAM_reference.png", dpi=200)
print("\nSaved: FigS2_IAM_reference.png, Pt_graphene_IAM_charge_reference.npy, "
      "Pt_graphene_IAM_summary.json, mean_pattern.png")

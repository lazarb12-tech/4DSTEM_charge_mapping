import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import abtem
from ase import Atoms
from ase.build import bulk
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude, generic_filter
from skimage.registration import phase_cross_correlation

QUICK_LOOK = False               # True: coarse + fast (minutes) — use this to check
                                # the ring is closed. False: production quality.

BEAM_ENERGY_KV      = 80

CONV_SEMIANGLE_MRAD = 7.0       # needs 2*alpha < 19.6 mrad. 7.0 -> clean.
DETECTOR_MAX_MRAD   = 11.0      # < 12.6 mrad, so diffracted discs are not even
                                # recorded. BF disc (radius 7) sits well inside.

if QUICK_LOOK:
    POTENTIAL_SAMPLING = 0.15   # A
    SCAN_SAMPLING      = 2.0    # A  (probe is ~7 A wide at 7 mrad, so this is ample)
else:
    POTENTIAL_SAMPLING = 0.10
    SCAN_SAMPLING      = 1.2

EDGE_PREBLUR_PX  = 1.0          # pre-blur of each pattern before the soft edge
MEASURE_BLUR_PX  = 1.5          # smoothing of the deflection FIELD before divergence
DISPLAY_BLUR_PX  = 2.0          # final display smoothing of the charge map

# "graphene"    : correct ORTHORHOMBIC few-layer graphene (what you asked for).
# "amorphous_c" : same mean inner potential, no diffraction discs at all. Use this
#                 if you want to run at the experimental 18.7 mrad.
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
print(f"graphene {{10-10}} at {g_graphene_mrad:.1f} mrad; BF disc edge at "
      f"{CONV_SEMIANGLE_MRAD:.1f} mrad; discs overlap BF disc if "
      f"{g_graphene_mrad:.1f} < {2*CONV_SEMIANGLE_MRAD:.1f}  -> "
      f"{'OVERLAP (BAD)' if g_graphene_mrad < 2*CONV_SEMIANGLE_MRAD else 'CLEAN'}")
if SUPPORT_TYPE == "graphene" and g_graphene_mrad < 2*CONV_SEMIANGLE_MRAD:
    print("  *** WARNING: the graphene discs overlap the BF disc. The edge tracker "
          "will be corrupted along the lattice directions. Lower CONV_SEMIANGLE_MRAD "
          "below 9.8, or set SUPPORT_TYPE='amorphous_c'. ***")

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

potential = abtem.Potential(atoms, sampling=POTENTIAL_SAMPLING,
                            projection='infinite', slice_thickness=2.0)
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
detector = abtem.PixelatedDetector(max_angle=DETECTOR_MAX_MRAD)
print(f"Scan: ~{int(2*half_fov/SCAN_SAMPLING)}^2 probe positions at {SCAN_SAMPLING} A")

meas = probe.scan(potential, scan=scan, detectors=detector)
meas.compute()
dp4d = np.asarray(meas.array)
s0, s1, q0, q1 = dp4d.shape
print(f"4D shape: {dp4d.shape}")

det_mrad_per_px = (2*DETECTOR_MAX_MRAD)/q0
bf_radius_px = CONV_SEMIANGLE_MRAD/det_mrad_per_px

mean_pat = dp4d.mean(axis=(0, 1))
yy, xx = np.mgrid[0:q0, 0:q1]
core = mean_pat > 0.5*mean_pat.max()
qc0 = (yy*core).sum()/core.sum()
qc1 = (xx*core).sum()/core.sum()
print(f"BF disc: centre ({qc1:.1f}, {qc0:.1f}), radius {bf_radius_px:.1f} px")
if bf_radius_px < 12:
    print("  *** WARNING: BF disc < 12 px. Lower DETECTOR_MAX_MRAD. ***")

plt.imsave("mean_pattern.png", np.log1p(mean_pat), cmap="inferno")

rr = np.hypot(xx - qc1, yy - qc0)
bf_window = 0.5*(1.0 - np.tanh((rr - 1.35*bf_radius_px)/2.0))

def disc_edge(p):
    """Soft, isotropic ring: |grad| of the smoothed, softly windowed disc."""
    s = gaussian_filter(p.astype(float), EDGE_PREBLUR_PX)*bf_window
    return gaussian_gradient_magnitude(s, 1.5)

ref_edge = disc_edge(mean_pat)

shifts = np.full((s0, s1, 2), np.nan)
com    = np.full((s0, s1, 2), np.nan)
for i in range(s0):
    for j in range(s1):
        pat = dp4d[i, j]
        sh, _, _ = phase_cross_correlation(ref_edge, disc_edge(pat),
                                           upsample_factor=20, normalization=None)
        if np.hypot(*sh) < bf_radius_px:
            shifts[i, j] = sh
        w = pat*bf_window
        com[i, j] = [(yy*w).sum()/w.sum(), (xx*w).sum()/w.sum()]

def _nanmed(v):
    v = v[np.isfinite(v)]
    return np.median(v) if v.size else 0.0
for k in range(2):
    c = shifts[..., k]
    bad = ~np.isfinite(c)
    if bad.any():
        c[bad] = generic_filter(np.nan_to_num(c), _nanmed, size=3)[bad]
        shifts[..., k] = c

print(f"disc-shift std:  axis0 {np.nanstd(shifts[...,0]):.3f} px   "
      f"axis1 {np.nanstd(shifts[...,1]):.3f} px   <-- MUST be comparable. If one is "
      f"~0 the tracker is still losing that component.")

def charge_from(comp0, comp1):
    """comp0, comp1 = disc displacement (px) along array axes 0 and 1."""
    a0 = -comp0*det_mrad_per_px*1e-3
    a1 = -comp1*det_mrad_per_px*1e-3
    m = np.zeros(a0.shape, bool); m[:4] = m[-4:] = True; m[:, :4] = m[:, -4:] = True
    a0 -= a0[m].mean(); a1 -= a1[m].mean()
    a0 = gaussian_filter(a0, MEASURE_BLUR_PX)
    a1 = gaussian_filter(a1, MEASURE_BLUR_PX)
    d0 = np.gradient(a0, SCAN_SAMPLING*0.1, axis=0)
    d1 = np.gradient(a1, SCAN_SAMPLING*0.1, axis=1)
    return gaussian_filter((d0 + d1)*prefactor, DISPLAY_BLUR_PX), np.hypot(a0, a1)

charge, defl = charge_from(shifts[..., 0], shifts[..., 1])
charge_com, _ = charge_from(com[..., 0] - np.nanmean(com[..., 0]),
                            com[..., 1] - np.nanmean(com[..., 1]))

print(f"max deflection: {defl.max()*1e3:.3f} mrad")
print(f"MIP-only charge density: {charge.min():.3f} to {charge.max():.3f} e/nm2")

ny_, nx_ = charge.shape
Y, X = np.mgrid[0:ny_, 0:nx_]
neg = charge < np.percentile(charge, 10)
cy, cx = (Y*neg).sum()/neg.sum(), (X*neg).sum()/neg.sum()
Rr = np.hypot(Y - cy, X - cx)
ang = np.degrees(np.arctan2(Y - cy, X - cx))
rim = (Rr > 0.30*nx_) & (Rr < 0.46*nx_)
quads = {'right': (ang >= -45) & (ang < 45), 'down': (ang >= 45) & (ang < 135),
         'left': (np.abs(ang) >= 135),        'up':  (ang >= -135) & (ang < -45)}
vals = {k: charge[rim & m].mean() for k, m in quads.items()}
print("\nRIM CHECK (all four should be POSITIVE and SIMILAR for a round particle):")
for k, v in vals.items():
    print(f"   {k:>5}: {v:+.3f} e/nm2")
spread = max(vals.values()) - min(vals.values())
mean_rim = np.mean(list(vals.values()))
print(f"   spread/mean = {abs(spread/mean_rim) if mean_rim else np.inf:.2f}  "
      f"(< ~0.4 = isotropic ring, GOOD; > 1 = still an axis artefact)")

corr = np.corrcoef(charge.ravel(), charge_com.ravel())[0, 1]
print(f"edge-tracker vs centre-of-mass charge map correlation: {corr:.3f} "
      f"(> ~0.9 means the two independent trackers agree)")

np.save("Pt_graphene_IAM_charge_reference.npy", charge)

fig, axp = plt.subplots(1, 3, figsize=(16, 4.6))
ext = [-half_fov/10, half_fov/10, -half_fov/10, half_fov/10]

vl = np.nanpercentile(np.abs(charge), 99)
im = axp[0].imshow(charge, cmap='RdBu_r', vmin=-vl, vmax=vl, extent=ext)
axp[0].set_title('MIP-only charge (edge-tracked)\nNO charge transfer by construction')
axp[0].set_xlabel('nm'); plt.colorbar(im, ax=axp[0], label='e/nm$^2$', fraction=0.046)

im = axp[1].imshow(defl*1e3, cmap='inferno', extent=ext)
axp[1].set_title('|Deflection| (mrad)\nshould be a closed ring around the particle')
axp[1].set_xlabel('nm'); plt.colorbar(im, ax=axp[1], fraction=0.046)

rbin = np.linspace(0, 0.5*nx_, 40)
idx = np.digitize(Rr.ravel(), rbin)
prof = np.array([charge.ravel()[idx == k].mean() if (idx == k).any() else np.nan
                 for k in range(1, len(rbin))])
rnm = 0.5*(rbin[:-1] + rbin[1:])*SCAN_SAMPLING*0.1
axp[2].axhline(0, color='k', lw=0.8)
axp[2].plot(rnm, prof, 'o-', color='crimson')
axp[2].set_xlabel('radius from particle centre (nm)')
axp[2].set_ylabel('charge density (e/nm$^2$)')
axp[2].set_title('Radial profile (azimuthal average)\nnegative core, positive rim')

plt.suptitle(f'Pt/{SUPPORT_TYPE} IAM MIP reference — {pt_size[0]/10:.1f} nm sphere, '
             f'tilt {OFF_ZONE_TILT_DEG}, {CONV_SEMIANGLE_MRAD} mrad', fontsize=12)
plt.tight_layout()
plt.savefig("Pt_graphene_IAM_reference.png", dpi=150)
print("\nSaved: Pt_graphene_IAM_reference.png, Pt_graphene_IAM_charge_reference.npy, "
      "mean_pattern.png")

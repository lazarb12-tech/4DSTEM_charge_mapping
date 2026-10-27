"""Multislice thickness series (SI Figure S5, Table S4): slabs of each material and orientation (abTEM, independent-atom
potentials, 80 kV, 18.7 mrad), with and without a uniform in-plane field, at 16 probe positions spanning one projected
unit cell; patterns every 1 nm are analysed with the trackers.
Usage: python thickness_series.py [names]  ->  thickness_out/<name>.npz"""
import os
import sys, time, json
import numpy as np
import scipy.fft as sfft
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trackers import (get_disk_edge, build_template_edge, track_xcorr, fit_ellipse, com,
                      edge_contamination, qc_metrics)
import abtem
from abtem.core.energy import energy2sigma, energy2wavelength
from ase import Atoms
from ase.build import bulk
from ase.spacegroup import crystal

E = 80e3
LAM = energy2wavelength(E)              # A
SIG = energy2sigma(E)                   # rad / (V A)
ALPHA = 18.7e-3
L = 64.0
NG = 1080
DX = L / NG
DZ = 2.0
K_DET = 0.18644e-3                      # rad / detector px
NDET = 256
PSF = 0.7
NE = 1.5e5
OUT = 'thickness_out'


def rutile():
    return crystal(['Ti', 'O'], basis=[(0, 0, 0), (0.3048, 0.3048, 0)], spacegroup=136,
                   cellpar=[4.594, 4.594, 2.959, 90, 90, 90])


def tin():
    return bulk('TiN', 'rocksalt', a=4.24, cubic=True)


def pt():
    return bulk('Pt', 'fcc', a=3.924, cubic=True)


def ru():
    return bulk('Ru', 'hcp', a=2.706, c=4.282, orthorhombic=True)


def graphite():
    return crystal(['C', 'C'], basis=[(0, 0, 0.25), (1/3, 2/3, 0.25)], spacegroup=194,
                   cellpar=[2.46, 2.46, 6.70, 90, 90, 120])


def rotation_to_z(uvw, cell):
    v = np.asarray(uvw, float) @ np.asarray(cell)
    v /= np.linalg.norm(v)
    z = np.array([0, 0, 1.0])
    ax = np.cross(v, z); s = np.linalg.norm(ax); c = v @ z
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1])
    ax /= s
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + s*K + (1 - c)*K @ K


def tilt_matrix(theta_deg, phi_deg):
    th = np.radians(theta_deg); ph = np.radians(phi_deg)
    ax = np.array([np.cos(ph), np.sin(ph), 0.0])
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(th)*K + (1 - np.cos(th))*K @ K


def slab(unit, uvw, tilt, t_max, seed=0):
    Rm = tilt_matrix(*tilt) @ rotation_to_z(uvw, unit.cell)
    half = 0.5*np.sqrt(2*L**2 + t_max**2) + 6.0
    lens = unit.cell.lengths()
    reps = [int(np.ceil(2.4*half / l)) + 2 for l in lens]
    big = unit.repeat(reps)
    pos = big.get_positions()
    pos -= pos.mean(0)
    pos = pos @ Rm.T
    box_c = np.array([L/2, L/2, t_max/2])
    pos = pos + box_c
    m = (pos[:, 0] >= 0) & (pos[:, 0] < L) & (pos[:, 1] >= 0) & (pos[:, 1] < L) & \
        (pos[:, 2] >= 0) & (pos[:, 2] < t_max)
    at = Atoms(numbers=big.numbers[m], positions=pos[m], cell=[L, L, t_max], pbc=True)
    return at


def amorphous_c(t_max, seed=0, density=0.10, dmin=1.3):
    rng = np.random.default_rng(seed)
    n_target = int(density*L*L*t_max)
    nc = [int(L//dmin), int(L//dmin), max(1, int(t_max//dmin))]
    cs = [L/nc[0], L/nc[1], t_max/nc[2]]
    grid = {}
    pts = np.empty((n_target, 3)); n = 0
    while n < n_target:
        cand = rng.uniform([0, 0, 0], [L, L, t_max], size=(20000, 3))
        for c in cand:
            ci = (int(c[0]//cs[0]) % nc[0], int(c[1]//cs[1]) % nc[1], min(int(c[2]//cs[2]), nc[2] - 1))
            ok = True
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        z = ci[2] + dz
                        if z < 0 or z >= nc[2]:
                            continue
                        for j in grid.get(((ci[0]+dx) % nc[0], (ci[1]+dy) % nc[1], z), ()):
                            d = pts[j] - c
                            d[0] -= L*np.round(d[0]/L); d[1] -= L*np.round(d[1]/L)
                            if d[0]*d[0] + d[1]*d[1] + d[2]*d[2] < dmin*dmin:
                                ok = False; break
                        if not ok: break
                    if not ok: break
                if not ok: break
            if ok:
                pts[n] = c; grid.setdefault(ci, []).append(n); n += 1
                if n >= n_target:
                    break
    return Atoms('C'*n, positions=pts[:n], cell=[L, L, t_max], pbc=True)


kx = sfft.fftfreq(NG, DX)
KX, KY = np.meshgrid(kx, kx, indexing='xy')
K2 = KX**2 + KY**2
KMAX_AA = (2/3)*(1/(2*DX))
AA = (np.sqrt(K2) < KMAX_AA)
PROP = (np.exp(-1j*np.pi*LAM*DZ*K2)*AA).astype(np.complex64)
APER = np.clip((ALPHA/LAM - np.sqrt(K2))/(1/L) + 0.5, 0, 1)
xg = (np.arange(NG))*DX
XG, YG = np.meshgrid(xg, xg, indexing='xy')


def probe(x0, y0):
    A = APER*np.exp(-2j*np.pi*(KX*x0 + KY*y0))
    p = sfft.ifft2(A)
    return (p/np.sqrt((np.abs(p)**2).sum())).astype(np.complex64)


KC = 35e-3/LAM
ICROP = int(np.ceil(KC*L))
M = 2*ICROP + 2
DXC = L/M
MP = int(round(LAM/K_DET/DXC))


def to_detector(psi):
    F = sfft.fft2(psi)
    Fs = sfft.fftshift(F)
    c = NG//2
    sub = Fs[c-M//2:c+M//2, c-M//2:c+M//2]
    low = sfft.ifft2(sfft.ifftshift(sub))
    pad = np.zeros((MP, MP), np.complex128)
    pad[:M, :M] = low
    pad = np.roll(pad, ((MP - M)//2, (MP - M)//2), axis=(0, 1))
    D = np.abs(sfft.fftshift(sfft.fft2(pad)))**2
    cc = MP//2
    D = D[cc-NDET//2:cc+NDET//2, cc-NDET//2:cc+NDET//2]
    D = gaussian_filter(D, PSF)
    return D/D.sum()*NE


def localisation(psi, x0, y0, rad=25.0):
    I = np.abs(psi)**2
    dx = (XG - x0 + L/2) % L - L/2; dy = (YG - y0 + L/2) % L - L/2
    return float(I[np.hypot(dx, dy) > rad].sum()/I.sum())


def probe_tilted(x0, y0, tx, ty):
    kt_x = tx*K_DET/LAM; kt_y = ty*K_DET/LAM
    Ak = np.clip((ALPHA/LAM - np.sqrt((KX - kt_x)**2 + (KY - kt_y)**2))/(1/L) + 0.5, 0, 1)
    p = sfft.ifft2(Ak*np.exp(-2j*np.pi*(KX*x0 + KY*y0)))
    return (p/np.sqrt((np.abs(p)**2).sum())).astype(np.complex64)


def cell_grid(a1, a2, n=4):
    a1 = np.asarray(a1, float); a2 = np.asarray(a2, float)
    g = np.array([((i + 0.5)/n)*a1 + ((j + 0.5)/n)*a2 for i in range(n) for j in range(n)])
    g -= g.mean(0)
    return tuple(map(tuple, g))


CELLS = {'graphite': ((2.46, 0.0), (-1.23, 2.1304)), 'rutile': ((4.594, 0.0), (0.0, 4.594)),
         'tin': ((4.24, 0.0), (0.0, 4.24)), 'pt011': ((3.924, 0.0), (0.0, 3.924/np.sqrt(2))),
         'aC': ((24.0, 0.0), (0.0, 24.0))}


def run(name, atoms, t_max, a_field, plane_every_A=10.0,
        offs=((-0.7, -0.7), (0.7, -0.7), (-0.7, 0.7), (0.7, 0.7)),
        dithers=((0.0, 0.0), (0.5, 0.5))):
    nsl = int(round(t_max/DZ))
    t0 = time.time()
    if len(atoms) == 0:
        V = np.zeros((nsl, NG, NG), np.float32)
    else:
        pot = abtem.Potential(atoms, gpts=(NG, NG), slice_thickness=DZ, projection='infinite',
                              parametrization='lobato')
        V = np.asarray(pot.build(lazy=False).array, dtype=np.float32)
        V = np.ascontiguousarray(np.transpose(V, (0, 2, 1)))
    mip = float(V.mean()/DZ)
    print(f'  potential {V.shape} built in {time.time()-t0:.0f} s; IAM mean potential {mip:.2f} V', flush=True)
    nsl = V.shape[0]
    every = max(1, int(round(plane_every_A/DZ)))
    planes = list(range(every - 1, nsl, every))
    vac = to_detector(probe(L/2, L/2))
    tb_bin = build_template_edge([vac])
    cm0 = np.array(com(vac)); e0 = fit_ellipse(vac)
    nP, nD = len(offs), len(dithers)
    shift = np.full((len(planes), nP, nD, 2, 3, 2), np.nan)
    qc = np.full((len(planes), nP, nD, 2, 4), np.nan)
    loc = np.full((len(planes), nP), np.nan)
    for ip, (ox, oy) in enumerate(offs):
        x0, y0 = L/2 + ox, L/2 + oy
        dxw = ((XG - x0 + L/2) % L) - L/2
        Vf = (a_field*dxw*DZ).astype(np.float32)
        for idt, (tx, ty) in enumerate(dithers):
            for fi in (0, 1):
                psi = probe_tilted(x0, y0, tx, ty)
                pi = 0
                for s in range(nsl):
                    ph = SIG*(V[s] + Vf) if fi else SIG*V[s]
                    psi = sfft.ifft2(sfft.fft2(psi*np.exp(1j*ph).astype(np.complex64), workers=2)*PROP, workers=2)
                    if pi < len(planes) and s == planes[pi]:
                        D = to_detector(psi)
                        eb = get_disk_edge(D)
                        bx, by, _ = track_xcorr(D, tb_bin, edge=eb)
                        ex, ey, _, _ = fit_ellipse(D, edge=eb)
                        cx, cy = com(D)
                        shift[pi, ip, idt, fi] = [[bx - tx, by - ty], [ex - e0[0] - tx, ey - e0[1] - ty],
                                                   [cx - cm0[0] - tx, cy - cm0[1] - ty]]
                        rmax, _ = edge_contamination(D, NDET/2 + bx, NDET/2 + by, ALPHA/K_DET)
                        q = qc_metrics(eb, NDET/2 + bx, NDET/2 + by, ALPHA/K_DET)
                        rr = np.hypot(*(np.indices(D.shape) - NDET/2))
                        qc[pi, ip, idt, fi] = [rmax, q['inlier_frac'], q['coverage'],
                                               D[rr < 0.9*ALPHA/K_DET].sum()/D.sum()]
                        if fi == 0 and idt == 0:
                            loc[pi, ip] = localisation(psi, x0, y0)
                        pi += 1
        print(f'  {name}: position {ip+1}/{nP} done ({time.time()-t0:.0f} s)', flush=True)
    thick = (np.array(planes) + 1)*DZ
    np.savez(f'{OUT}/{name}.npz', thick=thick, shift=shift, qc=qc, loc=loc, mip=mip, a_field=a_field,
             trackers=np.array(['bin_xcorr', 'ellipse', 'com']))
    return thick


SPECS = {
    'aC':           dict(t=300.0, cell='aC', build=lambda t: amorphous_c(t)),
    'graphite_001': dict(t=300.0, tf=100.0, cell='graphite', build=lambda t: slab(graphite(), [0, 0, 1], (0.0, 0.0), t)),
    'TiO2r_001':    dict(t=300.0, cell='rutile', build=lambda t: slab(rutile(), [0, 0, 1], (0.0, 0.0), t)),
    'TiO2r_t5':     dict(t=300.0, cell='rutile', build=lambda t: slab(rutile(), [0, 0, 1], (5.0, 17.0), t)),
    'TiO2r_t10':    dict(t=300.0, cell='rutile', build=lambda t: slab(rutile(), [0, 0, 1], (10.0, 63.0), t)),
    'TiN_001':      dict(t=300.0, cell='tin', build=lambda t: slab(tin(), [0, 0, 1], (0.0, 0.0), t)),
    'TiN_t5':       dict(t=300.0, cell='tin', build=lambda t: slab(tin(), [0, 0, 1], (5.0, 17.0), t)),
    'TiN_t10':      dict(t=300.0, cell='tin', build=lambda t: slab(tin(), [0, 0, 1], (10.0, 63.0), t)),
    'Pt_011':       dict(t=300.0, tf=120.0, cell='pt011', build=lambda t: slab(pt(), [0, 1, 1], (0.0, 0.0), t)),
    'Pt_t5':        dict(t=300.0, tf=120.0, cell='pt011', build=lambda t: slab(pt(), [0, 1, 1], (5.0, 17.0), t)),
    'Pt_t10':       dict(t=300.0, tf=120.0, cell='pt011', build=lambda t: slab(pt(), [0, 1, 1], (10.0, 63.0), t)),
}

if __name__ == '__main__':
    import os
    os.makedirs(OUT, exist_ok=True)
    names = sys.argv[1:] if len(sys.argv) > 1 else list(SPECS)
    for n in names:
        sp = SPECS[n]
        t0 = time.time()
        at = sp['build'](sp['t'])
        print(f'{n}: {len(at)} atoms, t_max {sp["t"]} A', flush=True)
        tf = sp.get('tf', sp['t'])
        a_field = 5.0/((LAM*SIG/(2*np.pi))*tf/K_DET)
        run(n, at, sp['t'], a_field, offs=cell_grid(*CELLS[sp['cell']]))
        vname = f"vacuum_t{int(sp['t'])}" + (f"_f{int(tf)}" if tf != sp['t'] else '')
        if not os.path.exists(f'{OUT}/{vname}.npz'):
            run(vname, Atoms(cell=[L, L, sp['t']], pbc=True), sp['t'], a_field)
        print(f'{n}: done in {time.time()-t0:.0f} s', flush=True)

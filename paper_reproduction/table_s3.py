"""Table S3: median error (px) per QC condition.
Usage (inside results_v1.1/): python ../table_s3.py tracker_benchmark_results.json"""
import json, sys, numpy as np, collections
res = json.load(open(sys.argv[1]))
keys = [k for k in ('ellipse', 'xcorr', 'com') if k in res[0]]
def err(r, k):
    v = r[k]; return np.hypot(*v) if np.all(np.isfinite(v)) else 1e3
def row(lab, sel):
    out = [lab, f'n={len(sel)}']
    for k in keys:
        e = np.array([err(r, k) for r in sel]) if sel else np.array([np.nan])
        out.append(f'{np.median(e):.2f}')
    print(' | '.join(out))
inc = [r for r in res if r['name'].startswith('disc_')]
print('cols:', keys)
row('redge<0.1 & inl>=0.95', [r for r in inc if r['r_edge_max'] < 0.1 and r['inlier_frac'] >= 0.95])
for lo, hi in ((0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 99)):
    row(f'redge {lo}-{hi}', [r for r in inc if lo <= r['r_edge_max'] < hi])
coh = [r for r in res if r['name'].startswith('coh_')]
row('coherent all', coh)
row('coherent redge 0.1-0.8', [r for r in coh if 0.1 <= r['r_edge_max'] < 0.8])
bands = [r for r in res if r['name'].startswith('band_')]
row('bands inl 0.90-0.95', [r for r in bands if 0.90 <= r['inlier_frac'] < 0.95])
row('bands inl <0.90', [r for r in bands if r['inlier_frac'] < 0.90])
row('dose 1e4', [r for r in res if r['name'] == 'dose_10000'])
row('dose 3e4', [r for r in res if r['name'] == 'dose_30000'])
for d in ('1.0', '1.5', '2.0', '3.0', '5.0'):
    row(f'dR {d}', [r for r in res if r['name'] == f'dR_{d}'])

"""Validation of the contamination test of tracker_qc_report.py on synthetic patterns with known shifts: compares the
measured contamination shift with the true tracker error.  ->  results_v1.1/impact_validation.json"""
import os, sys, json
import numpy as np
from joblib import Parallel, delayed
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import disc_tracking as dt
import tracker_benchmark as tb

T = ('edge_binary', 'ellipse')
NREP = 30
rng0 = np.random.default_rng(0)
REF = dt.build_reference([tb.make_pattern(rng0, np.zeros(2)) for _ in range(25)])
CONDS = ([dict(name='clean')] + [dict(name=f'disc_n1_r{r}', r=r, ndisc=1) for r in (0.1, 0.2, 0.3, 0.5, 0.8)]
         + [dict(name=f'disc_n4_r{r}', r=r, ndisc=4) for r in (0.2, 0.5)]
         + [dict(name=f'coh_n1_r{r}', r=r, ndisc=1, coh=True) for r in (0.2, 0.5)]
         + [dict(name=f'band_n{n}_D{D}', D=D, nband=n) for n in (1, 2) for D in (0.3, 0.6, 0.9)]
         + [dict(name='combo_r0.4_D0.8', r=0.4, ndisc=1, D=0.8, nband=1), dict(name='dose_3e4', Ne=3e4)])


def one(cond, seed):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi); mag = rng.uniform(0, 3)
    s = np.array([mag * np.cos(ang), mag * np.sin(ang)])
    p = tb.make_pattern(rng, s, **{k: v for k, v in cond.items() if k != 'name'})
    est = {t: np.array(dt.track(p, t, REF), float) for t in T}
    ci = dt.contamination_impact(p, REF, est, T)
    return {t: dict(err=(est[t] - s).tolist(), imp=ci['impact'][t].tolist(), res=(ci['clean'][t] - s).tolist()) for t in T} | \
        dict(spread=ci['spread'], overlap=ci['overlap'], modulation=ci['modulation'])


if __name__ == '__main__':
    jobs = [(c, 1000 * i + k) for i, c in enumerate(CONDS) for k in range(NREP)]
    res = Parallel(n_jobs=-1)(delayed(one)(c, sd) for c, sd in jobs)
    out = {}
    for (c, _), r in zip(jobs, res):
        out.setdefault(c['name'], []).append(r)
    os.makedirs(os.path.join(HERE, 'results_v1.1'), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, 'results_v1.1', 'impact_validation.json'), 'w'))
    m = lambda a: np.hypot(*np.array(a).T)
    print(f"{'condition':18s} " + ' | '.join(f'{t:>13s} err  imp  res' for t in T) + ' | spread')
    for name, rs in out.items():
        row = f'{name:18s} '
        for t in T:
            e, i, q = (m([r[t][k] for r in rs]) for k in ('err', 'imp', 'res'))
            row += f'| {np.median(e):13.2f} {np.median(i):4.2f} {np.median(q):4.2f} '
        print(row + f"| {np.median([r['spread'] for r in rs]):.2f}")

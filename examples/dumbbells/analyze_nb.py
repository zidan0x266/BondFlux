#!/usr/bin/env python3
"""Bond autocorrelation n_b(t) from a partners dump of fix bond/flux.

n_b(t) = fraction of bonds present at time t0 that are present at t0 + t
(same A atom bonded to the same B atom), averaged over time origins t0.
A bond = the nearest bondable partner within rbond (column f_ID[1] of the fix).

usage: analyze_nb.py partners.dump [dt_frame] [tmax]
"""

import sys

import numpy as np


def read_partners(path):
    frames = []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('ITEM: NUMBER OF ATOMS'):
                n = int(f.readline())
            elif line.startswith('ITEM: ATOMS'):
                rows = np.array([f.readline().split() for _ in range(n)], dtype=float)
                frames.append(rows[np.argsort(rows[:, 0]), 1].astype(np.int64))
    return np.array(frames)


def main():
    path = sys.argv[1]
    dt_frame = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    p = read_partners(path)
    nframe = len(p)
    tmax = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5 * nframe * dt_frame
    nlag = min(nframe - 1, int(tmax / dt_frame))
    origins = range(0, nframe - nlag, max(1, nlag // 50))

    nb = np.zeros(nlag + 1)
    for t0 in origins:
        bonded = p[t0] > 0
        same = (p[t0:t0 + nlag + 1] == p[t0]) & bonded
        nb += same.sum(1) / max(1, bonded.sum())
    nb /= len(origins)
    t = np.arange(nlag + 1) * dt_frame

    # exponential fit on 0.9 > n_b > 0.05
    sel = (nb < 0.9) & (nb > 0.05)
    tau = -1.0 / np.polyfit(t[sel], np.log(nb[sel]), 1)[0] if sel.sum() > 3 else float('nan')
    print(f'{path}: {nframe} frames, bonded fraction of A {np.mean(p > 0):.3f}, '
          f'fitted tau = {tau:.1f}')
    for frac in (0.5, 0.2, 0.1):
        i = np.argmax(nb < frac) if np.any(nb < frac) else -1
        if i > 0:
            print(f'   n_b falls below {frac} at t = {t[i]:.0f} (exp(-t/tau) gives {-tau * np.log(frac):.0f})')
    np.savetxt(path.replace('.dump', '_nb.txt'), np.column_stack([t, nb]), header='t n_b(t)')


if __name__ == '__main__':
    main()

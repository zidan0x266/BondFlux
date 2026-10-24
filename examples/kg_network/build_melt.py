#!/usr/bin/env python3
"""Write a LAMMPS data file for a Kremer-Grest melt of sticker chains.

Each chain has N beads; beads at the positions given by --stickers (1-based)
are stickers (atom type 2), all others are backbone beads (type 1).  Backbone
bonds are type 1; bond type 2 is reserved for the dynamic bonds created by
fix bond/flux.  Chains are random walks with bond length 0.97; overlaps are
removed later in LAMMPS with a soft push-off.

usage: build_melt.py [--chains 600] [--length 20] [--stickers 2,6,10,14,18]
                     [--density 0.85] [--seed 1] [--out melt.data]
"""

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chains', type=int, default=600)
    ap.add_argument('--length', type=int, default=20)
    ap.add_argument('--stickers', default='2,6,10,14,18')
    ap.add_argument('--density', type=float, default=0.85)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='melt.data')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    nch, n = args.chains, args.length
    sticky = {int(s) - 1 for s in args.stickers.split(',') if s}
    natoms = nch * n
    box = (natoms / args.density) ** (1.0 / 3.0)

    atoms, bonds = [], []
    for c in range(nch):
        r = rng.uniform(0.0, box, 3)
        for m in range(n):
            if m > 0:
                step = rng.normal(size=3)
                r = r + 0.97 * step / np.linalg.norm(step)
            aid = c * n + m + 1
            img = np.floor(r / box).astype(int)
            atoms.append((aid, c + 1, 2 if m in sticky else 1, *(r - img * box), *img))
            if m > 0:
                bonds.append((len(bonds) + 1, 1, aid - 1, aid))

    with open(args.out, 'w') as f:
        f.write(f'Kremer-Grest sticker melt: {nch} chains x {n} beads, stickers at '
                f'{sorted(s + 1 for s in sticky)}\n\n')
        f.write(f'{natoms} atoms\n2 atom types\n{len(bonds)} bonds\n2 bond types\n\n')
        f.write(f'0.0 {box:.10f} xlo xhi\n0.0 {box:.10f} ylo yhi\n0.0 {box:.10f} zlo zhi\n\n')
        f.write('Masses\n\n1 1.0\n2 1.0\n\nAtoms # bond\n\n')
        for a in atoms:
            f.write(f'{a[0]} {a[1]} {a[2]} {a[3]:.6f} {a[4]:.6f} {a[5]:.6f} {a[6]} {a[7]} {a[8]}\n')
        f.write('\nBonds\n\n')
        for b in bonds:
            f.write(f'{b[0]} {b[1]} {b[2]} {b[3]}\n')
    print(f'wrote {args.out}: {natoms} atoms, box {box:.4f}, '
          f'{nch * len(sticky)} stickers')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Contact diagnostics of fix bond/flux (BondFlux) in mc mode (keyword contact).

A gas of A-B dimers (dynamic bond type 1) and free A stickers can exchange bonds by shifts.
With a barrier of 60 kT no move is ever accepted, so the topology stays fixed, and the
contact statistics of the fix can be recounted exactly from the positions written at every
check: a B sticker is in contact while its bond is inside the reaction window and a free A
is within the window of B; A stickers have no candidates (no free B, swaps off).

Checks, for Nevery = Ncheck (no episode can be missed) and Nevery = 10 Ncheck:
  contact time, completed episodes and missed episodes equal the recount.

A second system tests permanent neighbors: rings A-B-A'-B' with dynamic bonds A-B and A'-B'
and permanent bonds B-A' and B'-A, plus free A stickers, with swaps on and the atoms sorted
at every step.  A permanent neighbor is never a candidate, although it is a bondable sticker
inside the reaction window; every sticker of a ring is bonded, so the candidates of a ring
sticker are free A and the stickers of other rings.

Run:  python3 test_contact.py ;  mpirun -np 4 python3 test_contact.py
"""

import argparse
import os
import sys

import numpy as np
from lammps import lammps

RC = 1.3
BOX = (22.0, 22.0, 11.0)
STEPS = 300
DT = 0.005


def write_data(path, seed=11):
    """A-B dimers on 15 % of the x-pairs of a cubic lattice (spacing 1.1), free A on 15 %
    of the remaining sites"""
    rng = np.random.default_rng(seed)
    a = 1.1
    nx, ny, nz = int(BOX[0] / a), int(BOX[1] / a), int(BOX[2] / a)
    atoms, bonds = [], []
    for z in range(nz):
        for y in range(ny):
            for x in range(0, nx - 1, 2):
                p = np.array([x, y, z]) * a + 0.3
                if rng.random() < 0.15:
                    atoms.append((len(atoms) + 1, 1, p))
                    atoms.append((len(atoms) + 1, 2, p + (a, 0.0, 0.0)))
                    bonds.append((len(atoms) - 1, len(atoms)))
                else:
                    for q in (p, p + (a, 0.0, 0.0)):
                        if rng.random() < 0.15:
                            atoms.append((len(atoms) + 1, 1, q))
    with open(path, 'w') as f:
        f.write(f'dimers and free stickers\n\n{len(atoms)} atoms\n{len(bonds)} bonds\n'
                f'2 atom types\n1 bond types\n\n0 {BOX[0]} xlo xhi\n0 {BOX[1]} ylo yhi\n'
                f'0 {BOX[2]} zlo zhi\n\nMasses\n\n1 1.0\n2 1.0\n\nAtoms # bond\n\n')
        for i, t, p in atoms:
            f.write(f'{i} {i} {t} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')
        f.write('\nBonds\n\n')
        for m, (i, j) in enumerate(bonds):
            f.write(f'{m + 1} 1 {i} {j}\n')
    return bonds


def write_ring_data(path, seed=12):
    """rings A-B-A'-B' (square of side 1.1, dynamic bonds A-B and A'-B' of type 1, permanent
    bonds B-A' and B'-A of type 2) on 15 % of the 2x2 blocks of a square lattice in each
    layer, free A on 15 % of the sites of the other blocks"""
    rng = np.random.default_rng(seed)
    a = 1.1
    nx, ny, nz = int(BOX[0] / a), int(BOX[1] / a), int(BOX[2] / a)
    atoms, dyn, perm = [], [], []
    for z in range(nz):
        for y in range(0, ny - 1, 2):
            for x in range(0, nx - 1, 2):
                p = np.array([x, y, z]) * a + 0.3
                corners = [p, p + (a, 0.0, 0.0), p + (a, a, 0.0), p + (0.0, a, 0.0)]
                if rng.random() < 0.15:
                    n0 = len(atoms)
                    for t, q in zip((1, 2, 1, 2), corners):
                        atoms.append((len(atoms) + 1, t, q))
                    ia, ib, ia2, ib2 = n0 + 1, n0 + 2, n0 + 3, n0 + 4
                    dyn += [(ia, ib), (ia2, ib2)]
                    perm += [(ib, ia2), (ib2, ia)]
                else:
                    for q in corners:
                        if rng.random() < 0.15:
                            atoms.append((len(atoms) + 1, 1, q))
    with open(path, 'w') as f:
        f.write(f'rings and free stickers\n\n{len(atoms)} atoms\n{len(dyn) + len(perm)} bonds\n'
                f'2 atom types\n2 bond types\n\n0 {BOX[0]} xlo xhi\n0 {BOX[1]} ylo yhi\n'
                f'0 {BOX[2]} zlo zhi\n\nMasses\n\n1 1.0\n2 1.0\n\nAtoms # bond\n\n')
        for i, t, p in atoms:
            f.write(f'{i} {i} {t} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')
        f.write('\nBonds\n\n')
        for m, (i, j) in enumerate(dyn):
            f.write(f'{m + 1} 1 {i} {j}\n')
        for m, (i, j) in enumerate(perm):
            f.write(f'{len(dyn) + m + 1} 2 {i} {j}\n')
    return dyn, perm


def run(nevery, outdir, me, rings=False):
    name = 'rings' if rings else 'contact'
    data = os.path.join(outdir, f'{name}.data')
    dump = os.path.join(outdir, f'{name}_{nevery}.dump')
    bonds, perm = [], []
    if me == 0:
        if rings:
            bonds, perm = write_ring_data(data)
        else:
            bonds = write_data(data)
    moves = 'atom_modify sort 1 0.55\nneigh_modify every 1 delay 0 check no' if rings else ''
    lmp = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    lmp.commands_string(f"""
units lj
atom_style bond
boundary p p p
special_bonds lj/coul 0.0 1.0 1.0
read_data {data} extra/bond/per/atom 1 extra/special/per/atom 2
pair_style lj/cut {2 ** (1 / 6)}
pair_coeff * * 1.0 1.0
pair_modify shift yes
bond_style harmonic
bond_coeff * 100.0 1.1
neighbor 0.3 bin
{moves}
comm_modify cutoff 3.0
timestep {DT}
velocity all create 1.0 4928 loop geom
fix nve all nve
fix lang all langevin 1.0 1.0 1.0 9134
fix fx all bond/flux mc {nevery} 1 {RC} 1.0 2718 bondable 1 2 swap {'yes' if rings else 'no'} barrier 60 contact 1
dump d all custom 1 {dump} id type x y z
dump_modify d sort id format float %.17g
run {STEPS}
""")
    vec = [lmp.extract_fix('fx', 0, 1, i) for i in range(16)]
    lmp.close()
    if me != 0:
        return True

    # recount from the dump: positions of steps 1..STEPS are those of the contact checks
    frames = []
    with open(dump) as f:
        lines = f.read().split('ITEM: TIMESTEP\n')[1:]
    for block in lines:
        rows = block.split('\n')
        step = int(rows[0])
        natoms = int(rows[2])
        arr = np.array([r.split() for r in rows[8:8 + natoms]], dtype=float)
        frames.append((step, arr))
    # a bonded sticker is in contact while its bond is in the window and a sticker of the
    #   other type (free, or also bonded if swaps are on) is in the window, other than its
    #   partner and its permanent neighbors
    box = np.array(BOX)
    typ = frames[0][1][:, 1].astype(int)
    n = len(typ)
    partner = np.full(n, -1)
    for i, j in bonds:
        partner[i - 1], partner[j - 1] = j - 1, i - 1
    allowed = typ[:, None] != typ[None, :]
    allowed[np.arange(n), np.arange(n)] = False
    allowed[partner >= 0, partner[partner >= 0]] = False
    for i, j in perm:
        allowed[i - 1, j - 1] = allowed[j - 1, i - 1] = False
    if not rings:
        allowed[:, partner >= 0] = False         # swaps off: free candidates only
    centers = np.where(partner >= 0)[0]
    c = np.zeros(n, dtype=bool)
    seen = np.zeros(n, dtype=bool)
    ctime, episodes, missed = 0, 0, 0
    for step, arr in frames:
        if step == 0:
            continue
        x = arr[:, 2:5]
        d = x[centers][:, None, :] - x[None, :, :]
        d -= box * np.round(d / box)
        near = np.any((np.einsum('ijk,ijk->ij', d, d) < RC * RC) & allowed[centers], axis=1)
        db = x[centers] - x[partner[centers]]
        db -= box * np.round(db / box)
        inwin = np.einsum('ij,ij->i', db, db) < RC * RC
        now = np.zeros(n, dtype=bool)
        now[centers] = near & inwin
        start = now & ~c
        seen[start] = False
        if step % nevery == 0:
            seen[now] = True
        ctime += int(now.sum())
        ended = c & ~now
        episodes += int(ended.sum())
        missed += int((ended & ~seen).sum())
        c = now

    fix = (vec[11], vec[12], vec[13])
    ref = (ctime * DT, episodes, missed)
    ok = (abs(fix[0] - ref[0]) < 1e-9 * max(1.0, ref[0]) and fix[1] == ref[1] and fix[2] == ref[2]
          and vec[2] == 0 and vec[3] == 0)
    if nevery == 1:
        ok = ok and (missed == 0)
    print(f'{name:8s} Nevery={nevery:3d}: fix contact time {fix[0]:.4f}, episodes {fix[1]:.0f}, missed '
          f'{fix[2]:.0f} (f_miss {vec[15]:.3f}); recount {ref[0]:.4f}, {ref[1]}, {ref[2]}; '
          f'accepted moves {vec[2] + vec[3]:.0f}')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='.')
    args = ap.parse_args()
    # this instance stays open, so that all ranks finalize MPI together at the end
    probe = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    me, nprocs = probe.extract_setting('world_rank'), probe.extract_setting('world_size')
    ok = all([run(1, args.outdir, me), run(10, args.outdir, me),
              run(10, args.outdir, me, rings=True)])
    if me == 0:
        print('PASS' if ok else 'FAIL', f'contact diagnostics np={nprocs}')
    probe.finalize()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

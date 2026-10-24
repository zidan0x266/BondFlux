#!/usr/bin/env python3
"""Contact diagnostics of fix bond/flux (BondFlux) in mc mode (keyword contact).

A gas of A-B dimers (dynamic bond type 1) and free A stickers can exchange bonds by shifts.
With a barrier of 60 kT no move is ever accepted, so the topology stays fixed, and the
contact statistics of the fix can be recounted exactly from the positions written at every
check: a B sticker is in contact while its bond is inside the reaction window and a free A
is within the window of B; A stickers have no candidates (no free B, swaps off).

Checks, for Nevery = Ncheck (no episode can be missed) and Nevery = 10 Ncheck:
  contact time, completed episodes and missed episodes equal the recount.

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


def run(nevery, outdir, me):
    data = os.path.join(outdir, 'contact.data')
    dump = os.path.join(outdir, f'contact_{nevery}.dump')
    bonds = write_data(data) if me == 0 else None
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
bond_coeff 1 100.0 1.1
neighbor 0.3 bin
comm_modify cutoff 3.0
timestep {DT}
velocity all create 1.0 4928 loop geom
fix nve all nve
fix lang all langevin 1.0 1.0 1.0 9134
fix fx all bond/flux mc {nevery} 1 {RC} 1.0 2718 bondable 1 2 swap no barrier 60 contact 1
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
    box = np.array(BOX)
    typ = frames[0][1][:, 1].astype(int)
    bonded = np.zeros(len(typ), dtype=bool)
    bpart = {}
    for i, j in bonds:
        bonded[[i - 1, j - 1]] = True
        bpart[j - 1] = i - 1                     # B sticker -> its A partner
    b_idx = np.array(sorted(bpart))
    a_free = np.where((typ == 1) & ~bonded)[0]
    c = np.zeros(len(typ), dtype=bool)
    seen = np.zeros(len(typ), dtype=bool)
    ctime, episodes, missed = 0, 0, 0
    for step, arr in frames:
        if step == 0:
            continue
        x = arr[:, 2:5]
        d = x[b_idx][:, None, :] - x[a_free][None, :, :]
        d -= box * np.round(d / box)
        near = np.any(np.einsum('ijk,ijk->ij', d, d) < RC * RC, axis=1)
        db = x[b_idx] - x[[bpart[b] for b in b_idx]]
        db -= box * np.round(db / box)
        inwin = np.einsum('ij,ij->i', db, db) < RC * RC
        now = np.zeros(len(typ), dtype=bool)
        now[b_idx] = near & inwin
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
    print(f'Nevery={nevery:3d}: fix contact time {fix[0]:.4f}, episodes {fix[1]:.0f}, missed '
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
    ok = all([run(1, args.outdir, me), run(10, args.outdir, me)])
    if me == 0:
        print('PASS' if ok else 'FAIL', f'contact diagnostics np={nprocs}')
    probe.finalize()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

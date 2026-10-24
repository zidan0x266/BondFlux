#!/usr/bin/env python3
"""Tests for fix bond/flux (BondFlux) in 3body mode (Sciortino three-body swap potential).

  1. forces = -gradient of the total potential energy (central finite differences)
  2. virial = -strain derivative of the fix energy (box scaling and shear)
  3. flat energy landscape for lambda = 1: a second partner approaching a bonded
     pair does not change the energy (Sciortino, Eur. Phys. J. E 40, 3 (2017))
  4. forces identical on any number of MPI ranks (written to forces_npN.npy and
     compared when both 1- and N-rank files exist)

Run:  python3 test_3body.py ;  mpirun -np 4 python3 test_3body.py ;  python3 test_3body.py --flat
"""

import argparse
import os
import sys

import numpy as np
from lammps import LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, lammps

RC3 = 1.3


def setup(lmp, grid, box, npow, lam, pos, types):
    lmp.commands_string(f"""
units lj
atom_style bond
boundary p p p
processors {grid[0]} {grid[1]} {grid[2]}
region box block 0 {box[0]} 0 {box[1]} 0 {box[2]}
create_box 3 box bond/types 1 extra/bond/per/atom 2 extra/special/per/atom 4
mass * 1.0
pair_style lj/cut {2 ** (1 / 6)}
pair_coeff * * 1.0 1.0
pair_coeff 1 2 0.0 1.0
pair_modify shift yes
bond_style harmonic
bond_coeff 1 100.0 1.0
""")
    for p, t in zip(pos, types):
        lmp.command(f'create_atoms {t} single {p[0]} {p[1]} {p[2]} units box')
    lmp.commands_string(f"""
group stick type 1 2
comm_modify cutoff 2.5
neighbor 0.3 bin
fix tb stick bond/flux 3body 1.0 1.0 {npow} {RC3} {lam} bondable 1 2
compute pfix all pressure NULL fix
thermo_style custom step pe c_pfix[1] c_pfix[2] c_pfix[3] c_pfix[4]
thermo_modify norm no
run 0
""")


def energies(lmp):
    return lmp.get_thermo('pe'), lmp.extract_fix('tb', 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npow', type=float, default=12.0)
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--outdir', default='.')
    ap.add_argument('--flat', action='store_true', help='run only the flat-landscape check')
    args = ap.parse_args()
    if args.flat:
        return flat_landscape()

    lmp = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    me = lmp.extract_setting('world_rank')
    nprocs = lmp.extract_setting('world_size')
    grid = {1: (1, 1, 1), 2: (2, 1, 1), 4: (2, 2, 1)}[nprocs]
    box = [12.0 * g if g > 1 else 12.0 for g in grid]
    center = np.array([b / 2.0 for b in box]) + 0.017

    # random cluster of A (1) and B (2) stickers plus inert beads (3), minimum distance 0.97
    rng = np.random.default_rng(7)
    pos, types = [], []
    while len(pos) < 14:
        p = rng.uniform(-1.6, 1.6, 3)
        if all(np.linalg.norm(p - q) > 0.97 for q in pos):
            pos.append(p)
            types.append([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 3, 3, 1][len(pos) - 1])
    pos = np.array(pos) + center
    natoms = len(pos)

    setup(lmp, grid, box, args.npow, args.lam, pos, types)
    f = np.ctypeslib.as_array(lmp.gather_atoms('f', 1, 3)).reshape(natoms, 3).copy()
    pe0, efix0 = energies(lmp)
    vol = box[0] * box[1] * box[2]
    pfix = lmp.numpy.extract_compute('pfix', LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR)
    wfix = np.array([pfix[0], pfix[1], pfix[2], pfix[3]]) * vol
    stats = [lmp.extract_fix('tb', 0, 1, i) for i in range(4)]

    # 1. finite-difference forces of the total energy
    h = 1.0e-6
    fd = np.zeros_like(f)
    for i in range(natoms):
        for d, name in enumerate('xyz'):
            ep = []
            for s in (+1.0, -1.0):
                lmp.command(f'set atom {i + 1} {name} {pos[i][d] + s * h}')
                lmp.command('run 0')
                ep.append(lmp.get_thermo('pe'))
            lmp.command(f'set atom {i + 1} {name} {pos[i][d]}')
            fd[i, d] = -(ep[0] - ep[1]) / (2.0 * h)
    ferr = np.max(np.abs(fd - f)) / max(1.0, np.max(np.abs(f)))

    # 2. virial from strain derivatives of the fix energy
    eps = 1.0e-6
    wfd = []
    for dim in 'xyz':
        e = []
        for s in (+1.0, -1.0):
            lmp.command(f'change_box all {dim} scale {1.0 + s * eps} remap units box')
            lmp.command('run 0')
            e.append(energies(lmp)[1])
            lmp.command(f'change_box all {dim} scale {1.0 / (1.0 + s * eps)} remap units box')
        wfd.append(-(e[0] - e[1]) / (2.0 * eps))
    lmp.command('change_box all triclinic')
    e = []
    for s in (+1.0, -1.0):
        lmp.command(f'change_box all xy final {s * eps * box[1]} remap units box')
        lmp.command('run 0')
        e.append(energies(lmp)[1])
    lmp.command('change_box all xy final 0.0 remap units box')
    wfd.append(-(e[0] - e[1]) / (2.0 * eps))
    wfd = np.array(wfd)
    werr = np.max(np.abs(wfd - wfix)) / max(1.0, np.max(np.abs(wfix)))
    lmp.finalize()

    if me != 0:
        return 0

    out = os.path.join(args.outdir, f'forces_n{args.npow:g}_l{args.lam:g}_np{nprocs}.npy')
    np.save(out, f)
    mpi_msg = ''
    ref = os.path.join(args.outdir, f'forces_n{args.npow:g}_l{args.lam:g}_np1.npy')
    mpierr = 0.0
    if nprocs > 1 and os.path.exists(ref):
        mpierr = np.max(np.abs(np.load(ref) - f))
        mpi_msg = f', max |F(np{nprocs}) - F(np1)| = {mpierr:.2e}'

    print(f'3body n={args.npow:g} lambda={args.lam:g} np={nprocs}: E_fix={efix0:.6f} '
          f'(E2={stats[0]:.4f} E3={stats[1]:.4f}), bonded pairs={stats[2]:.0f}, '
          f'multiply bonded={stats[3]:.0f}')
    print(f'   force check: max |F_fd - F| / max|F| = {ferr:.2e}')
    print(f'   virial check: W_fix = {np.array2string(wfix, precision=6)}')
    print(f'                 W_fd  = {np.array2string(wfd, precision=6)}  rel.err {werr:.2e}'
          f'{mpi_msg}')
    ok = (ferr < 1e-5) and (werr < 1e-5) and (mpierr < 1e-10)
    print('PASS' if ok else 'FAIL', f'3body n={args.npow:g} lambda={args.lam:g} np={nprocs}')
    return 0 if ok else 1


def flat_landscape():
    """A-B bonded at rmin; a second B approaches A from the opposite side."""
    lmp = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    npow = 100.0
    rmin = 2 ** (1 / npow)
    pos = [np.array([6.0, 6.0, 6.0]), np.array([6.0 + rmin, 6.0, 6.0]),
           np.array([4.0, 6.0, 6.0])]
    setup(lmp, (1, 1, 1), [12.0, 12.0, 12.0], npow, 1.0, pos, [1, 2, 2])
    out = []
    for r in np.linspace(rmin + 1e-3, RC3 + 0.05, 25):
        lmp.command(f'set atom 3 x {6.0 - r}')
        lmp.command('run 0')
        out.append((r, lmp.extract_fix('tb', 0, 0), lmp.extract_fix('tb', 0, 1, 0),
                    lmp.extract_fix('tb', 0, 1, 1)))
    lmp.close()
    out = np.array(out)
    spread = np.ptp(out[:, 1])
    print(f'flat landscape (n=100, lambda=1): E_total in [{out[:, 1].min():.12f}, '
          f'{out[:, 1].max():.12f}] while E2 ranges over [{out[:, 2].min():.4f}, '
          f'{out[:, 2].max():.4f}]')
    ok = spread < 1e-10
    print('PASS' if ok else 'FAIL', 'flat landscape')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

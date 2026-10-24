#!/usr/bin/env python3
"""Tests for fix bond/flux (BondFlux) in 3body mode (Sciortino three-body swap potential).

  1. forces = -gradient of the total potential energy (central finite differences)
  2. virial = -strain derivative of the fix energy (box scaling and shear)
  3. flat energy landscape for lambda = 1: a second partner approaching a bonded
     pair does not change the energy (Sciortino, Eur. Phys. J. E 40, 3 (2017))
  4. forces identical on any number of MPI ranks (written to forces_npN.npy and
     compared when both 1- and N-rank files exist)
  5. per-type-pair parameters (--coeff): stickers C (type 4) bond to A with their own
     epsilon, sigma, n, rc and lambda; checks 1, 2 and 4 as above, and a coeff equal
     to the defaults changes nothing
  6. mixing rule (--flat): with lambda = 1, a weaker partner approaching a stronger
     bond leaves the energy flat, and a stronger partner approaching a weaker bond
     lowers it monotonically, so the swap has no barrier

Run:  python3 test_3body.py ;  mpirun -np 4 python3 test_3body.py ;  python3 test_3body.py --flat
      python3 test_3body.py --coeff ;  mpirun -np 4 python3 test_3body.py --coeff
"""

import argparse
import os
import sys

import numpy as np
from lammps import LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, lammps

RC3 = 1.3
# C stickers (type 4) bond to A (type 1) with their own parameters: eps sigma n rc lambda
COEFF_AC = (1.6, 0.95, 18.0, 1.25, 1.2)


def setup(lmp, grid, box, npow, lam, pos, types, extra=''):
    """A (1) and B (2) stickers, inert beads (3), and C stickers (4) that bond to A;
    extra holds additional fix keywords, e.g. 'bondable 1 4 coeff 1 4 ...'"""
    lmp.commands_string(f"""
units lj
atom_style bond
boundary p p p
processors {grid[0]} {grid[1]} {grid[2]}
region box block 0 {box[0]} 0 {box[1]} 0 {box[2]}
create_box 4 box bond/types 1 extra/bond/per/atom 2 extra/special/per/atom 4
mass * 1.0
pair_style lj/cut {2 ** (1 / 6)}
pair_coeff * * 1.0 1.0
pair_coeff 1 2 0.0 1.0
pair_coeff 1 4 0.0 1.0
pair_modify shift yes
bond_style harmonic
bond_coeff 1 100.0 1.0
""")
    for p, t in zip(pos, types):
        lmp.command(f'create_atoms {t} single {p[0]} {p[1]} {p[2]} units box')
    lmp.commands_string(f"""
group stick type 1 2 4
comm_modify cutoff 2.5
neighbor 0.3 bin
fix tb stick bond/flux 3body 1.0 1.0 {npow} {RC3} {lam} bondable 1 2 {extra}
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
    ap.add_argument('--flat', action='store_true', help='run only the flat-landscape checks')
    ap.add_argument('--coeff', action='store_true',
                    help='add C stickers (type 4) with their own parameters for A-C pairs')
    args = ap.parse_args()
    if args.flat:
        return flat_landscape()

    lmp = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    me = lmp.extract_setting('world_rank')
    nprocs = lmp.extract_setting('world_size')
    grid = {1: (1, 1, 1), 2: (2, 1, 1), 4: (2, 2, 1)}[nprocs]
    box = [12.0 * g if g > 1 else 12.0 for g in grid]
    center = np.array([b / 2.0 for b in box]) + 0.017

    # random cluster of A (1) and B (2) stickers plus inert beads (3), minimum distance 0.97;
    #   with --coeff some B are replaced by C stickers (4) with their own A-C parameters
    rng = np.random.default_rng(7)
    seq = ([1, 2, 1, 4, 1, 2, 1, 4, 1, 2, 3, 4, 3, 1] if args.coeff else
           [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 3, 3, 1])
    pos, types = [], []
    while len(pos) < 14:
        p = rng.uniform(-1.6, 1.6, 3)
        if all(np.linalg.norm(p - q) > 0.97 for q in pos):
            pos.append(p)
            types.append(seq[len(pos) - 1])
    pos = np.array(pos) + center
    natoms = len(pos)
    extra = 'bondable 1 4 coeff 1 4 ' + ' '.join(f'{v:g}' for v in COEFF_AC) if args.coeff else ''
    tag = '_coeff' if args.coeff else ''

    # A centers with both a B and a C within their cutoffs exercise the mixing rule
    mixed = 0
    if args.coeff:
        for i in np.where(np.array(types) == 1)[0]:
            d = np.linalg.norm(pos - pos[i], axis=1)
            near_b = np.any((np.array(types) == 2) & (d < RC3))
            near_c = np.any((np.array(types) == 4) & (d < COEFF_AC[3]))
            mixed += int(near_b and near_c)

    # a coeff equal to the defaults must not change anything
    same = True
    if args.coeff:
        ref = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
        setup(ref, grid, box, args.npow, args.lam, pos, types, extra)
        f_ref = np.ctypeslib.as_array(ref.gather_atoms('f', 1, 3)).reshape(natoms, 3).copy()
        e_ref = ref.extract_fix('tb', 0, 0)
        ref.close()
        dup = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
        setup(dup, grid, box, args.npow, args.lam, pos, types,
              extra + f' coeff 1 2 1.0 1.0 {args.npow:g} {RC3} {args.lam:g}')
        f_dup = np.ctypeslib.as_array(dup.gather_atoms('f', 1, 3)).reshape(natoms, 3).copy()
        same = (dup.extract_fix('tb', 0, 0) == e_ref) and np.array_equal(f_dup, f_ref)
        dup.close()

    setup(lmp, grid, box, args.npow, args.lam, pos, types, extra)
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

    out = os.path.join(args.outdir, f'forces_n{args.npow:g}_l{args.lam:g}{tag}_np{nprocs}.npy')
    np.save(out, f)
    mpi_msg = ''
    ref = os.path.join(args.outdir, f'forces_n{args.npow:g}_l{args.lam:g}{tag}_np1.npy')
    mpierr = 0.0
    if nprocs > 1 and os.path.exists(ref):
        mpierr = np.max(np.abs(np.load(ref) - f))
        mpi_msg = f', max |F(np{nprocs}) - F(np1)| = {mpierr:.2e}'

    label = f'3body n={args.npow:g} lambda={args.lam:g}' + (' coeff A-C' if args.coeff else '')
    print(f'{label} np={nprocs}: E_fix={efix0:.6f} '
          f'(E2={stats[0]:.4f} E3={stats[1]:.4f}), bonded pairs={stats[2]:.0f}, '
          f'multiply bonded={stats[3]:.0f}')
    if args.coeff:
        print(f'   A centers with both B and C partners: {mixed}; '
              f'coeff equal to the defaults changes nothing: {same}')
    print(f'   force check: max |F_fd - F| / max|F| = {ferr:.2e}')
    print(f'   virial check: W_fix = {np.array2string(wfix, precision=6)}')
    print(f'                 W_fd  = {np.array2string(wfd, precision=6)}  rel.err {werr:.2e}'
          f'{mpi_msg}')
    ok = (ferr < 1e-5) and (werr < 1e-5) and (mpierr < 1e-10) and same
    if args.coeff:
        ok = ok and (mixed > 0)
    print('PASS' if ok else 'FAIL', f'{label} np={nprocs}')
    return 0 if ok else 1


def approach(bonded, incoming, extra=''):
    """A bonded to a partner of type `bonded` at its potential minimum; a partner of type
    `incoming` approaches A from the opposite side (n = 100, lambda = 1). Returns rows of
    (distance, total energy, two-body energy), from far to near."""
    npow = 100.0
    rmin = {2: 2 ** (1 / npow), 4: 2 ** (1 / npow)}
    rcut = {2: RC3, 4: RC3}
    lmp = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    pos = [np.array([6.0, 6.0, 6.0]), np.array([6.0 + rmin[bonded], 6.0, 6.0]),
           np.array([4.0, 6.0, 6.0])]
    setup(lmp, (1, 1, 1), [12.0, 12.0, 12.0], npow, 1.0, pos, [1, bonded, incoming], extra)
    out = []
    for r in np.linspace(rcut[incoming] + 0.05, rmin[incoming] + 1e-3, 25):
        lmp.command(f'set atom 3 x {6.0 - r}')
        lmp.command('run 0')
        out.append((r, lmp.extract_fix('tb', 0, 0), lmp.extract_fix('tb', 0, 1, 0)))
    lmp.close()
    return np.array(out)


def flat_landscape():
    """Sciortino's flat landscape for lambda = 1, and the mixing rule for unequal bonds."""
    ok = True

    # identical pairs: A-B bonded, a second B approaches
    out = approach(2, 2)
    spread = np.ptp(out[:, 1])
    print(f'flat landscape (n=100, lambda=1): E_total in [{out[:, 1].min():.12f}, '
          f'{out[:, 1].max():.12f}] while E2 ranges over [{out[:, 2].min():.4f}, '
          f'{out[:, 2].max():.4f}]')
    ok &= spread < 1e-10

    # unequal pairs: A-B (eps 1.0) and A-C (eps 0.6), same shape, lambda = 1
    extra = 'bondable 1 4 coeff 1 4 0.6 1.0 100 1.3 1.0'
    weak_in = approach(2, 4, extra)      # C approaches the stronger A-B bond
    strong_in = approach(4, 2, extra)    # B approaches the weaker A-C bond
    spread = np.ptp(weak_in[:, 1])
    rise = np.max(np.diff(strong_in[:, 1]))
    # with the three-body strength min(eps_AB, eps_AC) = 0.6:
    #   E = -0.6 + V_AB + 0.6 V3_AB = -0.6 - 0.4 V3_AB, and V3_AB = -(E2 + 0.6)
    expected = -0.6 + 0.4 * (strong_in[:, 2] + 0.6)
    dev = np.max(np.abs(strong_in[:, 1] - expected))
    print(f'mixed pairs, C (eps 0.6) approaching A-B (eps 1.0): E_total in '
          f'[{weak_in[:, 1].min():.12f}, {weak_in[:, 1].max():.12f}]')
    print(f'mixed pairs, B (eps 1.0) approaching A-C (eps 0.6): E_total from '
          f'{strong_in[0, 1]:.6f} to {strong_in[-1, 1]:.6f}, largest rise {max(rise, 0.0):.1e}, '
          f'max |E - E_expected| {dev:.1e}')
    ok &= (spread < 1e-10) and (rise < 1e-12) and (dev < 1e-10)

    print('PASS' if ok else 'FAIL', 'flat landscape')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

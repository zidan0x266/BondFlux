#!/usr/bin/env python3
"""Exact detailed-balance test for fix bond/flux (BondFlux) in mc mode.

A small cluster of sticker atoms is frozen in place (no time integration), so the
Monte Carlo moves are the only thing that changes.  The visited bond topologies
must then follow the Boltzmann distribution exp(-E/kT) over all topologies that
the moves can reach from the initial one.  The script

  1. enumerates the reachable topologies and their energies independently of
     LAMMPS (same move rules, energies from the analytic potentials),
  2. runs LAMMPS with fix bond/flux every step and records the topology,
  3. compares visit frequencies with the exact probabilities (block z-scores),
  4. checks that the LAMMPS potential energy changes by exactly the sum of the
     energy changes of the accepted moves (fix vector element 9).

Run on 1, 2 or 4 MPI ranks (the cluster sits where the processor sub-domains meet):
    python3 test_mc_balance.py --variant kg
    mpirun -np 4 python3 test_mc_balance.py --variant quartic
    python3 test_mc_balance.py --variant kg --form 0.5 --break 0.5 --ebond -20
"""

import argparse
import itertools
import math
import os
import sys

import numpy as np
from lammps import lammps

RC = 1.3        # fix bond/flux cutoff
KT = 4.0        # Metropolis temperature (high enough that several topologies matter)
RCUT_WCA = 2.0 ** (1.0 / 6.0)

# permanent bond = type 1 between atoms 1 and 2, dynamic bonds = type 2
VARIANTS = {
    # Kremer-Grest: fene with WCA inside the bond, special 0 1 1
    'kg': dict(special=(0.0, 1.0, 1.0), bond_style='fene',
               coeff1='1 30.0 1.5 1.0 1.0', coeff2='2 30.0 1.5 1.0 1.0',
               dyn=('fene', 30.0, 1.5, 1.0)),
    # breakable quartic bonds (special 1 1 1); LAMMPS cannot hybridize quartic with
    #   unbreakable styles, so the dynamic bond type is quartic as well
    'quartic': dict(special=(1.0, 1.0, 1.0), bond_style='quartic',
                    coeff1='1 1200.0 -0.55 0.25 1.3 34.6878',
                    coeff2='2 1200.0 -0.55 0.25 1.5 34.6878',
                    dyn=('quartic', 1200.0, -0.55, 0.25, 1.5, 34.6878)),
    # pure FENE bond with 1-2 pairs excluded: bonded pairs lose their WCA repulsion
    'fene0': dict(special=(0.0, 1.0, 1.0), bond_style='fene',
                  coeff1='1 30.0 1.5 1.0 1.0', coeff2='2 30.0 1.5 0.0 1.0',
                  dyn=('fene', 30.0, 1.5, 0.0)),
    # harmonic bond with a fractional 1-2 weight
    'harm05': dict(special=(0.5, 1.0, 1.0), bond_style='harmonic',
                   coeff1='1 100.0 1.0', coeff2='2 20.0 1.0', dyn=('harmonic', 20.0, 1.0)),
}

# positions of the six stickers relative to the cluster center
CLUSTER = np.array([
    [0.00, 0.00, 0.00],
    [1.00, 0.10, 0.00],
    [0.45, 0.95, 0.30],
    [-0.60, 0.70, -0.50],
    [0.50, -0.30, 1.00],
    [1.30, 0.90, 0.80],
])
PERMANENT = (1, 2)                       # atom IDs, permanently bonded
INITIAL = [(1, 4), (3, 6)]               # initial dynamic bonds


def wca(r):
    return 4.0 * (r ** -12 - r ** -6) + 1.0 if r < RCUT_WCA else 0.0


def dyn_bond_energy(r, dyn):
    if dyn[0] == 'fene':
        k, r0, eps = dyn[1], dyn[2], dyn[3]
        e = -0.5 * k * r0 * r0 * math.log(1.0 - (r / r0) ** 2)
        if r < RCUT_WCA:
            e += eps * wca(r)
        return e
    if dyn[0] == 'quartic':
        # bond_style quartic: its WCA term cancels the pair term it subtracts
        k, b1, b2, rq, u0 = dyn[1:]
        dr = r - rq
        return k * dr * dr * (dr - b1) * (dr - b2) + u0
    k, r0 = dyn[1], dyn[2]
    return k * (r - r0) ** 2


def reachable_states(pos, shift=True, swap=True, formbreak=False):
    """All topologies reachable from INITIAL with the move rules of fix bond/flux."""
    n = len(pos)
    dist = {(a, b): float(np.linalg.norm(pos[a - 1] - pos[b - 1]))
            for a in range(1, n + 1) for b in range(1, n + 1) if a != b}

    def ok(a, b):
        return (a, b) != PERMANENT and (b, a) != PERMANENT and dist[(a, b)] < RC

    def moves(state):
        partner = {}
        for a, b in state:
            partner[a], partner[b] = b, a
        out = set()
        for i, k in partner.items():
            if dist[(i, k)] >= RC:
                continue
            for j in range(1, n + 1):
                if j in (i, k) or not ok(i, j):
                    continue
                if j not in partner:
                    if shift:
                        new = set(state) - {tuple(sorted((i, k)))}
                        new.add(tuple(sorted((i, j))))
                        out.add(frozenset(new))
                elif swap:
                    l = partner[j]
                    if l in (i, k) or dist[(j, l)] >= RC or not ok(k, l):
                        continue
                    new = set(state) - {tuple(sorted((i, k))), tuple(sorted((j, l)))}
                    new |= {tuple(sorted((i, j))), tuple(sorted((k, l)))}
                    out.add(frozenset(new))
        if formbreak:
            for a, b in state:
                if dist[(a, b)] < RC:
                    out.add(frozenset(set(state) - {(a, b)}))
            for a, b in itertools.combinations(range(1, n + 1), 2):
                if a not in partner and b not in partner and ok(a, b):
                    out.add(frozenset(set(state) | {(a, b)}))
        return out

    start = frozenset(tuple(sorted(b)) for b in INITIAL)
    seen, todo = {start}, [start]
    while todo:
        s = todo.pop()
        for t in moves(s):
            if t not in seen:
                seen.add(t)
                todo.append(t)
    return sorted(seen, key=sorted), dist


def state_energy(state, dist, var, ebond=0.0):
    w12 = var['special'][0]
    n = len(CLUSTER)
    bonded = set(state) | {PERMANENT}
    e = 0.0
    for a, b in itertools.combinations(range(1, n + 1), 2):
        w = w12 if (a, b) in bonded else 1.0
        e += w * wca(dist[(a, b)])
    for a, b in state:
        e += dyn_bond_energy(dist[(a, b)], var['dyn']) + ebond
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='kg', choices=sorted(VARIANTS))
    ap.add_argument('--steps', type=int, default=200000)
    ap.add_argument('--seed', type=int, default=4357)
    ap.add_argument('--shift', default='yes')
    ap.add_argument('--swap', default='yes')
    ap.add_argument('--barrier', type=float, default=0.0)
    ap.add_argument('--form', type=float, default=0.0)
    ap.add_argument('--break', dest='pbreak', type=float, default=0.0)
    ap.add_argument('--ebond', type=float, default=0.0)
    ap.add_argument('--outdir', default='.')
    ap.add_argument('--save', default='',
                    help='write exact and MC probabilities to this HDF5 (.h5) file')
    ap.add_argument('--corner', action='store_true',
                    help='place the cluster across a periodic box corner')
    args = ap.parse_args()
    var = VARIANTS[args.variant]

    lmp = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    me = lmp.extract_setting('world_rank')
    nprocs = lmp.extract_setting('world_size')
    grid = {1: (1, 1, 1), 2: (2, 1, 1), 4: (2, 2, 1)}[nprocs]
    box = [12.0 * g for g in grid]
    center = np.array([b / 2.0 for b in box]) + 0.013
    if args.corner:
        center = np.array([0.013, 0.013, 0.013])
    pos = (CLUSTER + center) % np.array(box)

    tag = (f'{args.variant}_np{nprocs}_shift{args.shift}_swap{args.swap}_b{args.barrier:g}'
           f'_form{args.form:g}_break{args.pbreak:g}_e{args.ebond:g}'
           f'{"_corner" if args.corner else ""}')
    dumpfile = os.path.join(args.outdir, f'mc_{tag}.dump')
    efile = os.path.join(args.outdir, f'mc_{tag}.energy')

    cmds = f"""
units lj
atom_style bond
boundary p p p
processors {grid[0]} {grid[1]} {grid[2]}
region box block 0 {box[0]} 0 {box[1]} 0 {box[2]}
create_box 1 box bond/types 2 extra/bond/per/atom 4 extra/special/per/atom 8
mass 1 1.0
pair_style lj/cut {RCUT_WCA}
pair_coeff 1 1 1.0 1.0
pair_modify shift yes
special_bonds lj/coul {var['special'][0]} {var['special'][1]} {var['special'][2]}
bond_style {var['bond_style']}
bond_coeff {var['coeff1']}
bond_coeff {var['coeff2']}
"""
    lmp.commands_string(cmds)
    for p in pos:
        lmp.command(f'create_atoms 1 single {p[0]} {p[1]} {p[2]} units box')
    lmp.command(f'create_bonds single/bond 1 {PERMANENT[0]} {PERMANENT[1]}')
    for a, b in INITIAL:
        lmp.command(f'create_bonds single/bond 2 {a} {b}')
    lmp.commands_string(f"""
group stick type 1
comm_modify cutoff 3.0
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes
fix mc stick bond/flux mc 1 2 {RC} {KT} {args.seed} shift {args.shift} swap {args.swap} barrier {args.barrier} form {args.form} break {args.pbreak} ebond {args.ebond}
thermo_modify norm no
variable e equal pe
fix ave all ave/time 1 1 1 v_e f_mc[9] file {efile} format ' %.15g'
dump d all custom 1 {dumpfile} id f_mc[1]
dump_modify d sort id
thermo 10000
run {args.steps}
""")
    stats = [lmp.extract_fix('mc', 0, 1, i) for i in range(11)]
    lmp.finalize()
    if me != 0:
        return 0

    states, dist = reachable_states(CLUSTER, args.shift == 'yes', args.swap == 'yes',
                                    args.form > 0.0)
    energies = np.array([state_energy(s, dist, var, args.ebond) for s in states])
    p_exact = np.exp(-(energies - energies.min()) / KT)
    p_exact /= p_exact.sum()
    index = {s: i for i, s in enumerate(states)}

    # visited topologies, one per step (first frame = initial state before any move)
    natoms = len(CLUSTER)
    visits = []
    with open(dumpfile) as f:
        lines = f.read().split('ITEM: ATOMS id f_mc[1]\n')[1:]
    for block in lines:
        rows = [ln.split() for ln in block.strip().split('\n')[:natoms]]
        partner = {int(r[0]): int(float(r[1])) for r in rows}
        s = frozenset(tuple(sorted((a, b))) for a, b in partner.items() if b and a < b)
        if s not in index:
            print(f'FAIL {tag}: visited topology {sorted(s)} is not reachable/allowed')
            return 1
        visits.append(index[s])
    visits = np.array(visits[1:])

    nblock = 20
    blocks = np.array_split(visits, nblock)
    freq = np.array([np.bincount(b, minlength=len(states)) / len(b) for b in blocks])
    mean, err = freq.mean(0), freq.std(0, ddof=1) / math.sqrt(nblock)
    z = np.where(err > 0, (mean - p_exact) / np.maximum(err, 1e-12), 0.0)
    tv = 0.5 * np.abs(mean - p_exact).sum()

    # energy bookkeeping: pe(t) - pe(0) == sum of accepted energy changes
    edata = np.loadtxt(efile)
    drift = np.max(np.abs((edata[:, 1] - edata[0, 1]) - (edata[:, 2] - edata[0, 2])))
    # LAMMPS energy of each visited topology minus the analytic energy is a constant
    steps = edata[:, 0].astype(int)
    offs = edata[1:, 1] - energies[visits[steps[1:] - 1]] if len(steps) > 1 else np.zeros(1)
    offspread = float(np.ptp(offs))

    print(f'{tag}: {len(states)} reachable topologies, {len(visits)} samples, '
          f'accepted shifts {stats[2]:.0f} swaps {stats[3]:.0f} formations {stats[4]:.0f} '
          f'breakings {stats[5]:.0f} of {stats[1]:.0f} proposals')
    for i, s in enumerate(states):
        print(f'   {str(sorted(s)):24s} E={energies[i]:9.4f}  exact={p_exact[i]:.4f}  '
              f'mc={mean[i]:.4f} +- {err[i]:.4f}  z={z[i]:+.2f}')
    print(f'   total variation distance {tv:.4f}, max |z| {np.max(np.abs(z)):.2f}, '
          f'pe vs sum(dE) mismatch {drift:.2e}, pe - E_exact spread {offspread:.2e}')

    ok = (np.max(np.abs(z)) < 4.5) and (tv < 0.01) and (drift < 1e-6) and (offspread < 1e-6)
    if args.save:
        import h5py
        with h5py.File(args.save, 'w') as h5:
            h5['topology'] = np.array([str(sorted(x)) for x in states], dtype='S')
            h5['nbonds'] = np.array([len(x) for x in states])
            h5['energy'] = energies
            h5['p_exact'] = p_exact
            h5['p_mc'] = mean
            h5['p_mc_err'] = err
            h5.attrs['test'] = tag
            h5.attrs['total_variation'] = tv
            h5.attrs['samples'] = len(visits)
    print('PASS' if ok else 'FAIL', tag)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

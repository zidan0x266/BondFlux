#!/usr/bin/env python3
"""Cross-check fix bond/flux 3body against HOOMD-blue's RevCross three-body potential.

HOOMD-blue (hoomd.md.many_body.RevCross) implements the Sciortino swap potential
with the unshifted 2n-n Lennard-Jones bond, so fix bond/flux is run with
"eshift no".  For random clusters of A and B stickers (A-B bondable only) the
script compares the force on every particle, the virial tensor and the energy.

Finding (HOOMD-blue 7.2.0): RevCross accumulates only the two-body energy; the
three-body energy lambda*eps*V3*V3 is never added to the particle energies
(PotentialTersoff.h, RevCross branch), although its forces and virial are.  The
HOOMD energy is therefore compared with the two-body energy E2 of fix bond/flux,
and a finite-difference check of the HOOMD energy shows the inconsistency.

Needs the hoomd and lammps Python modules in the same Python:
    PYTHONPATH=<hoomd build dir>:<lammps python dir> python3 test_hoomd_revcross.py
"""

import sys

import hoomd
import hoomd.md
import numpy as np
from lammps import LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, lammps

L = 12.0


def cluster(seed, natoms, dmin, extent):
    rng = np.random.default_rng(seed)
    pos = []
    while len(pos) < natoms:
        p = rng.uniform(-extent, extent, 3)
        if all(np.linalg.norm(p - q) > dmin for q in pos):
            pos.append(p)
    types = np.array([i % 2 for i in range(natoms)])        # 0 = A, 1 = B
    return np.array(pos), types


def run_hoomd(pos, types, npow, rc, lam, eps):
    sim = hoomd.Simulation(device=hoomd.device.CPU(notice_level=0), seed=1)
    snap = hoomd.Snapshot()
    snap.configuration.box = [L, L, L, 0, 0, 0]
    snap.particles.N = len(pos)
    snap.particles.types = ['A', 'B']
    snap.particles.position[:] = pos
    snap.particles.typeid[:] = types
    snap.particles.mass[:] = 1.0
    sim.create_state_from_snapshot(snap)
    nl = hoomd.md.nlist.Cell(buffer=0.3)
    rev = hoomd.md.many_body.RevCross(nlist=nl, default_r_cut=rc)
    off = dict(sigma=0.0, n=0.0, epsilon=0.0, lambda3=0.0)
    rev.params[('A', 'A')] = off
    rev.params[('B', 'B')] = off
    rev.params[('A', 'B')] = dict(sigma=1.0, n=npow, epsilon=eps, lambda3=lam)
    integrator = hoomd.md.Integrator(dt=0.0, forces=[rev])
    sim.operations.integrator = integrator
    sim.always_compute_pressure = True
    sim.run(0)
    force = np.array(rev.forces)
    virial = np.array(rev.virials).sum(axis=0)     # xx xy xz yy yz zz
    return rev.energy, force, virial


def hoomd_fd_force(pos, types, npow, rc, lam, eps, atom, h=1e-6):
    """-dE/dx of the HOOMD energy for one atom, by central differences."""
    out = np.zeros(3)
    for d in range(3):
        e = []
        for s in (1.0, -1.0):
            p = pos.copy()
            p[atom, d] += s * h
            e.append(run_hoomd(p, types, npow, rc, lam, eps)[0])
        out[d] = -(e[0] - e[1]) / (2.0 * h)
    return out


def run_lammps(pos, types, npow, rc, lam, eps):
    lmp = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    lmp.commands_string(f"""
units lj
atom_style bond
boundary p p p
region box block {-L / 2} {L / 2} {-L / 2} {L / 2} {-L / 2} {L / 2}
create_box 2 box bond/types 1
mass * 1.0
pair_style zero {rc}
pair_coeff * *
""")
    for p, t in zip(pos, types):
        lmp.command(f'create_atoms {t + 1} single {p[0]} {p[1]} {p[2]} units box')
    lmp.commands_string(f"""
comm_modify cutoff {rc + 1.0}
neighbor 0.3 bin
fix tb all bond/flux 3body {eps} 1.0 {npow} {rc} {lam} bondable 1 2 eshift no
compute pfix all pressure NULL fix
thermo_style custom step pe c_pfix[1]
thermo_modify norm no
run 0
""")
    n = len(pos)
    force = np.ctypeslib.as_array(lmp.gather_atoms('f', 1, 3)).reshape(n, 3).copy()
    energy = lmp.extract_fix('tb', 0, 1, 0)      # two-body energy E2
    e3 = lmp.extract_fix('tb', 0, 1, 1)          # three-body energy E3
    p = lmp.numpy.extract_compute('pfix', LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR)
    vol = L ** 3
    # LAMMPS order xx yy zz xy xz yz -> HOOMD order xx xy xz yy yz zz
    virial = np.array([p[0], p[3], p[4], p[1], p[5], p[2]]) * vol
    lmp.close()
    return energy, e3, force, virial


def main():
    cases = [(12.0, 1.3, 1.0), (12.0, 1.3, 0.0), (12.0, 1.5, 1.3), (50.0, 1.3, 1.0),
             (100.0, 1.2, 1.0), (100.0, 1.3, 1.1)]
    ok = True
    for npow, rc, lam in cases:
        for seed in (1, 2, 3):
            dmin = 0.95 if npow < 60 else 0.99
            pos, types = cluster(seed, 40, dmin, 2.2)
            eh, fh, vh = run_hoomd(pos, types, npow, rc, lam, 30.0 if npow == 100.0 else 1.0)
            el, e3, fl, vl = run_lammps(pos, types, npow, rc, lam, 30.0 if npow == 100.0 else 1.0)
            fscale = max(1.0, np.abs(fh).max())
            de = abs(eh - el) / max(1.0, abs(eh))
            df = np.abs(fh - fl).max() / fscale
            dv = np.abs(vh - vl).max() / max(1.0, np.abs(vh).max())
            good = (de < 1e-9) and (df < 1e-9) and (dv < 1e-9)
            ok &= good
            print(f'n={npow:5g} rc={rc} lambda={lam}: E_hoomd={eh:14.8f} E2_lammps={el:14.8f} '
                  f'(E3_lammps={e3:12.6f}, not in HOOMD energy)  rel.diff E2 {de:.1e} '
                  f'F {df:.1e} W {dv:.1e}  {"ok" if good else "MISMATCH"}')

    # HOOMD energy vs HOOMD force for a particle with a three-body interaction
    pos, types = cluster(1, 40, 0.95, 2.2)
    _, _, fl, _ = run_lammps(pos, types, 12.0, 1.3, 1.0, 1.0)
    _, _, fl0, _ = run_lammps(pos, types, 12.0, 1.3, 0.0, 1.0)
    atom = int(np.argmax(np.abs(fl - fl0).sum(1)))
    _, fh, _ = run_hoomd(pos, types, 12.0, 1.3, 1.0, 1.0)
    fd = hoomd_fd_force(pos, types, 12.0, 1.3, 1.0, 1.0, atom)
    print(f'HOOMD self-consistency (n=12, lambda=1, atom {atom}): force {np.round(fh[atom], 6)}, '
          f'-dE_hoomd/dx {np.round(fd, 6)}, three-body part of force '
          f'{np.round(fl[atom] - fl0[atom], 6)}')
    print('PASS hoomd revcross comparison (forces, virial, two-body energy)' if ok else
          'FAIL hoomd revcross comparison')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

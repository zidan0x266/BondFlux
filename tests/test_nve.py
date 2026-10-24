#!/usr/bin/env python3
"""Energy of MD with Monte Carlo bond moves of fix bond/flux (BondFlux, mc mode) in NVE.

An accepted move changes the potential energy by exactly the Delta E of its Metropolis test
(f_ID[9] sums them), so in NVE  E_total - sum(Delta E)  should stay constant.  The MC step
falls between the two half-kicks of velocity Verlet; without a correction (impulse no) the
second half-kick uses the forces of the new bonds, which heats the system at every accepted
move.  With the velocity correction (impulse yes, the default) the drift disappears.

System: the gas of A-B dimers and free A stickers of test_contact.py, exchanging bonds by
shifts (barrier 0), first thermalized with a Langevin thermostat, then run in NVE.

With a finite time step, E_total also differs from the energy that velocity Verlet conserves by
an O(dt^2) term that depends on the forces; it changes at every move, so with the correction
E_total - sum(Delta E) still wanders a little, without a trend.

Checks: without moves the energy is conserved; with impulse no, E_total - sum(Delta E) rises;
with impulse yes its drift is below 25 % of that rise.

Run:  python3 test_nve.py ;  mpirun -np 4 python3 test_nve.py
"""

import argparse
import os
import sys

from lammps import lammps

from test_contact import write_data

DT = 0.01
NVE_STEPS = 12000


def run(case, outdir, me):
    """case: 'none' (moves blocked by a large barrier), 'yes', 'no' (impulse)"""
    data = os.path.join(outdir, 'nve.data')
    if me == 0:
        write_data(data)
    barrier = 60 if case == 'none' else 0
    impulse = 'no' if case == 'no' else 'yes'
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
fix fx all bond/flux mc 10 1 1.3 1.0 2718 bondable 1 2 swap no barrier {barrier} impulse {impulse}
run 2000
unfix lang
variable cons equal etotal-f_fx[9]
""")
    trace = []
    for _ in range(NVE_STEPS // 500):
        lmp.command('run 500')
        trace.append(lmp.extract_variable('cons'))
    moves = lmp.extract_fix('fx', 0, 1, 2) + lmp.extract_fix('fx', 0, 1, 3)
    lmp.close()
    start = trace[0]
    drift = trace[-1] - start
    spread = max(abs(c - start) for c in trace)
    return drift, spread, moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='.')
    args = ap.parse_args()
    # this instance stays open, so that all ranks finalize MPI together at the end
    probe = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    me, nprocs = probe.extract_setting('world_rank'), probe.extract_setting('world_size')
    res = {case: run(case, args.outdir, me) for case in ('none', 'no', 'yes')}
    ok = True
    if me == 0:
        for case, (drift, spread, moves) in res.items():
            label = {'none': 'no moves      ', 'no': 'impulse no    ', 'yes': 'impulse yes   '}[case]
            per = drift / moves if moves else 0.0
            print(f'{label}: E_total - sum(dE) drift {drift:9.3f} (max deviation {spread:.3f}), '
                  f'{moves:.0f} accepted moves, {per:.5f} per move')
        _, s_none, _ = res['none']
        d_no, _, m_no = res['no']
        d_yes, _, m_yes = res['yes']
        ok = (m_yes > 0 and m_no > 0 and s_none < 0.1 and d_no > 0.0
              and abs(d_yes) < 0.25 * d_no)
        print('PASS' if ok else 'FAIL', f'energy with MC moves in NVE np={nprocs}')
    probe.finalize()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Exact stationary distributions of bond-exchange Monte Carlo rules on a frozen cluster.

For the six-sticker cluster of test_mc_balance.py, the transition matrix of a
random-scan exchange Monte Carlo (pick a bonded sticker, pick a candidate
partner uniformly, accept or reject) is built exactly for several acceptance
rules, and its stationary distribution is compared with the Boltzmann
distribution exp(-E/kT) of the force field.

rules
  bondflux   fix bond/flux: every broken and formed bond inside the window, barrier as a
             symmetric gate: acc = exp(-Ub/kT) min(1, exp(-dE/kT))
  ub_inside  barrier added to the energy change (Xia and Olvera de la Cruz, Polymer 308,
             127371 (2024); Macromolecules 57, 8793 (2024)): acc = min(1, exp(-(dE+Ub)/kT))

No sampling is involved: the numbers are the exact long-time limits of each rule.

A reaction window that is checked only for the new bond (Xia et al., Macromolecules
56, 8080 (2023)) cannot be probed here: its bias acts through bond-length
fluctuations, which are absent in a frozen cluster.
"""

import itertools
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
import test_mc_balance as T    # noqa: E402  (cluster, energies)


def build(states, dist, energy, rlo, rhi, rule, ub, kt, shift=True, swap=True):
    index = {s: i for i, s in enumerate(states)}
    n = len(T.CLUSTER)
    nstate = len(states)
    m = np.zeros((nstate, nstate))
    gate = np.exp(-ub / kt)

    def inwin(a, b):
        return rlo <= dist[(a, b)] < rhi

    def allowed(a, b):
        return (a, b) != T.PERMANENT and (b, a) != T.PERMANENT

    for s in states:
        i0 = index[s]
        partner = {}
        for a, b in s:
            partner[a], partner[b] = b, a
        for c in range(1, n + 1):               # random scan over all stickers
            pc = 1.0 / n
            if c not in partner:
                continue
            k = partner[c]
            if rule != 'new_only' and not inwin(c, k):
                continue
            cands = [j for j in range(1, n + 1) if j not in (c, k) and allowed(c, j) and inwin(c, j)
                     and ((j not in partner and shift) or (j in partner and swap))]
            for j in cands:
                if j not in partner:
                    new = (set(s) - {tuple(sorted((c, k)))}) | {tuple(sorted((c, j)))}
                else:
                    l = partner[j]
                    if l in (c, k) or not allowed(k, l):
                        continue
                    if rule != 'new_only' and not inwin(j, l):
                        continue
                    if not inwin(k, l):
                        continue
                    new = (set(s) - {tuple(sorted((c, k))), tuple(sorted((j, l)))}) | \
                          {tuple(sorted((c, j))), tuple(sorted((k, l)))}
                new = frozenset(new)
                if new not in index:
                    continue
                de = energy[index[new]] - energy[i0]
                if rule == 'ub_inside':
                    acc = min(1.0, np.exp(-(de + ub) / kt))
                else:
                    acc = gate * min(1.0, np.exp(-de / kt))
                m[i0, index[new]] += pc / len(cands) * acc
        m[i0, i0] = 1.0 - m[i0].sum()
    return m


def stationary(m):
    w, v = np.linalg.eig(m.T)
    p = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    return p / p.sum()


def main():
    var = T.VARIANTS['kg']
    kt = T.KT
    states_all = []
    n = len(T.CLUSTER)
    _, dist = T.reachable_states(T.CLUSTER)
    pairs = [p for p in itertools.combinations(range(1, n + 1), 2)
             if p != T.PERMANENT and dist[p] < T.RC]
    for a, b in itertools.combinations(pairs, 2):          # all two-bond topologies
        if len({*a, *b}) == 4:
            states_all.append(frozenset({a, b}))
    energy = np.array([T.state_energy(s, dist, var) for s in states_all])

    print(f'cluster of test_mc_balance.py, kg force field, kT = {kt}')
    print('rows: total variation distance between the stationary distribution of the rule '
          'and the Boltzmann distribution (0 = exact)')
    for rlo, rhi, label in [(0.0, T.RC, f'window [0, {T.RC})'), (0.87, 1.14, 'window [0.87, 1.14]')]:
        print(f'\n{label}')
        print('   Ub/kT   bondflux   ub_inside')
        for ub in (0.0, 1.0, 2.0, 4.0, 8.0):
            row = []
            for rule in ('bondflux', 'ub_inside'):
                m = build(states_all, dist, energy, rlo, rhi, rule, ub * kt, kt)
                # restrict to the states connected to the initial topology
                start = states_all.index(frozenset(tuple(sorted(b)) for b in T.INITIAL))
                reach = {start}
                todo = [start]
                while todo:
                    i = todo.pop()
                    for j in np.nonzero(m[i] > 0)[0]:
                        if j not in reach:
                            reach.add(int(j))
                            todo.append(int(j))
                sel = sorted(reach)
                mm = m[np.ix_(sel, sel)]
                mm /= mm.sum(1, keepdims=True)
                p = stationary(mm)
                b = np.exp(-(energy[sel] - energy[sel].min()) / kt)
                b /= b.sum()
                row.append(0.5 * np.abs(p - b).sum())
            print(f'   {ub:5.1f}   {row[0]:8.2e}   {row[1]:9.2e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

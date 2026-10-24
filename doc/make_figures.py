#!/usr/bin/env python3
"""Collect the data of the BondFlux validity tests and draw the figures of the manual.

usage:
  make_figures.py collect --runs RUNDIR     read raw outputs, write data/bondflux_validation.h5
  make_figures.py plot                      draw figures/*.pdf from data/bondflux_validation.h5

RUNDIR layout (outputs of the examples and tests):
  RUNDIR/mc_balance_kg_formbreak.h5            tests/test_mc_balance.py --save
  RUNDIR/dumb/partners_lambda{1.0,1.1,1.3}.dump  examples/dumbbells/in.dumbbells
  RUNDIR/swap/swap_event.txt, swap_event.dump    examples/dumbbells/in.swap_event
  RUNDIR/kg/log.associate, partners_Ea*.dump     examples/kg_network/in.associate, in.exchange
"""

import argparse
import os
import re
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data', 'bondflux_validation.h5')
FIGS = os.path.join(HERE, 'figures')
sys.path.insert(0, os.path.join(HERE, '..', 'examples', 'dumbbells'))


def bond_autocorrelation(path, dt_frame, tmax):
    from analyze_nb import read_partners
    p = read_partners(path)
    nlag = min(len(p) - 1, int(tmax / dt_frame))
    origins = range(0, len(p) - nlag, max(1, nlag // 50))
    nb = np.zeros(nlag + 1)
    for t0 in origins:
        bonded = p[t0] > 0
        nb += ((p[t0:t0 + nlag + 1] == p[t0]) & bonded).sum(1) / max(1, bonded.sum())
    return np.arange(nlag + 1) * dt_frame, nb / len(origins)


def fit_tau(t, nb, lo=0.05, hi=0.9):
    sel = (nb < hi) & (nb > lo)
    if sel.sum() < 4:
        return np.nan
    return -1.0 / np.polyfit(t[sel], np.log(nb[sel]), 1)[0]


def thermo_blocks(log):
    """columns of every thermo block of a LAMMPS log, as a list of (header, array)."""
    out, header, rows = [], None, []
    for line in open(log):
        if line.strip().startswith('Step'):
            header, rows = line.split(), []
        elif header and line.startswith('Loop time'):
            out.append((header, np.array(rows, dtype=float)))
            header = None
        elif header:
            try:
                rows.append([float(v) for v in line.split()])
            except ValueError:
                pass
    return out


def collect(runs):
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    with h5py.File(DATA, 'w') as h5:
        with h5py.File(os.path.join(runs, 'mc_balance_kg_formbreak.h5')) as src:
            g = h5.create_group('mc_balance')
            for k in src:
                g[k] = src[k][()]
            for k, v in src.attrs.items():
                g.attrs[k] = v

        g = h5.create_group('dumbbells')
        for lam in ('1.0', '1.1', '1.3'):
            t, nb = bond_autocorrelation(os.path.join(runs, 'dumb', f'partners_lambda{lam}.dump'),
                                         1.0, 1000.0)
            s = g.create_group(f'lambda_{lam}')
            s['t'], s['nb'] = t, nb
            s.attrs['tau'] = fit_tau(t, nb)

        d = np.loadtxt(os.path.join(runs, 'swap', 'swap_event.txt'))
        g = h5.create_group('swap_event')
        g['t'] = d[:, 0] * 0.0002
        g['e2'], g['e3'], g['pe'] = d[:, 1], d[:, 2], d[:, 3]

        if not os.path.exists(os.path.join(runs, 'kg', 'log.associate')):
            print('no kg_network runs found, skipping')
            print(f'wrote {DATA}')
            return
        g = h5.create_group('kg_network')
        blocks = thermo_blocks(os.path.join(runs, 'kg', 'log.associate'))
        header, a = blocks[-1]
        g['associate_step'] = a[:, header.index('Step')]
        g['associate_nbonds'] = a[:, header.index('f_flux[10]')]
        for f in sorted(os.listdir(os.path.join(runs, 'kg'))):
            m = re.match(r'partners_Ea([\d.]+)\.dump$', f)
            if not m:
                continue
            t, nb = bond_autocorrelation(os.path.join(runs, 'kg', f), 10.0, 1000.0)
            s = g.create_group(f'exchange_Ea_{m.group(1)}')
            s['t'], s['c'] = t, nb
            s.attrs['tau'] = fit_tau(t, nb, 0.1, 0.9)
            hdr, b = thermo_blocks(os.path.join(runs, 'kg', f'log.exchange{m.group(1)}'))[-1]
            s['nbonds'] = b[:, hdr.index('f_flux[10]')]
    print(f'wrote {DATA}')


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 8,
                         'figure.dpi': 150, 'savefig.bbox': 'tight'})
    colors = ['#1b6ca8', '#d1495b', '#2e933c', '#e09f3e', '#6c4f9c']
    os.makedirs(FIGS, exist_ok=True)

    moves_schematic(plt, colors)

    with h5py.File(DATA) as h5:
        # exact detailed balance
        g = h5['mc_balance']
        order = np.argsort(-g['p_exact'][()])
        x = np.arange(len(order))
        fig, ax = plt.subplots(figsize=(6.2, 2.6))
        ax.bar(x - 0.2, g['p_exact'][()][order], 0.4, color=colors[0], label='exact Boltzmann')
        ax.bar(x + 0.2, g['p_mc'][()][order], 0.4, yerr=g['p_mc_err'][()][order], color=colors[1],
               label='fix bond/flux (4 MPI ranks)', error_kw=dict(lw=0.6, capsize=1.5))
        nb = g['nbonds'][()][order]
        ax.set_xticks(x, [f'{n}' for n in nb])
        ax.set_xlabel('topology (number of dynamic bonds), sorted by probability')
        ax.set_ylabel('probability')
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 1)
        ax.legend(frameon=False)
        fig.savefig(os.path.join(FIGS, 'mc_balance.pdf'))
        plt.close(fig)

        # dumbbells
        g = h5['dumbbells']
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        for c, lam in zip(colors, ('1.0', '1.1', '1.3')):
            s = g[f'lambda_{lam}']
            tau = s.attrs['tau']
            lab = rf'$\lambda={lam}$' + (rf', $\tau={tau:.0f}$' if np.isfinite(tau) else '')
            ax.semilogy(s['t'][()], s['nb'][()], color=c, label=lab)
        t = g['lambda_1.0']['t'][()]
        ax.semilogy(t, np.exp(-t / 116.0), 'k--', lw=0.8, label=r'Sciortino: $e^{-t/116}$')
        ax.set_xlim(0, 600)
        ax.set_ylim(1e-3, 1.1)
        ax.set_xlabel(r'time $t$ [$\sqrt{m\sigma^2/\epsilon}$]')
        ax.set_ylabel(r'bond autocorrelation $n_b(t)$')
        ax.legend(frameon=False)
        fig.savefig(os.path.join(FIGS, 'dumbbells.pdf'))
        plt.close(fig)

        # swap event
        g = h5['swap_event']
        t = g['t'][()]
        e3 = g['e3'][()]
        i = np.argmax(e3 > 0.05)
        sel = (t > t[i] - 0.6) & (t < t[i] + 1.4)
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        ax.plot(t[sel], g['e2'][()][sel], color=colors[0], label='two-body')
        ax.plot(t[sel], e3[sel], color=colors[1], label='three-body')
        ax.plot(t[sel], g['pe'][()][sel], color='k', lw=1.8, ls='--', label='total potential energy')
        ax.set_xlabel(r'time $t$ [$\sqrt{m\sigma^2/\epsilon}$]')
        ax.set_ylabel(r'energy / $\epsilon$')
        ax.legend(frameon=False, loc='center right')
        fig.savefig(os.path.join(FIGS, 'swap_event.pdf'))
        plt.close(fig)

        # KG network: association and exchange
        if 'kg_network' not in h5:
            print(f'wrote figures to {FIGS} (no kg_network data)')
            return
        g = h5['kg_network']
        fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
        step = g['associate_step'][()]
        axs[0].plot((step - step[0]) * 0.01, g['associate_nbonds'][()], color=colors[0])
        axs[0].set_xlabel(r'time since association started [$\tau$]')
        axs[0].set_ylabel('dynamic bonds')
        axs[0].set_title('formation + breaking', fontsize=9)
        keys = sorted([k for k in g if k.startswith('exchange_Ea_')],
                      key=lambda k: float(k.split('_')[-1]))
        for c, k in zip(colors, keys):
            s = g[k]
            ea = float(k.split('_')[-1])
            axs[1].semilogy(s['t'][()], s['c'][()], color=c,
                            label=rf'$E_a={ea:g}\,\epsilon$, $\tau_x={s.attrs["tau"]:.0f}$')
        axs[1].set_xlabel(r'time [$\tau$]')
        axs[1].set_ylabel(r'bond autocorrelation $C(t)$')
        axs[1].set_ylim(5e-3, 1.1)
        axs[1].set_title('exchange only (bond number fixed)', fontsize=9)
        axs[1].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, 'kg_network.pdf'))
        plt.close(fig)
    print(f'wrote figures to {FIGS}')


def moves_schematic(plt, colors):
    """the four Monte Carlo moves of fix bond/flux, before -> after"""
    sticker, other, bondc = colors[1], '#9aa5b1', colors[0]
    panels = [
        ('shift', {'i': (0, 0), 'k': (1, 0), 'j': (-0.55, 0.85)}, [('i', 'k')], [('i', 'j')]),
        ('swap', {'i': (0, 0), 'k': (1, 0), 'j': (0, 1), 'l': (1, 1)}, [('i', 'k'), ('j', 'l')],
         [('i', 'j'), ('k', 'l')]),
        ('formation', {'i': (0, 0), 'j': (1, 0)}, [], [('i', 'j')]),
        ('breaking', {'i': (0, 0), 'k': (1, 0)}, [('i', 'k')], []),
    ]
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.9))
    for ax, (name, pos, before, after) in zip(axs, panels):
        for shift, bonds in ((0.0, before), (2.9, after)):
            for a, b in bonds:
                xa, ya = pos[a]
                xb, yb = pos[b]
                ax.plot([xa + shift, xb + shift], [ya, yb], color=bondc, lw=2.5, zorder=1)
            for lab, (x, y) in pos.items():
                ax.add_patch(plt.Circle((x + shift, y), 0.22, color=sticker, zorder=2))
                ax.text(x + shift, y, lab, ha='center', va='center', color='white', fontsize=8,
                        zorder=3)
        ax.annotate('', xy=(2.15, 0.5), xytext=(1.45, 0.5),
                    arrowprops=dict(arrowstyle='->', color='k', lw=1.0))
        ax.set_title(name, fontsize=9)
        ax.set_xlim(-0.9, 4.2)
        ax.set_ylim(-0.45, 1.35)
        ax.set_aspect('equal')
        ax.axis('off')
    fig.savefig(os.path.join(FIGS, 'moves.pdf'))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['collect', 'plot'])
    ap.add_argument('--runs', default='.')
    args = ap.parse_args()
    if args.action == 'collect':
        collect(args.runs)
    else:
        plot()
    return 0


if __name__ == '__main__':
    sys.exit(main())

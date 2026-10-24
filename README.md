# BondFlux

A simulation toolkit for dynamic bonds in polymer networks. BondFlux adds bond formation,
breaking, and exchange to LAMMPS molecular dynamics. It is not limited to vitrimers: it also
handles dissociative networks, supramolecular stickers, and multiple-network elastomers whose
strands break under load.

BondFlux is a single LAMMPS fix style, `fix bond/flux`, with two modes:

| mode | how bonds change | dynamic bonds are |
|---|---|---|
| `mc` | Monte Carlo moves between MD steps: exchange (shift, swap) and, optionally, reversible formation and breaking | LAMMPS bonds of one bond type (FENE, quartic, harmonic, ...) |
| `3body` | MD with the Sciortino three-body swap potential | pair interactions between sticker atoms |

The Monte Carlo moves satisfy detailed balance with respect to the full force field (bond
style, pair style and `special_bonds` weights), in serial and in parallel. A reaction barrier
changes the exchange rate without changing the equilibrium. See the
[manual](manual.pdf) for the methods and the validation.

## Repository layout

```
src/        fix_bond_flux.cpp, fix_bond_flux.h   the LAMMPS fix
examples/   kg_network/   Kremer-Grest sticker melt -> dynamic network -> exchange (mc)
            dumbbells/    Sciortino dumbbell benchmark and swap-event energies (3body)
tests/      detailed-balance and acceptance-rule tests (mc), force/virial tests (3body)
manual.pdf  user manual: methods, LAMMPS commands and validation tests
```

## Building LAMMPS with BondFlux

The fix uses only core LAMMPS classes. Put the two source files into the LAMMPS `src`
directory (as links or copies) and build LAMMPS with CMake:

```bash
git clone https://github.com/zidan0x266/BondFlux.git
git clone -b stable https://github.com/lammps/lammps.git
ln -s $PWD/BondFlux/src/fix_bond_flux.h $PWD/BondFlux/src/fix_bond_flux.cpp lammps/src/
cd lammps
cmake -S cmake -B build -D PKG_MOLECULE=on -D PKG_EXTRA-PAIR=on -D BUILD_MPI=on -D BUILD_SHARED_LIBS=on
cmake --build build -j 8
build/lmp -h | grep bond/flux
```

`PKG_MOLECULE` provides the bond styles. `PKG_EXTRA-PAIR` (`mie/cut`) is needed only for the
dumbbell examples, and `BUILD_SHARED_LIBS` only for the Python tests. With the legacy make
build, copy the two files into `lammps/src` and run `make mpi`. BondFlux is tested with
LAMMPS stable 30 Sep 2026.

## Quick start

Kremer-Grest vitrimer: stickers of type 2 exchange their bonds of type 2, and the number of
bonds is conserved:

```
special_bonds   lj/coul 0.0 1.0 1.0
bond_style      fene
bond_coeff      * 30.0 1.5 1.0 1.0
comm_modify     cutoff 3.2
group           stick type 2
fix             flux stick bond/flux mc 100 2 1.3 1.0 777 barrier 2.0
```

Dissociative network: bonds also form and break; `ebond` is the association energy per bond:

```
fix             flux stick bond/flux mc 100 2 1.3 1.0 12347 form 1.0 break 0.5 ebond -24.0
```

Sciortino three-body swap potential between A (type 1) and B (type 2) stickers; the pair
style interaction between A and B must be switched off:

```
pair_style      mie/cut $(2.0^(1.0/100))       # A-A and B-B: repulsive 2n-n core
pair_coeff      * * 1.0 1.0 200 100
pair_coeff      1 2 0.0 1.0 200 100            # A-B: provided by the fix
pair_modify     shift yes
comm_modify     cutoff 2.0
fix             tb stick bond/flux 3body 1.0 1.0 100 1.3 1.0 bondable 1 2
```

## Syntax

```
fix ID group-ID bond/flux mc    Nevery btype rc T seed keyword value ...
fix ID group-ID bond/flux 3body eps sigma n rc lambda keyword value ...
```

**mc**: every `Nevery` steps each sticker (atom of the fix group) is offered one move. A
bonded sticker breaks its bond (probability `break`) or exchanges it with a candidate within
the reaction window; a free sticker forms a bond (probability `form`). Every bond broken or
formed must have a length in `[rmin, rc)`, which makes each move reversible. `btype` is the
dynamic bond type and `T` the Metropolis temperature (use the thermostat set point).

| keyword | value | default |
|---|---|---|
| `bondable` | `itype jtype` (type ranges allowed, repeatable) | all pairs |
| `shift`, `swap` | `yes`/`no`: exchange moves | `yes`, `yes` |
| `form`, `break` | attempt probabilities of formation / breaking (both > 0 or both 0) | 0, 0 |
| `ebond` | association energy per dynamic bond, added to the potential energy | 0 |
| `barrier` | activation energy `Ea`; moves are attempted with probability `exp(-Ea/kT)` | 0 |
| `prob` | additional attempt probability | 1 |
| `rmin` | lower end of the reaction window | 0 |
| `molecule` | `any`, `inter`, `intra` | `any` |

**3body**: V(r) = 4 eps [(sigma/r)^2n - (sigma/r)^n] between bondable stickers, plus
lambda eps V3(r_ij) V3(r_ik) for every sticker with two partners (V3 = 1 below the minimum,
-V/eps above). lambda = 1 makes the swap barrier-free; lambda > 1 adds a barrier of
(lambda - 1) eps. By default V is shifted to zero at `rc` and rescaled to a depth of -eps;
`eshift no` gives the unshifted form of Sciortino. Keywords: `bondable`,
`rbond` (distance that defines a bond for the output; default: the cutoff of each pair),
`eshift`, and `coeff`.

Different sticker chemistries get their own parameters with
`coeff itype jtype eps sigma n rc lambda` (repeatable, type ranges allowed; only bondable pairs);
the fix arguments are the defaults for all other pairs. Each bond, and each leg of the
three-body term, uses the parameters of its pair of types. The three-body term of a triplet
j-i-k has the strength min(lambda_ij eps_ij, lambda_ik eps_ik), so that with lambda = 1 a swap
between bonds of different strength has no barrier beyond their energy difference:

```
fix             tb stick bond/flux 3body 1.0 1.0 100 1.3 1.0 bondable 1 2 bondable 1 3 &
                coeff 1 3 0.6 1.0 100 1.3 1.0
```

**Output**: mc gives a vector of 11 values (attempts, proposals, accepted shifts, swaps,
formations and breakings, rejections, sum of energy changes, number of dynamic bonds,
accepted moves in the last MC step). 3body gives the energy as a scalar, and a vector
(two-body and three-body energy, bonded pairs, stickers with more than one partner). Both
modes give a per-atom array: partner ID, and number of moves (mc) or partners (3body).

**Requirements**

- mc mode needs:
  - a molecular atom style and an orthogonal box;
  - `special_bonds lj/coul w12 1.0 1.0`;
  - `comm_modify cutoff` >= 2 rc + skin/2;
  - processor sub-domains of at least 8 rc + skin;
  - `extra/bond/per/atom` and `extra/special/per/atom` when bonds can form;
  - rc < 0.95 R0 for FENE bonds.
- LAMMPS cannot combine `bond_style quartic` with unbreakable bond styles in `hybrid`. A
  network with breakable quartic strands therefore needs quartic dynamic bonds, with rc below
  their breaking distance.
- 3body mode needs `newton on` and `comm_modify cutoff` >= rc + skin (the largest rc of all pairs).
- Neither mode supports `run_style respa`.
- The fix writes no restart data: define it again after `read_restart`.

## Examples

```bash
cd examples/kg_network
python3 build_melt.py --chains 200 --out melt.data
lmp -in in.associate                              # reversible association -> network.data
lmp -in in.exchange -var Ea 2.0                   # exchange only, barrier 2 kT

cd ../dumbbells
lmp -in in.dumbbells -var lambda 1.0              # bond lifetime tau ~ 116 (Sciortino 2017)
python3 analyze_nb.py partners_lambda1.0.dump
lmp -in in.swap_event                             # energies during bond swaps
```

## Tests

The tests need the LAMMPS Python module (`cmake --build build --target install-python`),
numpy and h5py.

```bash
python3 tests/test_mc_balance.py --variant kg            # exact detailed balance; variants kg, quartic, fene0, harm05
mpirun -np 4 python3 tests/test_mc_balance.py --variant kg --form 0.5 --break 0.5 --ebond -20 --corner
python3 tests/compare_acceptance_rules.py                 # barrier as gate vs. inside the Metropolis exponent
python3 tests/test_3body.py                               # forces and virial vs. finite differences
python3 tests/test_3body.py --flat                        # barrier-free swap for lambda = 1, also between unequal bonds
python3 tests/test_3body.py --coeff                       # per-type-pair parameters (coeff keyword)
```

## Validation summary

- **MC detailed balance**: the visited topologies of a frozen sticker cluster match the
  exact Boltzmann distribution. This holds for FENE, quartic and harmonic bonds, every move
  type, with and without a barrier, on 1, 2 and 4 MPI ranks, including clusters across periodic
  box corners (48 runs). The LAMMPS potential energy changes by the computed energy change
  of the moves to 1e-10.
- **Barrier**: with the gate the stationary distribution stays exact (deviation <= 2e-11).
  Adding the barrier to the Metropolis energy change instead biases it by 0.24-0.28 (total
  variation distance) for barriers >= kT.
- **Kremer-Grest network**: the bond count is conserved during exchange. The number of
  accepted exchanges drops by exp(2) for each 2 kT of barrier.
- **3body**: forces and virial agree with finite differences to 1e-9, also with per-type-pair
  parameters. For lambda = 1 the potential energy stays at -eps while a bond swaps; between
  bonds of different strength it follows the mixing rule to 1e-16, without a barrier.
- **Sciortino dumbbells**: the bond lifetime is 116.7 (reference: 116). lambda = 1.1 raises it
  to 404, and with lambda = 1.3 bonds practically do not exchange.

## References

Hybrid MD/MC bond exchange:

- A. Perego, F. Khabaz, *Volumetric and rheological properties of vitrimers: a hybrid molecular
  dynamics and Monte Carlo simulation study*, Macromolecules **53**, 8406 (2020).
  [doi:10.1021/acs.macromol.0c01423](https://doi.org/10.1021/acs.macromol.0c01423)
- J. Xia, J. A. Kalow, M. Olvera de la Cruz, *Structure, dynamics, and rheology of vitrimers*,
  Macromolecules **56**, 8080 (2023).
  [doi:10.1021/acs.macromol.3c01366](https://doi.org/10.1021/acs.macromol.3c01366)

Three-body swap potential:

- F. Sciortino, *Three-body potential for simulating bond swaps in molecular dynamics*,
  Eur. Phys. J. E **40**, 3 (2017).
  [doi:10.1140/epje/i2017-11496-5](https://doi.org/10.1140/epje/i2017-11496-5)
- H. Zhao, X. Wei, Y. Fang, K. Gao, T. Yue, L. Zhang, V. Ganesan, F. Meng, J. Liu, *Molecular
  dynamics simulation of the structural, mechanical, and reprocessing properties of vitrimers
  based on a dynamic covalent polymer network*, Macromolecules **55**, 1091 (2022).
  [doi:10.1021/acs.macromol.1c02034](https://doi.org/10.1021/acs.macromol.1c02034)
- M.-H. Lin, T.-W. Lin, C. E. Sing, *Molecular simulation study of penetrant diffusion in
  vitrimer networks*, J. Chem. Phys. **164**, 144908 (2026).
  [doi:10.1063/5.0322457](https://doi.org/10.1063/5.0322457)

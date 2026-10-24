/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   BondFlux: a simulation toolkit for dynamic bonds in polymer networks
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(bond/flux,FixBondFlux);
// clang-format on
#else

#ifndef LMP_FIX_BOND_FLUX_H
#define LMP_FIX_BOND_FLUX_H

#include "fix.h"

namespace LAMMPS_NS {

class FixBondFlux : public Fix {
 public:
  FixBondFlux(class LAMMPS *, int, char **);
  ~FixBondFlux() override;
  int setmask() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void setup(int) override;
  void setup_pre_reverse(int, int) override;
  void min_setup(int) override;
  void post_integrate() override;
  void pre_reverse(int, int) override;
  void min_pre_reverse(int, int) override;
  double compute_scalar() override;
  double compute_vector(int) override;

  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  void set_arrays(int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  double memory_usage() override;

 private:
  enum { MC, THREEBODY };
  enum { ANY, INTER, INTRA };
  enum { FWD_PARTNER, FWD_SPECIAL, REV_TOPO, REV_PHASE };

  int mode;         // MC or THREEBODY
  int commflag;     // selects the data exchanged in forward/reverse comm
  int **bondable;   // bondable[itype][jtype] = 1 if the two sticker types can bond
  double **store;   // per-atom: [0] partner atom ID, [1] move count or coordination

  // Monte Carlo moves of topological bonds

  int btype;                       // dynamic bond type
  double rc, rcsq;                 // max length of any dynamic bond broken or formed
  double rlo, rlosq;               // min length of any dynamic bond broken or formed
  double temperature, kT;          // Metropolis temperature
  double prob, barrier, gate;      // attempt gate = prob * exp(-barrier/kT)
  int shiftflag, swapflag;         // allowed exchange moves
  double pform, pbreak;            // probabilities to attempt formation / breaking
  double ebond;                    // association energy added per dynamic bond
  int molflag;                     // ANY, INTER, INTRA
  int pairflag;                    // 1 if special 1-2 weights are not 1.0
  double wlj, wcoul;               // special 1-2 weights
  class RanMars *random;

  int nmax_work;                   // allocated length of work arrays
  tagint *porig;                   // partner at start of an MC step (owned + ghost)
  int *chflag;                     // 1 if partner changed in the current phase
  double *dcount;                  // move counts accumulated in the current phase
  int *sector;                     // sector index of owned atoms
  int maxorder;
  int *order;                      // randomized list of owned atoms in one sector
  int maxcand;
  int *cand;                       // candidate list for one center atom

  // bins for sticker atoms (owned + ghost)

  int nbinx, nbiny, nbinz, maxbin;
  double binlo[3], bininv[3];
  int *binhead, *binnext;

  // statistics, accumulated on this proc

  double nattempt, nproposed, nshift, nswap, nform, nbreak;
  double nreject_geom, nreject_energy, sum_de;
  bigint nlast;    // accepted moves (all procs) in the last MC step

  // three-body swap potential

  struct Param3 {                  // parameters of one pair of sticker types
    double epsilon, sigma, npow, rc, rcsq, lambda;
    double rmin, rminsq, phic, rbondsq;
  };

  double epsilon, sigma, npow, rc3, lambda;    // defaults from the fix arguments
  double rc3max;                   // largest cutoff of any bondable pair
  Param3 **param3;                 // param3[itype][jtype]
  int eshift;                      // 1 = two-body potential shifted to zero at the cutoff
  double e2, e3, nbondpair, nmulti;
  int maxshort;
  int *neighshort;
  class NeighList *list;

  // Monte Carlo helpers

  void check_domain();
  void topology_to_partner();
  void mc_step();
  void attempt(int);
  int find_candidates(int, tagint, int);
  void move_exchange(int, int, double);
  void move_form(int);
  void move_break(int, int, double);
  int accept(double, double);
  void mark(int);
  void reconcile_topology();
  void build_bins();
  int coord2bin(const double *);
  int permanent_neighbor(int, tagint);
  int allowed_pair(int, int);
  double distsq(int, int);
  double bond_energy(int, int, double);
  double pair_delta(int, int, double);
  double count_bonds();

  // three-body helpers

  void threebody_compute(int, int);
  void set_param3(Param3 &, double, double, double, double, double, double);
  void bond_potential(const Param3 &, double, double &, double &);
  void work_grow();
};

}    // namespace LAMMPS_NS

#endif
#endif

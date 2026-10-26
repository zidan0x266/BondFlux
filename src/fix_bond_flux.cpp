/* ----------------------------------------------------------------------
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
   BondFlux: a simulation toolkit for dynamic bonds in polymer networks.

   mode mc:    Metropolis Monte Carlo moves of topological bonds of one bond
               type: exchange ("shift" = bonded pair + free sticker, "swap" =
               two bonded pairs), and optionally reversible formation and
               breaking.  Proposals are symmetric or corrected by the
               Metropolis-Hastings ratio, so the moves satisfy detailed
               balance with respect to the full force field.  Parallel moves
               are made conflict-free by processing the sub-domain of each
               MPI rank in 2^dim sectors.

   mode 3body: Sciortino three-body swap potential (Eur. Phys. J. E 40, 3
               (2017)) between sticker atoms, added as a force, energy and
               virial contribution, with a 2n-n Lennard-Jones or a
               Stillinger-Weber shaped bond.
------------------------------------------------------------------------- */

#include "fix_bond_flux.h"

#include "atom.h"
#include "bond.h"
#include "citeme.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "random_mars.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;

static const char cite_fix_bond_flux_3body[] =
    "fix bond/flux 3body command: https://doi.org/10.1140/epje/i2017-11496-5\n\n"
    "@Article{Sciortino17,\n"
    " author = {F. Sciortino},\n"
    " title = {Three-Body Potential for Simulating Bond Swaps in Molecular Dynamics},\n"
    " journal = {Eur.\\ Phys.\\ J.~E},\n"
    " year =    2017,\n"
    " volume =  40,\n"
    " pages =   {3}\n"
    "}\n\n";

static constexpr double BIG = 1.0e30;
static constexpr int DELTA = 64;

enum { CAND_EXCHANGE, CAND_FREE };

// three-body parameters of one coeff keyword: type ranges and values
struct Coeff3 {
  int ilo, ihi, jlo, jhi, iarg;
  double val[7];
};

// three-body strength of one triplet keyword: center and leg type ranges
struct Triplet3 {
  int ilo, ihi, jlo, jhi, klo, khi, iarg;
  double strength;
};

/* ---------------------------------------------------------------------- */

FixBondFlux::FixBondFlux(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), bondable(nullptr), store(nullptr), random(nullptr), porig(nullptr),
    chflag(nullptr), dcount(nullptr), sector(nullptr), order(nullptr), cand(nullptr),
    binhead(nullptr), binnext(nullptr), cstate(nullptr), param3(nullptr), strength3(nullptr),
    neighshort(nullptr), list(nullptr)
{
  const std::string fixname = fmt::format("fix {}", style);
  if (narg < 4) utils::missing_cmd_args(FLERR, fixname, error);

  if (strcmp(arg[3], "mc") == 0)
    mode = MC;
  else if (strcmp(arg[3], "3body") == 0)
    mode = THREEBODY;
  else
    error->all(FLERR, 3, "Unknown fix {} mode {}", style, arg[3]);

  // defaults

  btype = 0;
  rc = rcsq = rlo = rlosq = temperature = kT = 0.0;
  eshift = 1;
  prob = 1.0;
  barrier = 0.0;
  gate = 1.0;
  shiftflag = swapflag = 1;
  pform = pbreak = 0.0;
  ebond = 0.0;
  molflag = ANY;
  impulse = 1;
  pairflag = 0;
  wlj = wcoul = 1.0;
  form3 = BOND_LJ;
  nval3 = 5;
  for (double &v : val3) v = 0.0;
  rc3max = 0.0;
  ncontact = 0;
  contact_time = nepisode = nmissed = 0.0;
  double rbond = -1.0;
  int seed = 0;
  int iarg = 0;
  int ifirst3 = 0;
  std::vector<Coeff3> coeffs;
  std::vector<Triplet3> triplets;

  // reads one set of three-body parameters starting at arg[ifirst]
  //   lj: eps sigma n rc lambda      sw: eps sigma B p q rc lambda

  auto read3 = [&](int ifirst, double *v) {
    for (int m = 0; m < nval3; m++) v[m] = utils::numeric(FLERR, arg[ifirst + m], false, lmp);
    if (v[0] <= 0.0) error->all(FLERR, ifirst, "Illegal fix {} 3body epsilon value {}", style, v[0]);
    if (v[1] <= 0.0) error->all(FLERR, ifirst + 1, "Illegal fix {} 3body sigma value {}", style, v[1]);
    if (form3 == BOND_LJ) {
      if (v[2] <= 0.0) error->all(FLERR, ifirst + 2, "Illegal fix {} 3body exponent value {}", style, v[2]);
    } else {
      if (v[2] <= 0.0) error->all(FLERR, ifirst + 2, "Illegal fix {} 3body sw B value {}", style, v[2]);
      if ((v[4] < 0.0) || (v[3] <= v[4]))
        error->all(FLERR, ifirst + 3, "Fix {} 3body sw exponents must satisfy p > q >= 0", style);
    }
    if (v[nval3 - 2] <= 0.0)
      error->all(FLERR, ifirst + nval3 - 2, "Illegal fix {} 3body cutoff value {}", style, v[nval3 - 2]);
    if (v[nval3 - 1] < 0.0)
      error->all(FLERR, ifirst + nval3 - 1, "Illegal fix {} 3body lambda value {}", style, v[nval3 - 1]);
  };

  if (mode == MC) {
    if (narg < 9) utils::missing_cmd_args(FLERR, fixname + " mc", error);
    nevery = utils::inumeric(FLERR, arg[4], false, lmp);
    if (nevery <= 0) error->all(FLERR, 4, "Illegal fix {} mc Nevery value {}", style, nevery);
    btype = utils::expand_type_int(FLERR, arg[5], Atom::BOND, lmp);
    if ((btype < 1) || (btype > atom->nbondtypes))
      error->all(FLERR, 5, "Invalid bond type {} in fix {} command", btype, style);
    rc = utils::numeric(FLERR, arg[6], false, lmp);
    if (rc <= 0.0) error->all(FLERR, 6, "Illegal fix {} mc cutoff value {}", style, rc);
    rcsq = rc * rc;
    temperature = utils::numeric(FLERR, arg[7], false, lmp);
    if (temperature <= 0.0)
      error->all(FLERR, 7, "Illegal fix {} mc temperature value {}", style, temperature);
    seed = utils::inumeric(FLERR, arg[8], false, lmp);
    if (seed <= 0) error->all(FLERR, 8, "Illegal fix {} mc seed value {}", style, seed);
    iarg = 9;
  } else {
    ifirst3 = 4;
    if ((narg > 4) && (strcmp(arg[4], "lj") == 0)) {
      ifirst3 = 5;
    } else if ((narg > 4) && (strcmp(arg[4], "sw") == 0)) {
      form3 = BOND_SW;
      ifirst3 = 5;
    }
    nval3 = (form3 == BOND_LJ) ? 5 : 7;
    if (narg < ifirst3 + nval3) utils::missing_cmd_args(FLERR, fixname + " 3body", error);
    read3(ifirst3, val3);
    iarg = ifirst3 + nval3;
  }

  // all pairs of atom types are bondable unless the bondable keyword is used

  const int ntypes = atom->ntypes;
  memory->create(bondable, ntypes + 1, ntypes + 1, "bond/flux:bondable");
  for (int i = 0; i <= ntypes; i++)
    for (int j = 0; j <= ntypes; j++) bondable[i][j] = 1;
  int firstbondable = 1;

  while (iarg < narg) {
    if (strcmp(arg[iarg], "bondable") == 0) {
      if (iarg + 3 > narg) utils::missing_cmd_args(FLERR, fixname + " bondable", error);
      if (firstbondable) {
        for (int i = 0; i <= ntypes; i++)
          for (int j = 0; j <= ntypes; j++) bondable[i][j] = 0;
        firstbondable = 0;
      }
      int ilo, ihi, jlo, jhi;
      utils::bounds_typelabel(FLERR, arg[iarg + 1], 1, ntypes, ilo, ihi, lmp, Atom::ATOM);
      utils::bounds_typelabel(FLERR, arg[iarg + 2], 1, ntypes, jlo, jhi, lmp, Atom::ATOM);
      for (int i = ilo; i <= ihi; i++)
        for (int j = jlo; j <= jhi; j++) bondable[i][j] = bondable[j][i] = 1;
      iarg += 3;
    } else if ((strcmp(arg[iarg], "prob") == 0) && (mode == MC)) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, fixname + " prob", error);
      prob = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if ((prob <= 0.0) || (prob > 1.0))
        error->all(FLERR, iarg + 1, "Illegal fix {} prob value {}", style, prob);
      iarg += 2;
    } else if ((strcmp(arg[iarg], "barrier") == 0) && (mode == MC)) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, fixname + " barrier", error);
      barrier = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (barrier < 0.0) error->all(FLERR, iarg + 1, "Illegal fix {} barrier value {}", style, barrier);
      iarg += 2;
    } else if ((strcmp(arg[iarg], "shift") == 0) && (mode == MC)) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, fixname + " shift", error);
      shiftflag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if ((strcmp(arg[iarg], "swap") == 0) && (mode == MC)) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, fixname + " swap", error);
      swapflag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if ((strcmp(arg[iarg], "form") == 0) && (mode == MC)) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, fixname + " form", error);
      pform = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if ((pform < 0.0) || (pform > 1.0))
        error->all(FLERR, iarg + 1, "Illegal fix {} form probability {}", style, pform);
      iarg += 2;
    } else if ((strcmp(arg[iarg], "break") == 0) && (mode == MC)) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, fixname + " break", error);
      pbreak = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if ((pbreak < 0.0) || (pbreak > 1.0))
        error->all(FLERR, iarg + 1, "Illegal fix {} break probability {}", style, pbreak);
      iarg += 2;
    } else if ((strcmp(arg[iarg], "ebond") == 0) && (mode == MC)) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, fixname + " ebond", error);
      ebond = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if ((strcmp(arg[iarg], "rmin") == 0) && (mode == MC)) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, fixname + " rmin", error);
      rlo = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if ((rlo < 0.0) || (rlo >= rc))
        error->all(FLERR, iarg + 1, "Fix {} rmin {} must be between 0 and the cutoff {}", style,
                   rlo, rc);
      rlosq = rlo * rlo;
      iarg += 2;
    } else if ((strcmp(arg[iarg], "eshift") == 0) && (mode == THREEBODY)) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, fixname + " eshift", error);
      eshift = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if ((strcmp(arg[iarg], "molecule") == 0) && (mode == MC)) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, fixname + " molecule", error);
      if (strcmp(arg[iarg + 1], "any") == 0)
        molflag = ANY;
      else if (strcmp(arg[iarg + 1], "inter") == 0)
        molflag = INTER;
      else if (strcmp(arg[iarg + 1], "intra") == 0)
        molflag = INTRA;
      else
        error->all(FLERR, iarg + 1, "Unknown fix {} molecule option {}", style, arg[iarg + 1]);
      iarg += 2;
    } else if ((strcmp(arg[iarg], "coeff") == 0) && (mode == THREEBODY)) {
      if (iarg + 3 + nval3 > narg) utils::missing_cmd_args(FLERR, fixname + " coeff", error);
      Coeff3 c;
      c.iarg = iarg;
      utils::bounds_typelabel(FLERR, arg[iarg + 1], 1, ntypes, c.ilo, c.ihi, lmp, Atom::ATOM);
      utils::bounds_typelabel(FLERR, arg[iarg + 2], 1, ntypes, c.jlo, c.jhi, lmp, Atom::ATOM);
      read3(iarg + 3, c.val);
      coeffs.push_back(c);
      iarg += 3 + nval3;
    } else if ((strcmp(arg[iarg], "triplet") == 0) && (mode == THREEBODY)) {
      if (iarg + 5 > narg) utils::missing_cmd_args(FLERR, fixname + " triplet", error);
      Triplet3 t;
      t.iarg = iarg;
      utils::bounds_typelabel(FLERR, arg[iarg + 1], 1, ntypes, t.ilo, t.ihi, lmp, Atom::ATOM);
      utils::bounds_typelabel(FLERR, arg[iarg + 2], 1, ntypes, t.jlo, t.jhi, lmp, Atom::ATOM);
      utils::bounds_typelabel(FLERR, arg[iarg + 3], 1, ntypes, t.klo, t.khi, lmp, Atom::ATOM);
      t.strength = utils::numeric(FLERR, arg[iarg + 4], false, lmp);
      if (t.strength < 0.0)
        error->all(FLERR, iarg + 4, "Illegal fix {} 3body triplet strength {}", style, t.strength);
      triplets.push_back(t);
      iarg += 5;
    } else if ((strcmp(arg[iarg], "impulse") == 0) && (mode == MC)) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, fixname + " impulse", error);
      impulse = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if ((strcmp(arg[iarg], "contact") == 0) && (mode == MC)) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, fixname + " contact", error);
      ncontact = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (ncontact <= 0) error->all(FLERR, iarg + 1, "Illegal fix {} contact value {}", style, ncontact);
      iarg += 2;
    } else if ((strcmp(arg[iarg], "rbond") == 0) && (mode == THREEBODY)) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, fixname + " rbond", error);
      rbond = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (rbond <= 0.0) error->all(FLERR, iarg + 1, "Illegal fix {} rbond value {}", style, rbond);
      iarg += 2;
    } else
      error->all(FLERR, iarg, "Unknown fix {} {} keyword {}", style, arg[3], arg[iarg]);
  }

  if (mode == MC) {
    if ((pform > 0.0) != (pbreak > 0.0))
      error->all(FLERR, Error::NOLASTLINE,
                 "Fix {} form and break must be used together: bond formation and breaking "
                 "are each other's reverse move", style);
    if (!shiftflag && !swapflag && (pform == 0.0))
      error->all(FLERR, Error::NOLASTLINE, "Fix {} has all Monte Carlo moves disabled", style);
    if (atom->molecular != Atom::MOLECULAR)
      error->all(FLERR, Error::NOLASTLINE, "Fix {} mc requires a molecular system", style);
    if ((molflag != ANY) && !atom->molecule_flag)
      error->all(FLERR, Error::NOLASTLINE,
                 "Fix {} molecule keyword requires an atom style with molecule IDs", style);
    if (ncontact && (nevery % ncontact))
      error->all(FLERR, Error::NOLASTLINE,
                 "Fix {} contact interval {} must divide the MC interval {}", style, ncontact, nevery);
    force_reneighbor = 1;
    next_reneighbor = -1;
    vector_flag = 1;
    size_vector = 16;
    global_freq = 1;
    extvector = 0;
    if (ebond != 0.0) {
      scalar_flag = 1;
      extscalar = 1;
      energy_global_flag = 1;
      thermo_energy = 1;
    }
    random = new RanMars(lmp, seed + comm->me);
  } else {
    if (lmp->citeme) lmp->citeme->add(cite_fix_bond_flux_3body);

    // the fix arguments are the parameters of every pair of sticker types,
    //   coeff keywords override them for selected bondable pairs

    Param3 pdefault;
    set_param3(pdefault, val3, rbond, ifirst3);
    memory->create(param3, ntypes + 1, ntypes + 1, "bond/flux:param3");
    for (int i = 0; i <= ntypes; i++)
      for (int j = 0; j <= ntypes; j++) param3[i][j] = pdefault;
    for (const auto &c : coeffs)
      for (int i = c.ilo; i <= c.ihi; i++)
        for (int j = c.jlo; j <= c.jhi; j++) {
          if (!bondable[i][j])
            error->all(FLERR, c.iarg, "Fix {} 3body coeff for atom types {} and {} that are not bondable",
                       style, i, j);
          set_param3(param3[i][j], c.val, rbond, c.iarg + 3);
          param3[j][i] = param3[i][j];
        }
    rc3max = 0.0;
    for (int i = 1; i <= ntypes; i++)
      for (int j = 1; j <= ntypes; j++)
        if (bondable[i][j]) rc3max = MAX(rc3max, param3[i][j].rc);

    // three-body strength of a triplet j-i-k: the smaller lambda*eps of its two legs,
    //   unless set by a triplet keyword

    memory->create(strength3, ntypes + 1, ntypes + 1, ntypes + 1, "bond/flux:strength3");
    for (int i = 0; i <= ntypes; i++)
      for (int j = 0; j <= ntypes; j++)
        for (int k = 0; k <= ntypes; k++)
          strength3[i][j][k] = MIN(param3[i][j].lambda * param3[i][j].epsilon,
                                   param3[i][k].lambda * param3[i][k].epsilon);
    for (const auto &t : triplets)
      for (int i = t.ilo; i <= t.ihi; i++)
        for (int j = t.jlo; j <= t.jhi; j++)
          for (int k = t.klo; k <= t.khi; k++) {
            if (!bondable[i][j] || !bondable[i][k])
              error->all(FLERR, t.iarg,
                         "Fix {} 3body triplet {} {} {}: both legs must be bondable pairs", style, i,
                         j, k);
            strength3[i][j][k] = strength3[i][k][j] = t.strength;
          }
    scalar_flag = 1;
    vector_flag = 1;
    size_vector = 4;
    global_freq = 1;
    extscalar = 1;
    extvector = 1;
    energy_global_flag = 1;
    virial_global_flag = 1;
    thermo_energy = thermo_virial = 1;
  }

  peratom_flag = 1;
  size_peratom_cols = 2;
  peratom_freq = 1;
  create_attribute = 1;
  maxexchange = 4;
  comm_forward = 0;
  comm_reverse = 3;

  FixBondFlux::grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);
  for (int i = 0; i < atom->nlocal; i++) store[i][0] = store[i][1] = cstate[i][0] = cstate[i][1] = 0.0;

  nmax_work = 0;
  maxorder = maxcand = maxshort = 0;
  nbinx = nbiny = nbinz = 1;
  maxbin = 0;
  nattempt = nproposed = nshift = nswap = nform = nbreak = 0.0;
  nreject_geom = nreject_energy = sum_de = 0.0;
  nlast = 0;
  e2 = e3 = nbondpair = nmulti = 0.0;
}

/* ---------------------------------------------------------------------- */

FixBondFlux::~FixBondFlux()
{
  atom->delete_callback(id, Atom::GROW);

  delete random;
  memory->destroy(bondable);
  memory->destroy(param3);
  memory->destroy(strength3);
  memory->destroy(store);
  memory->destroy(cstate);
  memory->destroy(porig);
  memory->destroy(chflag);
  memory->destroy(dcount);
  memory->destroy(sector);
  memory->destroy(order);
  memory->destroy(cand);
  memory->destroy(binhead);
  memory->destroy(binnext);
  memory->destroy(neighshort);
}

/* ---------------------------------------------------------------------- */

int FixBondFlux::setmask()
{
  int mask = 0;
  if (mode == MC) mask |= POST_INTEGRATE;
  if (mode == THREEBODY) {
    mask |= PRE_REVERSE;
    mask |= MIN_PRE_REVERSE;
  }
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBondFlux::init()
{
  if (utils::strmatch(update->integrate_style, "^respa"))
    error->all(FLERR, Error::NOLASTLINE, "Fix {} does not support run style respa", style);

  if (mode == MC) {
    if (force->bond == nullptr)
      error->all(FLERR, Error::NOLASTLINE, "Fix {} mc requires a bond style", style);
    if (!force->bond->setflag[btype])
      error->all(FLERR, Error::NOLASTLINE, "Fix {} mc: bond coefficients for bond type {} are not set",
                 style, btype);
    if (domain->triclinic)
      error->all(FLERR, Error::NOLASTLINE, "Fix {} mc requires an orthogonal simulation box", style);

    // 1-3 and 1-4 neighbors must not be special, so that a bond move only
    //   changes the 1-2 status of the atoms involved

    const int qflag = atom->q_flag;
    if ((force->special_lj[2] != 1.0) || (force->special_lj[3] != 1.0) || force->special_onefive ||
        (qflag && ((force->special_coul[2] != 1.0) || (force->special_coul[3] != 1.0))))
      error->all(FLERR, Error::NOLASTLINE,
                 "Fix {} mc requires special_bonds weights of 1.0 for 1-3 and 1-4 neighbors", style);
    if (!qflag && ((force->special_coul[2] != 1.0) || (force->special_coul[3] != 1.0)) &&
        (comm->me == 0))
      error->warning(FLERR,
                     "Fix {} mc does not update 1-3 and 1-4 special neighbor lists; set their "
                     "Coulomb weights to 1.0 as well (e.g. special_bonds lj/coul 0.0 1.0 1.0) so "
                     "that LAMMPS does not create them", style);

    wlj = force->special_lj[1];
    wcoul = qflag ? force->special_coul[1] : 1.0;
    pairflag = 0;
    if (force->pair && ((wlj != 1.0) || (wcoul != 1.0))) {
      pairflag = 1;
      if (force->pair->single_enable == 0)
        error->all(FLERR, Error::NOLASTLINE,
                   "Pair style {} does not support fix {} mc with special_bonds 1-2 weights "
                   "other than 1.0", force->pair_style, style);
    }

    if ((force->angle || force->dihedral || force->improper) && (comm->me == 0))
      error->warning(FLERR,
                     "Fix {} mc does not update angles, dihedrals, or impropers; "
                     "these must not include bonds of type {}", style, btype);

    // FENE bonds: avoid the region where bond style fene reports overstretched bonds

    Bond *bfene = force->bond_match("fene");
    if (bfene && bfene->setflag[btype]) {
      int dim;
      auto *r0 = (double *) bfene->extract("r0", dim);
      if (r0 && (rcsq >= 0.9 * r0[btype] * r0[btype]))
        error->all(FLERR, Error::NOLASTLINE,
                   "Fix {} mc cutoff {} must be shorter than {:.6} (0.95 R0) for fene bond type {}",
                   style, rc, sqrt(0.9) * r0[btype], btype);
    }

    kT = force->boltz * temperature;
    gate = prob * exp(-barrier / kT);
    comm_forward = 2 + atom->maxspecial;

  } else {
    if (!force->newton)
      error->all(FLERR, Error::NOLASTLINE, "Fix {} 3body requires newton on", style);

    auto *req = neighbor->add_request(this, NeighConst::REQ_FULL);
    req->set_cutoff_fixed(rc3max);
    comm_forward = 0;

    // the fix supplies the complete interaction between bondable sticker types,
    //   so a non-zero pair style interaction between them is likely a setup error

    if (force->pair && force->pair->single_enable && force->pair->cutsq) {
      const int ntypes = atom->ntypes;
      std::vector<int> present(ntypes + 1, 0), all(ntypes + 1, 0);
      for (int i = 0; i < atom->nlocal; i++)
        if (atom->mask[i] & groupbit) present[atom->type[i]] = 1;
      MPI_Allreduce(present.data(), all.data(), ntypes + 1, MPI_INT, MPI_MAX, world);
      int nonzero = 0;
      if (atom->nlocal > 0) {
        double fdummy;
        for (int i = 1; i <= ntypes; i++)
          for (int j = i; j <= ntypes; j++) {
            if (!all[i] || !all[j] || !bondable[i][j]) continue;
            const double rsq = 0.81 * param3[i][j].rminsq;
            if (rsq >= force->pair->cutsq[i][j]) continue;
            if (force->pair->single(0, 0, i, j, rsq, 1.0, 1.0, fdummy) != 0.0) nonzero = 1;
          }
      }
      int nonzeroall;
      MPI_Allreduce(&nonzero, &nonzeroall, 1, MPI_INT, MPI_MAX, world);
      if (nonzeroall && (comm->me == 0))
        error->warning(FLERR,
                       "Fix {} 3body adds the full interaction between bondable sticker types; "
                       "the pair style interaction between them should be switched off", style);
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixBondFlux::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixBondFlux::setup(int /*vflag*/)
{
  if (mode == MC) {
    check_domain();
    topology_to_partner();
  } else {
    double cutghost = MIN(comm->cutghost[0], comm->cutghost[1]);
    if (domain->dimension == 3) cutghost = MIN(cutghost, comm->cutghost[2]);
    if (!domain->triclinic && (cutghost < rc3max + neighbor->skin))
      error->all(FLERR, Error::NOLASTLINE,
                 "Fix {} 3body needs a communication cutoff of at least {:.6}; "
                 "use comm_modify cutoff", style, rc3max + neighbor->skin);
  }
}

/* ---------------------------------------------------------------------- */

void FixBondFlux::min_setup(int vflag)
{
  setup(vflag);
}

/* ---------------------------------------------------------------------- */

void FixBondFlux::setup_pre_reverse(int eflag, int vflag)
{
  if (mode == THREEBODY) threebody_compute(eflag, vflag);
}

/* ---------------------------------------------------------------------- */

void FixBondFlux::pre_reverse(int eflag, int vflag)
{
  if (mode == THREEBODY) threebody_compute(eflag, vflag);
}

/* ---------------------------------------------------------------------- */

void FixBondFlux::min_pre_reverse(int eflag, int vflag)
{
  if (mode == THREEBODY) threebody_compute(eflag, vflag);
}

/* ---------------------------------------------------------------------- */

void FixBondFlux::post_integrate()
{
  if (mode != MC) return;
  const int mcstep = (update->ntimestep % nevery) == 0;
  if (ncontact && (update->ntimestep % ncontact == 0)) contact_check(mcstep);
  if (mcstep) mc_step();
}

/* ----------------------------------------------------------------------
   check that ghost atoms and sub-domain sizes allow conflict-free moves
   a move centered on atom I involves atoms within 2*rc of I
   moves of two procs in the same sector are separated by at least
   half a sub-domain minus the drift of atoms out of their sub-domain
------------------------------------------------------------------------- */

void FixBondFlux::check_domain()
{
  const int dim = domain->dimension;
  const double skin = neighbor->skin;
  const double ghostneed = 2.0 * rc + 0.5 * skin;
  const double subneed = 8.0 * rc + skin;

  double cutghost = MIN(comm->cutghost[0], comm->cutghost[1]);
  if (dim == 3) cutghost = MIN(cutghost, comm->cutghost[2]);
  if (cutghost < ghostneed)
    error->all(FLERR, Error::NOLASTLINE,
               "Fix {} mc needs a communication cutoff of at least {:.6}; use comm_modify cutoff",
               style, ghostneed);

  int flag = 0;
  for (int d = 0; d < dim; d++)
    if (domain->subhi[d] - domain->sublo[d] < subneed) flag = 1;
  int flagall;
  MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_MAX, world);
  if (flagall)
    error->all(FLERR, Error::NOLASTLINE,
               "Processor sub-domains are too small for fix {} mc: each side must be at least "
               "{:.6}; use fewer MPI processes or the processors command", style, subneed);
}

/* ----------------------------------------------------------------------
   set partner IDs of owned and ghost atoms from bonds of type btype
------------------------------------------------------------------------- */

void FixBondFlux::topology_to_partner()
{
  work_grow();

  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;
  const int newton_bond = force->newton_bond;

  for (int i = 0; i < nall; i++) {
    store[i][0] = 0.0;
    dcount[i] = 0.0;
  }

  int flag = 0;
  for (int i = 0; i < nlocal; i++) {
    for (int m = 0; m < num_bond[i]; m++) {
      if (bond_type[i][m] != btype) continue;
      if (!(mask[i] & groupbit)) flag |= 1;
      tagint jtag = bond_atom[i][m];
      store[i][0] = (double) jtag;
      dcount[i] += 1.0;
      if (newton_bond) {
        int j = atom->map(jtag);
        if (j < 0) {
          flag |= 2;
          continue;
        }
        store[j][0] = (double) tag[i];
        dcount[j] += 1.0;
      }
    }
  }

  if (newton_bond) {
    commflag = REV_TOPO;
    comm->reverse_comm(this);
  }

  for (int i = 0; i < nlocal; i++)
    if (dcount[i] > 1.5) flag |= 4;

  commflag = FWD_PARTNER;
  comm->forward_comm(this);

  for (int i = 0; i < nlocal; i++) {
    tagint p = (tagint) store[i][0];
    if (p == 0) continue;
    int j = atom->map(p);
    if (j < 0) {
      flag |= 2;
      continue;
    }
    if (!(mask[j] & groupbit)) flag |= 1;
    if (!allowed_pair(i, j)) flag |= 8;
  }

  int flagall;
  MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_BOR, world);
  if (flagall & 1)
    error->all(FLERR, Error::NOLASTLINE,
               "All atoms with bonds of type {} must be in the group of fix {}", btype, style);
  if (flagall & 2)
    error->all(FLERR, Error::NOLASTLINE,
               "Fix {} could not find a bonded atom of a type {} bond; "
               "increase the communication cutoff", style, btype);
  if (flagall & 4)
    error->all(FLERR, Error::NOLASTLINE, "Fix {} found an atom with more than one bond of type {}",
               style, btype);
  if (flagall & 8)
    error->all(FLERR, Error::NOLASTLINE,
               "A bond of type {} connects atoms that fix {} does not allow to bond", btype, style);

  for (int i = nlocal; i < nall; i++) dcount[i] = 0.0;
}

/* ----------------------------------------------------------------------
   one MC step: every owned sticker is visited once, sector by sector
------------------------------------------------------------------------- */

void FixBondFlux::mc_step()
{
  check_domain();

  // current ghost positions (called between integration and Verlet comm)

  comm->forward_comm();

  // partner IDs from the current topology, then 1-2 special lists of ghosts

  topology_to_partner();
  commflag = FWD_SPECIAL;
  comm->forward_comm(this);

  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;
  const int dim = domain->dimension;
  int *mask = atom->mask;
  double **x = atom->x;

  for (int i = 0; i < nall; i++) {
    porig[i] = (tagint) store[i][0];
    chflag[i] = 0;
    dcount[i] = 0.0;
  }

  build_bins();

  // assign each owned sticker to one of 2^dim sectors of the sub-domain

  for (int i = 0; i < nlocal; i++) {
    sector[i] = -1;
    if (!(mask[i] & groupbit)) continue;
    int s = 0;
    for (int d = 0; d < dim; d++)
      if (x[i][d] >= 0.5 * (domain->sublo[d] + domain->subhi[d])) s |= (1 << d);
    sector[i] = s;
  }

  if (nlocal > maxorder) {
    maxorder = atom->nmax;
    memory->destroy(order);
    memory->create(order, maxorder, "bond/flux:order");
  }

  const double accepted_before = nshift + nswap + nform + nbreak;
  const double bonds_before = nform - nbreak;
  const int nsector = 1 << dim;

  for (int s = 0; s < nsector; s++) {
    int n = 0;
    for (int i = 0; i < nlocal; i++)
      if (sector[i] == s) order[n++] = i;

    // random permutation of the sticker order (Fisher-Yates)

    for (int m = 0; m < n - 1; m++) {
      int k = m + static_cast<int>(random->uniform() * (n - m));
      if (k >= n) k = n - 1;
      int tmp = order[m];
      order[m] = order[k];
      order[k] = tmp;
    }

    for (int m = 0; m < n; m++) {
      int i = order[m];
      if ((store[i][0] == 0.0) && (pform == 0.0)) continue;
      if ((gate < 1.0) && (random->uniform() >= gate)) continue;
      attempt(i);
    }

    // send changes of ghost atoms to their owners, then refresh all ghosts

    commflag = REV_PHASE;
    comm->reverse_comm(this);
    commflag = FWD_PARTNER;
    comm->forward_comm(this);
    for (int i = nlocal; i < nall; i++) {
      chflag[i] = 0;
      dcount[i] = 0.0;
    }
  }

  double local[2] = {nshift + nswap + nform + nbreak - accepted_before,
                     nform - nbreak - bonds_before};
  double total[2];
  MPI_Allreduce(local, total, 2, MPI_DOUBLE, MPI_SUM, world);
  nlast = (bigint) total[0];
  atom->nbonds += (bigint) total[1];

  if (impulse) correct_velocities();
  reconcile_topology();
}

/* ----------------------------------------------------------------------
   offer one move to owned sticker I:
     bonded: break the bond with probability pbreak, else try an exchange
     free:   form a bond with probability pform
   every bond that is broken or formed must have a length in [rmin,rc),
   so that each move can be proposed in reverse from the same sticker
------------------------------------------------------------------------- */

void FixBondFlux::attempt(int i)
{
  const tagint pi = (tagint) store[i][0];

  if (pi == 0) {
    if (random->uniform() < pform) move_form(i);
    return;
  }

  int k = atom->map(pi);
  if (k < 0) return;
  k = domain->closest_image(i, k);
  const double rsq_ik = distsq(i, k);
  if ((rsq_ik >= rcsq) || (rsq_ik < rlosq)) return;

  if ((pbreak > 0.0) && (random->uniform() < pbreak))
    move_break(i, k, rsq_ik);
  else if (shiftflag || swapflag)
    move_exchange(i, k, rsq_ik);
}

/* ----------------------------------------------------------------------
   collect stickers within rc of I that may bond with I
   which = CAND_EXCHANGE: free (if shifts allowed) and bonded (if swaps allowed)
   which = CAND_FREE:     free stickers only
   PI = current partner of I (excluded), 0 if none
------------------------------------------------------------------------- */

int FixBondFlux::find_candidates(int i, tagint pi, int which)
{
  double **x = atom->x;
  tagint *tag = atom->tag;

  int ncand = 0;
  const int ib = coord2bin(x[i]);
  const int ibx = ib % nbinx;
  const int iby = (ib / nbinx) % nbiny;
  const int ibz = ib / (nbinx * nbiny);
  const int zlo = (domain->dimension == 3) ? -1 : 0;
  const int zhi = (domain->dimension == 3) ? 1 : 0;

  for (int dz = zlo; dz <= zhi; dz++) {
    int jbz = ibz + dz;
    if ((jbz < 0) || (jbz >= nbinz)) continue;
    for (int dy = -1; dy <= 1; dy++) {
      int jby = iby + dy;
      if ((jby < 0) || (jby >= nbiny)) continue;
      for (int dx = -1; dx <= 1; dx++) {
        int jbx = ibx + dx;
        if ((jbx < 0) || (jbx >= nbinx)) continue;
        for (int j = binhead[(jbz * nbiny + jby) * nbinx + jbx]; j >= 0; j = binnext[j]) {
          if ((tag[j] == tag[i]) || (tag[j] == pi)) continue;
          const double rsq = distsq(i, j);
          if ((rsq >= rcsq) || (rsq < rlosq)) continue;
          if (!allowed_pair(i, j)) continue;
          if (permanent_neighbor(i, tag[j])) continue;
          if (store[j][0] == 0.0) {
            if ((which == CAND_EXCHANGE) && !shiftflag) continue;
          } else {
            if ((which == CAND_FREE) || !swapflag) continue;
          }
          if (ncand == maxcand) {
            maxcand += DELTA;
            memory->grow(cand, maxcand, "bond/flux:cand");
          }
          cand[ncand++] = j;
        }
      }
    }
  }
  return ncand;
}

/* ----------------------------------------------------------------------
   contact diagnostics, every ncontact steps before the MC step:
   a sticker is in contact while it has a candidate for the move it would be
   offered; a contact episode during which no MC step took place is missed
   by the MC interval Nevery
------------------------------------------------------------------------- */

void FixBondFlux::contact_check(int mcstep)
{
  comm->forward_comm();
  topology_to_partner();
  build_bins();

  int *mask = atom->mask;
  const int nlocal = atom->nlocal;
  const double dtc = ncontact * update->dt;

  // the dynamic partner of each sticker for permanent_neighbor(): porig of the last MC step
  //   is stale here, since atoms are sorted and migrate between MC steps

  for (int i = 0; i < nlocal; i++) porig[i] = (tagint) store[i][0];

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    const int now = has_candidate(i);
    if (now) {
      if (cstate[i][0] == 0.0) cstate[i][1] = 0.0;
      if (mcstep) cstate[i][1] = 1.0;
      contact_time += dtc;
    } else if (cstate[i][0] != 0.0) {
      nepisode += 1.0;
      if (cstate[i][1] == 0.0) nmissed += 1.0;
    }
    cstate[i][0] = (double) now;
  }
}

/* ----------------------------------------------------------------------
   1 if owned sticker I has a candidate: an exchange partner if I is bonded
   (with its bond inside the reaction window), a free partner if I is free
   and bonds can form
------------------------------------------------------------------------- */

int FixBondFlux::has_candidate(int i)
{
  const tagint pi = (tagint) store[i][0];
  if (pi == 0) return ((pform > 0.0) && (find_candidates(i, 0, CAND_FREE) > 0)) ? 1 : 0;

  if (!shiftflag && !swapflag) return 0;
  int k = atom->map(pi);
  if (k < 0) return 0;
  k = domain->closest_image(i, k);
  const double rsq_ik = distsq(i, k);
  if ((rsq_ik >= rcsq) || (rsq_ik < rlosq)) return 0;
  return (find_candidates(i, pi, CAND_EXCHANGE) > 0) ? 1 : 0;
}

/* ----------------------------------------------------------------------
   Metropolis(-Hastings) test with energy change DE and proposal ratio RATIO
------------------------------------------------------------------------- */

int FixBondFlux::accept(double de, double ratio)
{
  const double arg = log(ratio) - de / kT;
  if (arg >= 0.0) return 1;
  return (random->uniform() < exp(arg)) ? 1 : 0;
}

/* ----------------------------------------------------------------------
   flag a changed atom and count its move
------------------------------------------------------------------------- */

void FixBondFlux::mark(int a)
{
  chflag[a] = 1;
  if (a < atom->nlocal)
    store[a][1] += 1.0;
  else
    dcount[a] += 1.0;
}

/* ----------------------------------------------------------------------
   exchange move of bonded sticker I with partner K:
     shift: I-K + J      ->  I-J + K
     swap:  I-K + J-L    ->  I-J + K-L
------------------------------------------------------------------------- */

void FixBondFlux::move_exchange(int i, int k, double rsq_ik)
{
  double **x = atom->x;
  tagint *tag = atom->tag;
  int *mask = atom->mask;

  nattempt += 1.0;
  const int ncand = find_candidates(i, tag[k], CAND_EXCHANGE);
  if (ncand == 0) return;
  nproposed += 1.0;

  int pick = static_cast<int>(random->uniform() * ncand);
  if (pick >= ncand) pick = ncand - 1;
  const int j = cand[pick];
  const tagint pj = (tagint) store[j][0];
  const double rsq_ij = distsq(i, j);

  if (pj == 0) {
    double de = bond_energy(i, j, rsq_ij) - bond_energy(i, k, rsq_ik) +
        pair_delta(i, k, rsq_ik) - pair_delta(i, j, rsq_ij);
    if (!accept(de, 1.0)) {
      nreject_energy += 1.0;
      return;
    }
    store[i][0] = (double) tag[j];
    store[j][0] = (double) tag[i];
    store[k][0] = 0.0;
    mark(i);
    mark(j);
    mark(k);
    nshift += 1.0;
    sum_de += de;
    return;
  }

  int l = atom->map(pj);
  if (l < 0) {
    nreject_geom += 1.0;
    return;
  }
  l = domain->closest_image(j, l);
  if ((tag[l] == tag[i]) || (tag[l] == tag[k])) {
    nreject_geom += 1.0;
    return;
  }
  const double rsq_jl = distsq(j, l);
  double del[3] = {x[k][0] - x[l][0], x[k][1] - x[l][1], x[k][2] - x[l][2]};
  domain->minimum_image(FLERR, del);
  const double rsq_kl = del[0] * del[0] + del[1] * del[1] + del[2] * del[2];
  if ((rsq_jl >= rcsq) || (rsq_kl >= rcsq) || (rsq_jl < rlosq) || (rsq_kl < rlosq) ||
      !(mask[l] & groupbit) || !allowed_pair(k, l) || permanent_neighbor(k, tag[l])) {
    nreject_geom += 1.0;
    return;
  }

  double de = bond_energy(i, j, rsq_ij) + bond_energy(k, l, rsq_kl) -
      bond_energy(i, k, rsq_ik) - bond_energy(j, l, rsq_jl) + pair_delta(i, k, rsq_ik) +
      pair_delta(j, l, rsq_jl) - pair_delta(i, j, rsq_ij) - pair_delta(k, l, rsq_kl);
  if (!accept(de, 1.0)) {
    nreject_energy += 1.0;
    return;
  }
  store[i][0] = (double) tag[j];
  store[j][0] = (double) tag[i];
  store[k][0] = (double) tag[l];
  store[l][0] = (double) tag[k];
  mark(i);
  mark(j);
  mark(k);
  mark(l);
  nswap += 1.0;
  sum_de += de;
}

/* ----------------------------------------------------------------------
   formation move of free sticker I with a free candidate J
   reverse move = breaking of I-J proposed from I, which has probability pbreak
   while formation has probability pform/ncand -> Hastings ratio
------------------------------------------------------------------------- */

void FixBondFlux::move_form(int i)
{
  tagint *tag = atom->tag;

  nattempt += 1.0;
  const int ncand = find_candidates(i, 0, CAND_FREE);
  if (ncand == 0) return;
  nproposed += 1.0;

  int pick = static_cast<int>(random->uniform() * ncand);
  if (pick >= ncand) pick = ncand - 1;
  const int j = cand[pick];
  const double rsq_ij = distsq(i, j);

  double de = bond_energy(i, j, rsq_ij) - pair_delta(i, j, rsq_ij) + ebond;
  if (!accept(de, pbreak * ncand / pform)) {
    nreject_energy += 1.0;
    return;
  }
  store[i][0] = (double) tag[j];
  store[j][0] = (double) tag[i];
  mark(i);
  mark(j);
  nform += 1.0;
  sum_de += de;
}

/* ----------------------------------------------------------------------
   breaking move of bonded sticker I with partner K
   reverse move = formation of I-K proposed from I among the free candidates
   of I after breaking (the current free ones plus K)
------------------------------------------------------------------------- */

void FixBondFlux::move_break(int i, int k, double rsq_ik)
{
  nattempt += 1.0;
  nproposed += 1.0;
  const int nafter = find_candidates(i, atom->tag[k], CAND_FREE) + 1;

  double de = pair_delta(i, k, rsq_ik) - bond_energy(i, k, rsq_ik) - ebond;
  if (!accept(de, pform / (pbreak * nafter))) {
    nreject_energy += 1.0;
    return;
  }
  store[i][0] = 0.0;
  store[k][0] = 0.0;
  mark(i);
  mark(k);
  nbreak += 1.0;
  sum_de += de;
}

/* ----------------------------------------------------------------------
   apply partner changes of owned atoms to bond and 1-2 special lists
   a new bond is stored with the atom with the lower ID if newton_bond is set
------------------------------------------------------------------------- */

void FixBondFlux::reconcile_topology()
{
  tagint *tag = atom->tag;
  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  const int nlocal = atom->nlocal;
  const int newton_bond = force->newton_bond;
  const int maxspecial = atom->maxspecial;

  int changed = 0;
  int flag = 0;

  for (int i = 0; i < nlocal; i++) {
    const tagint po = porig[i];
    const tagint pf = (tagint) store[i][0];
    if (po == pf) continue;
    changed = 1;

    if (po) {
      for (int m = 0; m < num_bond[i]; m++) {
        if ((bond_type[i][m] == btype) && (bond_atom[i][m] == po)) {
          num_bond[i]--;
          bond_type[i][m] = bond_type[i][num_bond[i]];
          bond_atom[i][m] = bond_atom[i][num_bond[i]];
          break;
        }
      }
      for (int m = 0; m < nspecial[i][0]; m++) {
        if (special[i][m] == po) {
          for (int n = m; n < nspecial[i][2] - 1; n++) special[i][n] = special[i][n + 1];
          nspecial[i][0]--;
          nspecial[i][1]--;
          nspecial[i][2]--;
          break;
        }
      }
    }

    if (pf) {
      if (!newton_bond || (tag[i] < pf)) {
        if (num_bond[i] == atom->bond_per_atom) {
          flag |= 1;
        } else {
          bond_type[i][num_bond[i]] = btype;
          bond_atom[i][num_bond[i]] = pf;
          num_bond[i]++;
        }
      }
      int present = 0;
      for (int m = 0; m < nspecial[i][0]; m++)
        if (special[i][m] == pf) present = 1;
      if (!present) {
        if (nspecial[i][2] >= maxspecial) {
          flag |= 2;
        } else {
          for (int n = nspecial[i][2]; n > nspecial[i][0]; n--) special[i][n] = special[i][n - 1];
          special[i][nspecial[i][0]] = pf;
          nspecial[i][0]++;
          nspecial[i][1]++;
          nspecial[i][2]++;
        }
      }
    }
  }

  int flagall, changedall;
  MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_BOR, world);
  if (flagall & 1)
    error->all(FLERR, Error::NOLASTLINE,
               "New bond from fix {} exceeds the bonds per atom limit; "
               "use the extra/bond/per/atom keyword of read_data or create_box", style);
  if (flagall & 2)
    error->all(FLERR, Error::NOLASTLINE,
               "New bond from fix {} exceeds the special list size limit; "
               "use the extra/special/per/atom keyword of read_data or create_box", style);

  MPI_Allreduce(&changed, &changedall, 1, MPI_INT, MPI_MAX, world);
  if (changedall) next_reneighbor = update->ntimestep;
}

/* ----------------------------------------------------------------------
   bin all stickers (owned + ghost) with bins at least rc wide
------------------------------------------------------------------------- */

void FixBondFlux::build_bins()
{
  const int dim = domain->dimension;
  const int nall = atom->nlocal + atom->nghost;
  const double margin = MAX(comm->cutghost[0], MAX(comm->cutghost[1], comm->cutghost[2])) +
      neighbor->skin;
  int nb[3] = {1, 1, 1};

  for (int d = 0; d < 3; d++) {
    if (d < dim) {
      binlo[d] = domain->sublo[d] - margin;
      double extent = domain->subhi[d] - domain->sublo[d] + 2.0 * margin;
      nb[d] = MAX(1, static_cast<int>(extent / rc));
      bininv[d] = nb[d] / extent;
    } else {
      binlo[d] = domain->sublo[d];
      bininv[d] = 0.0;
    }
  }
  nbinx = nb[0];
  nbiny = nb[1];
  nbinz = nb[2];

  const bigint nbins = (bigint) nbinx * nbiny * nbinz;
  if (nbins > MAXSMALLINT)
    error->one(FLERR, Error::NOLASTLINE, "Too many bins for fix {} mc", style);
  if (nbins > maxbin) {
    maxbin = (int) nbins;
    memory->destroy(binhead);
    memory->create(binhead, maxbin, "bond/flux:binhead");
  }
  for (int b = 0; b < nbins; b++) binhead[b] = -1;

  int *mask = atom->mask;
  double **x = atom->x;
  for (int i = nall - 1; i >= 0; i--) {
    binnext[i] = -1;
    if (!(mask[i] & groupbit)) continue;
    int b = coord2bin(x[i]);
    binnext[i] = binhead[b];
    binhead[b] = i;
  }
}

/* ---------------------------------------------------------------------- */

int FixBondFlux::coord2bin(const double *xi)
{
  int ix = static_cast<int>(floor((xi[0] - binlo[0]) * bininv[0]));
  int iy = static_cast<int>(floor((xi[1] - binlo[1]) * bininv[1]));
  int iz = static_cast<int>(floor((xi[2] - binlo[2]) * bininv[2]));
  ix = MAX(0, MIN(nbinx - 1, ix));
  iy = MAX(0, MIN(nbiny - 1, iy));
  iz = MAX(0, MIN(nbinz - 1, iz));
  return (iz * nbiny + iy) * nbinx + ix;
}

/* ----------------------------------------------------------------------
   1 if atom ID T is a 1-2 neighbor of local atom I through a bond
   other than the dynamic bond I had at the start of the MC step
------------------------------------------------------------------------- */

int FixBondFlux::permanent_neighbor(int i, tagint t)
{
  if (t == porig[i]) return 0;
  tagint *slist = atom->special[i];
  const int n = atom->nspecial[i][0];
  for (int m = 0; m < n; m++)
    if (slist[m] == t) return 1;
  return 0;
}

/* ---------------------------------------------------------------------- */

int FixBondFlux::allowed_pair(int i, int j)
{
  if (!bondable[atom->type[i]][atom->type[j]]) return 0;
  if (molflag == INTER) {
    if (atom->molecule[i] == atom->molecule[j]) return 0;
  } else if (molflag == INTRA) {
    if (atom->molecule[i] != atom->molecule[j]) return 0;
  }
  return 1;
}

/* ---------------------------------------------------------------------- */

double FixBondFlux::distsq(int i, int j)
{
  double **x = atom->x;
  const double delx = x[i][0] - x[j][0];
  const double dely = x[i][1] - x[j][1];
  const double delz = x[i][2] - x[j][2];
  return delx * delx + dely * dely + delz * delz;
}

/* ---------------------------------------------------------------------- */

double FixBondFlux::bond_energy(int i, int j, double rsq)
{
  double fdummy;
  return force->bond->single(btype, rsq, i, j, fdummy);
}

/* ----------------------------------------------------------------------
   change of the pair energy when the I,J pair stops being a 1-2 neighbor
------------------------------------------------------------------------- */

double FixBondFlux::pair_delta(int i, int j, double rsq)
{
  if (!pairflag) return 0.0;
  const int itype = atom->type[i];
  const int jtype = atom->type[j];
  if (rsq >= force->pair->cutsq[itype][jtype]) return 0.0;
  double fdummy;
  return force->pair->single(i, j, itype, jtype, rsq, 1.0, 1.0, fdummy) -
      force->pair->single(i, j, itype, jtype, rsq, wcoul, wlj, fdummy);
}

/* ----------------------------------------------------------------------
   force on I (= del_IJ * returned value) from the part of the potential that
   depends on whether I,J is a dynamic bond: U_bond(r) - pair_delta(r)
------------------------------------------------------------------------- */

double FixBondFlux::topology_force(int i, int j, double rsq)
{
  double fbond, fpair;
  force->bond->single(btype, rsq, i, j, fbond);
  if (!pairflag) return fbond;
  const int itype = atom->type[i];
  const int jtype = atom->type[j];
  if (rsq >= force->pair->cutsq[itype][jtype]) return fbond;
  double f = fbond;
  force->pair->single(i, j, itype, jtype, rsq, wcoul, wlj, fpair);
  f += fpair;
  force->pair->single(i, j, itype, jtype, rsq, 1.0, 1.0, fpair);
  return f - fpair;
}

/* ----------------------------------------------------------------------
   the MC step falls between the two half-kicks of velocity Verlet, so the
   second half-kick of this step uses the forces of the new topology; adding
   (F_old - F_new) dt/2m to every atom whose partner changed makes the step
   end as without the moves, and the moves then change only the potential
   energy, by the Delta E of the Metropolis test (no heating by the switch)
------------------------------------------------------------------------- */

void FixBondFlux::correct_velocities()
{
  double **x = atom->x;
  double **v = atom->v;
  int *type = atom->type;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  const int nlocal = atom->nlocal;
  const double dtf = 0.5 * update->dt * force->ftm2v;

  for (int i = 0; i < nlocal; i++) {
    const tagint pold = porig[i];
    const tagint pnew = (tagint) store[i][0];
    if (pold == pnew) continue;

    double df[3] = {0.0, 0.0, 0.0};
    const tagint partner[2] = {pold, pnew};
    const double sign[2] = {1.0, -1.0};
    for (int m = 0; m < 2; m++) {
      if (partner[m] == 0) continue;
      int j = atom->map(partner[m]);
      if (j < 0) continue;
      j = domain->closest_image(i, j);
      const double del[3] = {x[i][0] - x[j][0], x[i][1] - x[j][1], x[i][2] - x[j][2]};
      const double rsq = del[0] * del[0] + del[1] * del[1] + del[2] * del[2];
      const double f = sign[m] * topology_force(i, j, rsq);
      for (int d = 0; d < 3; d++) df[d] += f * del[d];
    }
    const double dtfm = dtf / (atom->rmass_flag ? rmass[i] : mass[type[i]]);
    for (int d = 0; d < 3; d++) v[i][d] += dtfm * df[d];
  }
}

/* ----------------------------------------------------------------------
   number of dynamic bonds, from the partner IDs of owned atoms
------------------------------------------------------------------------- */

double FixBondFlux::count_bonds()
{
  double one = 0.0, all = 0.0;
  for (int i = 0; i < atom->nlocal; i++)
    if (store[i][0] != 0.0) one += 1.0;
  MPI_Allreduce(&one, &all, 1, MPI_DOUBLE, MPI_SUM, world);
  return 0.5 * all;
}

/* ----------------------------------------------------------------------
   set the three-body parameters of one pair of sticker types from V
     lj: eps sigma n rc lambda      sw: eps sigma B p q rc lambda
   rbond <= 0 means that a bond is any pair within the cutoff
   IFIRST = index of the first value in the fix arguments, for error messages
------------------------------------------------------------------------- */

void FixBondFlux::set_param3(Param3 &p, const double *v, double rbond, int ifirst)
{
  p.epsilon = v[0];
  p.sigma = v[1];
  p.npow = p.bigb = p.powp = p.powq = 0.0;
  p.norm = 1.0;
  p.phic = 0.0;

  if (form3 == BOND_LJ) {
    p.npow = v[2];
    p.rc = v[3];
    p.lambda = v[4];
    p.rmin = pow(2.0, 1.0 / p.npow) * p.sigma;
    if (p.rc <= p.rmin)
      error->all(FLERR, ifirst + 3,
                 "Fix {} 3body cutoff {} must be larger than the potential minimum {}", style, p.rc,
                 p.rmin);
    const double sr = pow(p.sigma / p.rc, p.npow);
    if (eshift) p.phic = 4.0 * (sr * sr - sr);

  } else {
    p.bigb = v[2];
    p.powp = v[3];
    p.powq = v[4];
    p.rc = v[5];
    p.lambda = v[6];

    // the minimum is the first zero of dV/dr below the cutoff, where the slope
    //   s(r) = (-p B (sigma/r)^p + q (sigma/r)^q) / r - sigma u(r) / (r - rc)^2,
    //   u(r) = B (sigma/r)^p - (sigma/r)^q, changes from negative to positive

    auto slope = [&p](double r) {
      const double sp = p.bigb * pow(p.sigma / r, p.powp);
      const double sq = pow(p.sigma / r, p.powq);
      const double rr = 1.0 / (r - p.rc);
      return (-p.powp * sp + p.powq * sq) / r - p.sigma * rr * rr * (sp - sq);
    };
    const int nscan = 4000;
    const double rlo = MIN(0.3 * p.sigma, 0.5 * p.rc);
    double ra = 0.0, rb = 0.0;
    double r0 = rlo, s0 = slope(rlo);
    for (int m = 1; m < nscan; m++) {
      const double r1 = rlo + (p.rc - rlo) * m / nscan;
      const double s1 = slope(r1);
      if ((s0 < 0.0) && (s1 >= 0.0)) {
        ra = r0;
        rb = r1;
        break;
      }
      r0 = r1;
      s0 = s1;
    }
    if (rb == 0.0)
      error->all(FLERR, ifirst, "Fix {} 3body sw bond has no minimum below the cutoff {}", style,
                 p.rc);
    for (int m = 0; m < 200; m++) {
      const double rm = 0.5 * (ra + rb);
      if (slope(rm) < 0.0)
        ra = rm;
      else
        rb = rm;
      if (rb - ra < 1.0e-15 * p.rc) break;
    }
    p.rmin = 0.5 * (ra + rb);
    const double u = p.bigb * pow(p.sigma / p.rmin, p.powp) - pow(p.sigma / p.rmin, p.powq);
    p.norm = -u * exp(p.sigma / (p.rmin - p.rc));
    if (p.norm <= 0.0)
      error->all(FLERR, ifirst, "Fix {} 3body sw bond has no attractive minimum below the cutoff {}",
                 style, p.rc);
  }

  p.rcsq = p.rc * p.rc;
  p.rminsq = p.rmin * p.rmin;
  const double rb = (rbond > 0.0) ? rbond : p.rc;
  p.rbondsq = rb * rb;
}

/* ----------------------------------------------------------------------
   two-body bonding potential of the three-body model, depth -epsilon at rmin
     lj: 2n-n Lennard-Jones form, shifted to zero at the cutoff (eshift yes)
     sw: Stillinger-Weber form eps [B (sigma/r)^p - (sigma/r)^q] exp(sigma/(r-rc)) / norm,
         which goes to zero smoothly at the cutoff
------------------------------------------------------------------------- */

void FixBondFlux::bond_potential(const Param3 &p, double r, double &v, double &dvdr)
{
  if (form3 == BOND_LJ) {
    const double sr = pow(p.sigma / r, p.npow);
    const double phi = 4.0 * (sr * sr - sr);
    const double dphi = -4.0 * p.npow * (2.0 * sr * sr - sr) / r;
    const double scale = p.epsilon / (1.0 + p.phic);
    v = scale * (phi - p.phic);
    dvdr = scale * dphi;
  } else {
    const double sp = p.bigb * pow(p.sigma / r, p.powp);
    const double sq = pow(p.sigma / r, p.powq);
    const double rr = 1.0 / (r - p.rc);
    const double scale = p.epsilon * exp(p.sigma * rr) / p.norm;
    v = scale * (sp - sq);
    dvdr = scale * ((-p.powp * sp + p.powq * sq) / r - p.sigma * rr * rr * (sp - sq));
  }
}

/* ----------------------------------------------------------------------
   two-body and three-body forces between stickers, as in pair style sw:
   full neighbor list, each pair computed once, triplets centered on owned atoms,
   forces on ghost atoms are summed to their owners by the reverse communication
   each pair of types has its own parameters; the three-body term of a triplet
   j-i-k has the strength min(lambda_ij eps_ij, lambda_ik eps_ik), which keeps
   a swap between bonds of different strength free of barriers for lambda = 1,
   unless the strength of the triplet is set with the triplet keyword
------------------------------------------------------------------------- */

void FixBondFlux::threebody_compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int *mask = atom->mask;
  const int nlocal = atom->nlocal;

  e2 = e3 = nbondpair = nmulti = 0.0;
  for (int i = 0; i < nlocal; i++) store[i][0] = store[i][1] = 0.0;

  const int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;
    const tagint itag = tag[i];
    const int itype = type[i];
    const double xtmp = x[i][0];
    const double ytmp = x[i][1];
    const double ztmp = x[i][2];
    int *jlist = firstneigh[i];
    const int jnum = numneigh[i];

    int numshort = 0;
    int ncoord = 0;
    tagint nearest = 0;
    double rnear = BIG;

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      if ((j >> SBBITS) & 3) continue;
      j &= NEIGHMASK;
      if (!(mask[j] & groupbit)) continue;
      if (!bondable[itype][type[j]]) continue;
      const Param3 &pij = param3[itype][type[j]];

      const double delx = xtmp - x[j][0];
      const double dely = ytmp - x[j][1];
      const double delz = ztmp - x[j][2];
      const double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq >= pij.rcsq) continue;

      if (numshort == maxshort) {
        maxshort += DELTA;
        memory->grow(neighshort, maxshort, "bond/flux:neighshort");
      }
      neighshort[numshort++] = j;

      if (rsq < pij.rbondsq) {
        ncoord++;
        if (rsq < rnear) {
          rnear = rsq;
          nearest = tag[j];
        }
      }

      // compute each pair only once, same convention as pair style sw

      const tagint jtag = tag[j];
      if (itag > jtag) {
        if ((itag + jtag) % 2 == 0) continue;
      } else if (itag < jtag) {
        if ((itag + jtag) % 2 == 1) continue;
      } else {
        if (x[j][2] < ztmp) continue;
        if ((x[j][2] == ztmp) && (x[j][1] < ytmp)) continue;
        if ((x[j][2] == ztmp) && (x[j][1] == ytmp) && (x[j][0] < xtmp)) continue;
      }

      const double r = sqrt(rsq);
      double v, dvdr;
      bond_potential(pij, r, v, dvdr);
      const double fpair = -dvdr / r;

      f[i][0] += delx * fpair;
      f[i][1] += dely * fpair;
      f[i][2] += delz * fpair;
      f[j][0] -= delx * fpair;
      f[j][1] -= dely * fpair;
      f[j][2] -= delz * fpair;

      e2 += v;
      if (vflag_global) {
        virial[0] += delx * delx * fpair;
        virial[1] += dely * dely * fpair;
        virial[2] += delz * delz * fpair;
        virial[3] += delx * dely * fpair;
        virial[4] += delx * delz * fpair;
        virial[5] += dely * delz * fpair;
      }
    }

    store[i][0] = (double) nearest;
    store[i][1] = (double) ncoord;
    nbondpair += ncoord;
    if (ncoord > 1) nmulti += 1.0;

    // three-body term: E = pre V3(r_ij) V3(r_ik) for all pairs j < k,
    //   with V3 of each leg from the parameters of its pair of types

    for (int jj = 0; jj < numshort - 1; jj++) {
      const int j = neighshort[jj];
      const Param3 &pij = param3[itype][type[j]];
      const double delr1[3] = {x[j][0] - xtmp, x[j][1] - ytmp, x[j][2] - ztmp};
      const double rsq1 = delr1[0] * delr1[0] + delr1[1] * delr1[1] + delr1[2] * delr1[2];
      const double r1 = sqrt(rsq1);
      double v3j = 1.0, dv3j = 0.0;
      if (rsq1 > pij.rminsq) {
        double v, dvdr;
        bond_potential(pij, r1, v, dvdr);
        v3j = -v / pij.epsilon;
        dv3j = -dvdr / pij.epsilon;
      }

      for (int kk = jj + 1; kk < numshort; kk++) {
        const int k = neighshort[kk];
        const Param3 &pik = param3[itype][type[k]];
        const double pre = strength3[itype][type[j]][type[k]];
        if (pre == 0.0) continue;
        const double delr2[3] = {x[k][0] - xtmp, x[k][1] - ytmp, x[k][2] - ztmp};
        const double rsq2 = delr2[0] * delr2[0] + delr2[1] * delr2[1] + delr2[2] * delr2[2];
        const double r2 = sqrt(rsq2);
        double v3k = 1.0, dv3k = 0.0;
        if (rsq2 > pik.rminsq) {
          double v, dvdr;
          bond_potential(pik, r2, v, dvdr);
          v3k = -v / pik.epsilon;
          dv3k = -dvdr / pik.epsilon;
        }

        e3 += pre * v3j * v3k;

        const double fjscale = -pre * v3k * dv3j / r1;
        const double fkscale = -pre * v3j * dv3k / r2;
        const double fj[3] = {fjscale * delr1[0], fjscale * delr1[1], fjscale * delr1[2]};
        const double fk[3] = {fkscale * delr2[0], fkscale * delr2[1], fkscale * delr2[2]};

        f[j][0] += fj[0];
        f[j][1] += fj[1];
        f[j][2] += fj[2];
        f[k][0] += fk[0];
        f[k][1] += fk[1];
        f[k][2] += fk[2];
        f[i][0] -= fj[0] + fk[0];
        f[i][1] -= fj[1] + fk[1];
        f[i][2] -= fj[2] + fk[2];

        if (vflag_global) {
          virial[0] += delr1[0] * fj[0] + delr2[0] * fk[0];
          virial[1] += delr1[1] * fj[1] + delr2[1] * fk[1];
          virial[2] += delr1[2] * fj[2] + delr2[2] * fk[2];
          virial[3] += delr1[0] * fj[1] + delr2[0] * fk[1];
          virial[4] += delr1[0] * fj[2] + delr2[0] * fk[2];
          virial[5] += delr1[1] * fj[2] + delr2[1] * fk[2];
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   mc:    association energy ebond * number of dynamic bonds
   3body: total energy of the sticker interactions
------------------------------------------------------------------------- */

double FixBondFlux::compute_scalar()
{
  if (mode == MC) return ebond * count_bonds();

  double one = e2 + e3;
  double all;
  MPI_Allreduce(&one, &all, 1, MPI_DOUBLE, MPI_SUM, world);
  return all;
}

/* ----------------------------------------------------------------------
   mc:    1 attempts, 2 proposals, 3 accepted shifts, 4 accepted swaps,
          5 accepted formations, 6 accepted breakings,
          7 rejected (bond too long or not allowed), 8 rejected (Metropolis test),
          9 sum of energy changes of accepted moves, 10 number of dynamic bonds,
          11 accepted moves in the last MC step,
          contact diagnostics (keyword contact, else 0):
          12 time in contact summed over stickers, 13 completed contact episodes,
          14 episodes missed by the MC steps, 15 accepted shifts, swaps and
          formations per unit contact time, 16 fraction of episodes missed
   3body: 1 two-body energy, 2 three-body energy, 3 bonded pairs,
          4 stickers with more than one bonded partner
------------------------------------------------------------------------- */

double FixBondFlux::compute_vector(int n)
{
  double one = 0.0, all = 0.0;

  if (mode == MC) {
    if (n >= 11) {
      double local[4] = {contact_time, nepisode, nmissed, nshift + nswap + nform};
      double total[4];
      MPI_Allreduce(local, total, 4, MPI_DOUBLE, MPI_SUM, world);
      if (n <= 13) return total[n - 11];
      if (n == 14) return (total[0] > 0.0) ? total[3] / total[0] : 0.0;
      return (total[1] > 0.0) ? total[2] / total[1] : 0.0;
    }
    if (n == 10) return (double) nlast;
    if (n == 9) return count_bonds();
    const double values[9] = {nattempt, nproposed,    nshift,         nswap, nform,
                              nbreak,   nreject_geom, nreject_energy, sum_de};
    one = values[n];
  } else {
    const double values[4] = {e2, e3, 0.5 * nbondpair, nmulti};
    one = values[n];
  }

  MPI_Allreduce(&one, &all, 1, MPI_DOUBLE, MPI_SUM, world);
  return all;
}

/* ---------------------------------------------------------------------- */

int FixBondFlux::pack_forward_comm(int n, int *sendlist, double *buf, int /*pbc_flag*/,
                                   int * /*pbc*/)
{
  int m = 0;
  if (commflag == FWD_PARTNER) {
    for (int i = 0; i < n; i++) buf[m++] = store[sendlist[i]][0];
  } else {
    int **nspecial = atom->nspecial;
    tagint **special = atom->special;
    for (int i = 0; i < n; i++) {
      const int j = sendlist[i];
      buf[m++] = store[j][0];
      const int ns = nspecial[j][0];
      buf[m++] = ubuf(ns).d;
      for (int k = 0; k < ns; k++) buf[m++] = ubuf(special[j][k]).d;
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixBondFlux::unpack_forward_comm(int n, int first, double *buf)
{
  int m = 0;
  const int last = first + n;
  if (commflag == FWD_PARTNER) {
    for (int i = first; i < last; i++) store[i][0] = buf[m++];
  } else {
    int **nspecial = atom->nspecial;
    tagint **special = atom->special;
    for (int i = first; i < last; i++) {
      store[i][0] = buf[m++];
      const int ns = (int) ubuf(buf[m++]).i;
      nspecial[i][0] = ns;
      for (int k = 0; k < ns; k++) special[i][k] = (tagint) ubuf(buf[m++]).i;
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixBondFlux::pack_reverse_comm(int n, int first, double *buf)
{
  int m = 0;
  const int last = first + n;
  if (commflag == REV_TOPO) {
    for (int i = first; i < last; i++) {
      buf[m++] = dcount[i];
      buf[m++] = store[i][0];
    }
  } else {
    for (int i = first; i < last; i++) {
      buf[m++] = (double) chflag[i];
      buf[m++] = store[i][0];
      buf[m++] = dcount[i];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixBondFlux::unpack_reverse_comm(int n, int *recvlist, double *buf)
{
  int m = 0;
  if (commflag == REV_TOPO) {
    for (int i = 0; i < n; i++) {
      const int j = recvlist[i];
      const double count = buf[m++];
      const double partner = buf[m++];
      if (count > 0.0) {
        dcount[j] += count;
        store[j][0] = partner;
      }
    }
  } else {

    // the receiving atom can itself be a ghost atom (multi-hop communication
    //   near sub-domain corners): keep the change flagged so the next hop forwards it

    const int nlocal = atom->nlocal;
    for (int i = 0; i < n; i++) {
      const int j = recvlist[i];
      const double changed = buf[m++];
      const double partner = buf[m++];
      const double count = buf[m++];
      if (changed != 0.0) {
        store[j][0] = partner;
        chflag[j] = 1;
      }
      if (j < nlocal)
        store[j][1] += count;
      else
        dcount[j] += count;
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixBondFlux::grow_arrays(int nmax)
{
  memory->grow(store, nmax, 2, "bond/flux:store");
  memory->grow(cstate, nmax, 2, "bond/flux:cstate");
  array_atom = store;
}

/* ---------------------------------------------------------------------- */

void FixBondFlux::copy_arrays(int i, int j, int /*delflag*/)
{
  store[j][0] = store[i][0];
  store[j][1] = store[i][1];
  cstate[j][0] = cstate[i][0];
  cstate[j][1] = cstate[i][1];
}

/* ---------------------------------------------------------------------- */

void FixBondFlux::set_arrays(int i)
{
  store[i][0] = store[i][1] = 0.0;
  cstate[i][0] = cstate[i][1] = 0.0;
}

/* ---------------------------------------------------------------------- */

int FixBondFlux::pack_exchange(int i, double *buf)
{
  buf[0] = store[i][0];
  buf[1] = store[i][1];
  buf[2] = cstate[i][0];
  buf[3] = cstate[i][1];
  return 4;
}

/* ---------------------------------------------------------------------- */

int FixBondFlux::unpack_exchange(int nlocal, double *buf)
{
  store[nlocal][0] = buf[0];
  store[nlocal][1] = buf[1];
  cstate[nlocal][0] = buf[2];
  cstate[nlocal][1] = buf[3];
  return 4;
}

/* ---------------------------------------------------------------------- */

void FixBondFlux::work_grow()
{
  if (atom->nmax > nmax_work) {
    nmax_work = atom->nmax;
    memory->destroy(porig);
    memory->destroy(chflag);
    memory->destroy(dcount);
    memory->destroy(sector);
    memory->destroy(binnext);
    memory->create(porig, nmax_work, "bond/flux:porig");
    memory->create(chflag, nmax_work, "bond/flux:chflag");
    memory->create(dcount, nmax_work, "bond/flux:dcount");
    memory->create(sector, nmax_work, "bond/flux:sector");
    memory->create(binnext, nmax_work, "bond/flux:binnext");
  }
}

/* ---------------------------------------------------------------------- */

double FixBondFlux::memory_usage()
{
  double bytes = 4.0 * atom->nmax * sizeof(double);
  bytes += (double) nmax_work * (sizeof(tagint) + 3 * sizeof(int) + sizeof(double));
  bytes += (double) maxbin * sizeof(int);
  bytes += (double) (maxorder + maxcand + maxshort) * sizeof(int);
  const double ntp1 = atom->ntypes + 1;
  bytes += ntp1 * ntp1 * sizeof(int);
  if (param3) bytes += ntp1 * ntp1 * sizeof(Param3);
  if (strength3) bytes += ntp1 * ntp1 * ntp1 * sizeof(double);
  return bytes;
}

#ifndef Nucleus_H
#define Nucleus_H

#include "base/Global.h"

#ifdef VECGEOM_NVCC
#include "base/Map.h"
#include "base/Vector.h"
#include <string.h>
#else
#include <map>
#include <vector>
#include <string>
#endif
#include <fstream>
#include <sstream>
#include <math.h>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class Nucleus;);

inline namespace VECGEOM_IMPL_NAMESPACE {

class Nucleus;

#ifdef VECGEOM_NVCC
using NucleusMap_t   = vecgeom::map<int, vecgeom::Vector<Nucleus *>>;
using NucleusIndex_t = vecgeom::map<int, Nucleus *>;
#else
using NucleusMap_t                = std::map<int, std::vector<Nucleus *>>;
using NucleusIndex_t              = std::map<int, Nucleus *>;
#endif

#ifdef VECGEOM_NVCC
extern VECCORE_ATT_DEVICE NucleusIndex_t *gNucleiDev;   // Nuclei list indexed by 10,000*z + 10*n + iso
extern NucleusIndex_t *gNucleiHost;                     // Nuclei list indexed by 10,000*z + 10*n + iso
extern VECCORE_ATT_DEVICE NucleusMap_t *gIsoListDev;    // List of isotopes for a given z
extern NucleusMap_t *gIsoListHost;                      // List of isotopes for a given z
extern VECCORE_ATT_DEVICE NucleusMap_t *gNatIsoListDev; // List of stable isotopes for a given z
extern NucleusMap_t *gNatIsoListHost;                   // List of stable isotopes for a given z
#endif

class Nucleus {
public:
  class Decay;

#ifdef VECGEOM_NVCC
  using VectorDecay_t = vecgeom::Vector<Decay>;
#else
  using VectorDecay_t             = std::vector<Decay>;
#endif

  VECCORE_ATT_HOST_DEVICE
  Nucleus(const char *name, int n, int z, int iso, double a, double dm, double life, double natab, double toxa,
          double toxb, int ind1, int ind2);

  VECCORE_ATT_HOST_DEVICE
  double A() const { return fA; }
  VECCORE_ATT_HOST_DEVICE
  double Z() const { return fZ; }
  VECCORE_ATT_HOST_DEVICE
  double Iso() const { return fIso; }
  VECCORE_ATT_HOST_DEVICE
  double Life() const { return fLife; }

  VECCORE_ATT_HOST_DEVICE
  double ToxA() const { return fToxa; }
  VECCORE_ATT_HOST_DEVICE
  double ToxB() const { return fToxb; }
  VECCORE_ATT_HOST_DEVICE
  int Indx1() const { return fInd1; }
  VECCORE_ATT_HOST_DEVICE
  int Indx2() const { return fInd2; }

  VECCORE_ATT_HOST_DEVICE
  static void CreateNuclei();

  VECCORE_ATT_HOST_DEVICE
  const VectorDecay_t &DecayList() const { return fDecayList; }

  std::string Name() const
  {
    char name[15];
    snprintf(name, 14, "%d-%s-%d-%d", fZ, fName, fN, fIso);
    name[14] = '\0';
    return std::string(name);
  }

#ifndef VECGEOM_NVCC
  static void ReadFile(std::string infilename, bool outfile = false);
#endif

  VECCORE_ATT_HOST_DEVICE
  void NormDecay();

  friend std::ostream &operator<<(std::ostream &os, const Nucleus &nuc);

  VECCORE_ATT_HOST_DEVICE
  void AddDecay(int da, int dz, int diso, double qval, double br);

  VECCORE_ATT_HOST_DEVICE
  static const NucleusIndex_t &Nuclei()
  {
#ifndef VECGEOM_NVCC
    if (!fNuclei) fNuclei = new NucleusIndex_t;
    return *fNuclei;
#else
#ifndef VECCORE_CUDA_DEVICE_COMPILATION
    if (!gNucleiHost) gNucleiHost = new NucleusIndex_t;
    return *gNucleiHost;
#else
    if (!gNucleiDev) gNucleiDev = new NucleusIndex_t;
    return *gNucleiDev;
#endif
#endif
  }

  VECCORE_ATT_HOST_DEVICE
  static const NucleusMap_t &IsoList()
  {
#ifndef VECGEOM_NVCC
    if (!fIsoList) fIsoList = new NucleusMap_t;
    return *fIsoList;
#else
#ifndef VECCORE_CUDA_DEVICE_COMPILATION
    if (!gIsoListHost) gIsoListHost = new NucleusMap_t;
    return *gIsoListHost;
#else
    if (!gIsoListDev) gIsoListDev = new NucleusMap_t;
    return *gIsoListDev;
#endif
#endif
  }

  VECCORE_ATT_HOST_DEVICE
  static const NucleusMap_t &NatIsoList()
  {
#ifndef VECGEOM_NVCC
    if (!fNatIsoList) fNatIsoList = new NucleusMap_t;
    return *fNatIsoList;
#else
#ifndef VECCORE_CUDA_DEVICE_COMPILATION
    return *gNatIsoListHost;
#else
    return *gNatIsoListDev;
#endif
#endif
  }

  class Decay {
  public:
    VECCORE_ATT_HOST_DEVICE
    Decay() : fDa(0), fDz(0), fDiso(0), fQval(0), fBr(0) {}

    VECCORE_ATT_HOST_DEVICE
    Decay(int da, int dz, int diso, double qval, double br) : fDa(da), fDz(dz), fDiso(diso), fQval(qval), fBr(br) {}

    VECCORE_ATT_HOST_DEVICE
    bool operator==(const Decay &d1) const
    {
      return (fDa == d1.fDa) && (fDz == d1.fDz) && (fDiso == d1.fDiso) &&
             (fabs(fQval - d1.fQval) / (fQval + d1.fQval) > 0 ? (fQval + d1.fQval) : 1 > 5e-7) &&
             (fabs(fBr - d1.fBr) / (fBr + d1.fBr) > 0 ? (fBr + d1.fBr) : 1 > 5e-7);
    }

    VECCORE_ATT_HOST_DEVICE
    int Dz() const { return fDz; }
    VECCORE_ATT_HOST_DEVICE
    int Da() const { return fDa; }
    VECCORE_ATT_HOST_DEVICE
    int Diso() const { return fDiso; }
    VECCORE_ATT_HOST_DEVICE
    double Qval() const { return fQval; }
    VECCORE_ATT_HOST_DEVICE
    double Br() const { return fBr; }
    VECCORE_ATT_HOST_DEVICE
    void Br(double br) { fBr = br; }
    const std::string Name() const;

  private:
    int fDa;
    int fDz;
    int fDiso;
    double fQval;
    double fBr;
  };

private:
#ifndef VECGEOM_NVCC
  static void Getmat(std::string line, int &n, int &z, int &iso, std::string &name, double &a, double &dm, double &life,
                     int &da, int &dz, int &diso, double &br, double &qval, double &natab, double &toxa, double &toxb,
                     int &ind1, int &ind2);
#endif

  char fName[50];           // Name
  int fN;                   // Nucleon number
  int fZ;                   // Atomic number
  int fIso;                 // Isomer level
  double fA;                // Atomic weight
  int fIsolevel;            // Isomeric mass excess
  double fLife;             // Lifetime
  double fNatab;            // Natural abundance
  double fToxa;             // Radiotoxicity
  double fToxb;             // Radiotoxicity
  int fInd1;                // Misterious index 1
  int fInd2;                // Misterious index 2
  VectorDecay_t fDecayList; // Decay channels

#ifndef VECGEOM_NVCC
  static NucleusIndex_t *fNuclei;   // Nuclei list indexed by 10,000*z + 10*n + iso
  static NucleusMap_t *fIsoList;    // List of isotopes for a given z
  static NucleusMap_t *fNatIsoList; // List of stable isotopes for a given z
#endif
};
}
}
#endif

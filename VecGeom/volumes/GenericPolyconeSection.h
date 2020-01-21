/// @file GenericPolyconeSection.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_GENERICPOLYCONESECTION_H_
#define VECGEOM_GENERICPOLYCONESECTION_H_

#include "VecGeom/volumes/ConeStruct.h"
#include "VecGeom/volumes/CoaxialConesStruct.h"

namespace vecgeom {

// helper structure to encapsulate a section
VECGEOM_DEVICE_FORWARD_DECLARE(struct GenericPolyconeSection;);
VECGEOM_DEVICE_DECLARE_CONV(struct, GenericPolyconeSection);

inline namespace VECGEOM_IMPL_NAMESPACE {

struct GenericPolyconeSection {
  VECCORE_ATT_HOST_DEVICE
  GenericPolyconeSection() : fCoaxialCones(0), fShift(0.0), fTubular(0), fConvex(0) {}

  VECCORE_ATT_HOST_DEVICE
  ~GenericPolyconeSection() {}

  CoaxialConesStruct<Precision> *fCoaxialCones;
  double fShift;
  bool fTubular;
  bool fConvex; // TRUE if all points in section are concave in regards to whole polycone, will be determined
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif

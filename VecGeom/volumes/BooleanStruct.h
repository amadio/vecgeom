#ifndef VECGEOM_VOLUMES_BOOLEANSTRUCT_H_
#define VECGEOM_VOLUMES_BOOLEANSTRUCT_H_
#include "VecGeom/base/Global.h"

namespace vecgeom {

// Declare types shared by cxx and cuda.
enum BooleanOperation { kUnion, kIntersection, kSubtraction };

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * A class representing a simple UNPLACED boolean volume A-B
 * It takes two template arguments:
 * 1.: the mother (or left) volume A in unplaced form
 * 2.: the (or right) volume B in placed form, acting on A with a boolean operation;
 * the placement is with respect to the left volume
 */
struct BooleanStruct {
  VPlacedVolume const *fLeftVolume;
  VPlacedVolume const *fRightVolume;
  BooleanOperation const fOp;
  mutable double fCapacity    = -1;
  mutable double fSurfaceArea = -1;

  VECCORE_ATT_HOST_DEVICE
  BooleanStruct(BooleanOperation op, VPlacedVolume const *left, VPlacedVolume const *right)
      : fLeftVolume(left), fRightVolume(right), fOp(op)
  {
  }
}; // End struct

} // End impl namespace

} // End global namespace

#endif /* VECGEOM_VOLUMES_BOOLEANSTRUCT_H_ */

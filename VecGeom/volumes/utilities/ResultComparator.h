/*
 * ResultComparator.h
 *
 *  Created on: Nov 27, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_RESULTCOMPARATOR_H_
#define VECGEOM_RESULTCOMPARATOR_H_

#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/base/Global.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

// utility functions to compare distance results against ROOT/Geant4/ etc.
// they are mainly used by VecGeom shapes to detect discrepances to other libraries on the fly
namespace DistanceComparator {

void CompareUnplacedContains(VPlacedVolume const *vol, bool vecgeomresult, Vector3D<Precision> const &point);
void PrintPointInformation(VPlacedVolume const *vol, Vector3D<Precision> const &point);
void CompareDistanceToIn(VPlacedVolume const *vol, Precision vecgeomresult, Vector3D<Precision> const &point,
                         Vector3D<Precision> const &direction, Precision const stepMax = VECGEOM_NAMESPACE::kInfLength);
void CompareDistanceToOut(VPlacedVolume const *vol, Precision vecgeomresult, Vector3D<Precision> const &point,
                          Vector3D<Precision> const &direction,
                          Precision const stepMax = VECGEOM_NAMESPACE::kInfLength);

} // end inner namespace
}
} // end namespace

#endif // VECGEOM_RESULTCOMPARATOR_H_

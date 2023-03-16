/// @file SpecializedCoaxialCones.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDCOAXIALCONES_H_
#define VECGEOM_VOLUMES_SPECIALIZEDCOAXIALCONES_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/CoaxialConesImplementation.h"
#include "VecGeom/volumes/PlacedCoaxialCones.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedCoaxialCones.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

using SpecializedCoaxialCones = SpecializedVolImplHelper<CoaxialConesImplementation>;

using SimpleCoaxialCones = SpecializedCoaxialCones;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDCOAXIALCONES_H_

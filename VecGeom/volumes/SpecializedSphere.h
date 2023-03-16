/// @file SpecializedSphere.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDSPHERE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDSPHERE_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/kernel/SphereImplementation.h"
#include "VecGeom/volumes/PlacedSphere.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedSphere.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

using SpecializedSphere = SpecializedVolImplHelper<SphereImplementation>;
using SimpleSphere      = SpecializedSphere;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDSPHERE_H_

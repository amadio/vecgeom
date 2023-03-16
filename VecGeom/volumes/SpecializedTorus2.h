/// \file SpecializedTorus2.h

#ifndef VECGEOM_VOLUMES_SPECIALIZEDTORUS2_H_
#define VECGEOM_VOLUMES_SPECIALIZEDTORUS2_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/TorusImplementation2.h"
#include "VecGeom/volumes/PlacedTorus2.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

// NOTE: we may want to specialize the torus like we do for the tube
// at the moment this is not done

using SpecializedTorus2 = SpecializedVolImplHelper<TorusImplementation2>;

using SimpleTorus2 = SpecializedTorus2;

} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDTORUS2_H_

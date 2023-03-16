/// @file SpecializedMultiUnion.h
/// @author mihaela.gheata@cern.ch

#ifndef VECGEOM_VOLUMES_SPECIALIZEDMULTIUNION_H_
#define VECGEOM_VOLUMES_SPECIALIZEDMULTIUNION_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/MultiUnionImplementation.h"
#include "VecGeom/volumes/PlacedMultiUnion.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

using SpecializedMultiUnion = SpecializedVolImplHelper<MultiUnionImplementation>;

using SimpleMultiUnion = SpecializedMultiUnion;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDMULTIUNION_H_

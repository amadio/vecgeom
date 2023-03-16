#ifndef VECGEOM_VOLUMES_SPECIALIZEDBOOLEAN_H
#define VECGEOM_VOLUMES_SPECIALIZEDBOOLEAN_H

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/BooleanImplementation.h"
#include "VecGeom/volumes/UnplacedBooleanVolume.h"
#include "VecGeom/volumes/PlacedBooleanVolume.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

template <BooleanOperation boolOp>
using SpecializedBooleanVolume = SpecializedVolImplHelper<BooleanImplementation<boolOp>>;

using GenericPlacedUnionVolume        = SpecializedBooleanVolume<kUnion>;
using GenericPlacedIntersectionVolume = SpecializedBooleanVolume<kIntersection>;
using GenericPlacedSubtractionVolume  = SpecializedBooleanVolume<kSubtraction>;

using GenericUnionVolume        = SpecializedBooleanVolume<kUnion>;
using GenericIntersectionVolume = SpecializedBooleanVolume<kIntersection>;
using GenericSubtractionVolume  = SpecializedBooleanVolume<kSubtraction>;

} // namespace VECGEOM_IMPL_NAMESPACE

} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDBOOLEAN_H

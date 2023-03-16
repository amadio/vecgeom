/// @file SpecializedTessellated.h
/// @author mihaela.gheata@cern.ch

#ifndef VECGEOM_VOLUMES_SPECIALIZEDTESSELLATED_H_
#define VECGEOM_VOLUMES_SPECIALIZEDTESSELLATED_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/TessellatedImplementation.h"
#include "VecGeom/volumes/PlacedTessellated.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

using SpecializedTessellated = SpecializedVolImplHelper<TessellatedImplementation>;

using SimpleTessellated = SpecializedTessellated;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDTESSELLATED_H_

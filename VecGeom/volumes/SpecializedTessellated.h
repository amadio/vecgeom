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

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedTessellated = SpecializedVolImplHelper<TessellatedImplementation, transCodeT, rotCodeT>;

using SimpleTessellated = SpecializedTessellated<translation::kGeneric, rotation::kGeneric>;
}
} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDTESSELLATED_H_

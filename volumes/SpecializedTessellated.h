/// @file SpecializedTessellated.h
/// @author mihaela.gheata@cern.ch

#ifndef VECGEOM_VOLUMES_SPECIALIZEDTESSELLATED_H_
#define VECGEOM_VOLUMES_SPECIALIZEDTESSELLATED_H_

#include "base/Global.h"

#include "volumes/kernel/TessellatedImplementation.h"
#include "volumes/PlacedTessellated.h"
#include "volumes/SpecializedPlacedVolImplHelper.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedTessellated = LoopSpecializedVolImplHelper<TessellatedImplementation, transCodeT, rotCodeT>;

using SimpleTessellated = SpecializedTessellated<translation::kGeneric, rotation::kGeneric>;
}
} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDTESSELLATED_H_

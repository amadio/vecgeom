/// @file SpecializedExtruded.h
/// @author mihaela.gheata@cern.ch

#ifndef VECGEOM_VOLUMES_SPECIALIZEDEXTRUDED_H_
#define VECGEOM_VOLUMES_SPECIALIZEDEXTRUDED_H_

#include "base/Global.h"

#include "volumes/kernel/TessellatedImplementation.h"
#include "volumes/PlacedExtruded.h"
#include "volumes/SpecializedPlacedVolImplHelper.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedExtruded = LoopSpecializedVolImplHelper<TessellatedImplementation, transCodeT, rotCodeT>;

using SimpleExtruded = SpecializedExtruded<translation::kGeneric, rotation::kGeneric>;
}
} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDEXTRUDED_H_

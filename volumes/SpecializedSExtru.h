#ifndef VECGEOM_VOLUMES_SPECIALIZEDPLACEDSEXTRU_H_
#define VECGEOM_VOLUMES_SPECIALIZEDPLACEDSEXTRU_H_

#include "base/Global.h"

#include "volumes/kernel/SExtruImplementation.h"
#include "volumes/PlacedSExtru.h"
#include "volumes/SpecializedPlacedVolImplHelper.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedSExtru = LoopSpecializedVolImplHelper<SExtruImplementation, transCodeT, rotCodeT>;

using SimpleSExtru = SpecializedSExtru<translation::kGeneric, rotation::kGeneric>;
}
} // End global namespace

#endif

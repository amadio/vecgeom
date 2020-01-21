#ifndef VECGEOM_VOLUMES_SPECIALIZEDPLACEDSEXTRU_H_
#define VECGEOM_VOLUMES_SPECIALIZEDPLACEDSEXTRU_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/SExtruImplementation.h"
#include "VecGeom/volumes/PlacedSExtru.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedSExtru = LoopSpecializedVolImplHelper<SExtruImplementation, transCodeT, rotCodeT>;

using SimpleSExtru = SpecializedSExtru<translation::kGeneric, rotation::kGeneric>;
}
} // End global namespace

#endif

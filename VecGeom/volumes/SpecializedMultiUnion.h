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

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedMultiUnion = SpecializedVolImplHelper<MultiUnionImplementation, transCodeT, rotCodeT>;

using SimpleMultiUnion = SpecializedMultiUnion<translation::kGeneric, rotation::kGeneric>;
}
} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDMULTIUNION_H_

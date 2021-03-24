/// @file SpecializedExtruded.h
/// @author mihaela.gheata@cern.ch

#ifndef VECGEOM_VOLUMES_SPECIALIZEDEXTRUDED_H_
#define VECGEOM_VOLUMES_SPECIALIZEDEXTRUDED_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/ExtrudedImplementation.h"
#include "VecGeom/volumes/PlacedExtruded.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedExtruded = SpecializedVolImplHelper<ExtrudedImplementation, transCodeT, rotCodeT>;

using SimpleExtruded = SpecializedExtruded<translation::kGeneric, rotation::kGeneric>;
}
} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDEXTRUDED_H_

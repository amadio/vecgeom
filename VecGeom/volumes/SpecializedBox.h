#ifndef VECGEOM_VOLUMES_SPECIALIZEDBOX_H_
#define VECGEOM_VOLUMES_SPECIALIZEDBOX_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/BoxImplementation.h"
#include "VecGeom/volumes/PlacedBox.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedBox.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedBox = SpecializedVolImplHelper<BoxImplementation, transCodeT, rotCodeT>;

using SimpleBox = SpecializedBox<translation::kGeneric, rotation::kGeneric>;
}
} // End global namespace

#endif

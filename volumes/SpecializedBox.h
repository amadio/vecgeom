#ifndef VECGEOM_VOLUMES_SPECIALIZEDBOX_H_
#define VECGEOM_VOLUMES_SPECIALIZEDBOX_H_

#include "base/Global.h"

#include "volumes/kernel/BoxImplementation.h"
#include "volumes/PlacedBox.h"
#include "volumes/SpecializedPlacedVolImplHelper.h"
#include "volumes/UnplacedBox.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedBox = SIMDSpecializedVolImplHelper<BoxImplementation, transCodeT, rotCodeT>;

using SimpleBox = SpecializedBox<translation::kGeneric, rotation::kGeneric>;
}
} // End global namespace

#endif

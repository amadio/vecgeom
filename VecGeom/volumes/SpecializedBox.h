#ifndef VECGEOM_VOLUMES_SPECIALIZEDBOX_H_
#define VECGEOM_VOLUMES_SPECIALIZEDBOX_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/BoxImplementation.h"
#include "VecGeom/volumes/PlacedBox.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedBox.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

using SpecializedBox = SpecializedVolImplHelper<BoxImplementation>;

using SimpleBox = SpecializedBox;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif

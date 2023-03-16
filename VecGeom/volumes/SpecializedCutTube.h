#ifndef VECGEOM_VOLUMES_SPECIALIZEDCUTTUBE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDCUTTUBE_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/CutTubeImplementation.h"
#include "VecGeom/volumes/PlacedCutTube.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedCutTube.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

using SpecializedCutTube = SpecializedVolImplHelper<CutTubeImplementation>;

using SimpleCutTube = SpecializedCutTube;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif

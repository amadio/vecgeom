#ifndef VECGEOM_VOLUMES_SPECIALIZEDCUTTUBE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDCUTTUBE_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/CutTubeImplementation.h"
#include "VecGeom/volumes/PlacedCutTube.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedCutTube.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedCutTube = SIMDSpecializedVolImplHelper<CutTubeImplementation, transCodeT, rotCodeT>;

using SimpleCutTube = SpecializedCutTube<translation::kGeneric, rotation::kGeneric>;
}
} // End global namespace

#endif

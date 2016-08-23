/// \file SpecializedScaledShape.h
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDSCALEDSHAPE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDSCALEDSHAPE_H_

#include "base/Global.h"
#include "volumes/kernel/ScaledShapeImplementation.h"
#include "volumes/PlacedScaledShape.h"
#include "volumes/UnplacedScaledShape.h"
#include "volumes/SpecializedPlacedVolImplHelper.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedScaledShape = LoopSpecializedVolImplHelper<ScaledShapeImplementation, transCodeT, rotCodeT>;
// SIMD interface below to be enabled when available in UnplacedVolume
// using SpecializedScaledShape = SIMDSpecializedVolImplHelper<ScaledShapeImplementation, transCodeT, rotCodeT>;

using SimpleScaledShape = SpecializedScaledShape<translation::kGeneric, rotation::kGeneric>;
}
} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDSCALEDSHAPE_H_

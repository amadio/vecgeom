/// @file SpecializedEllipticalTube.h
/// @author Raman Sehgal (raman.sehgal@cern.ch), Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDELLIPTICALTUBE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDELLIPTICALTUBE_H_

#include "base/Global.h"

#include "volumes/kernel/EllipticalTubeImplementation.h"
#include "volumes/PlacedEllipticalTube.h"
#include "volumes/SpecializedPlacedVolImplHelper.h"
#include "volumes/UnplacedEllipticalTube.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedEllipticalTube = SIMDSpecializedVolImplHelper<EllipticalTubeImplementation, transCodeT, rotCodeT>;

using SimpleEllipticalTube = SpecializedEllipticalTube<translation::kGeneric, rotation::kGeneric>;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDELLIPTICALTUBE_H_

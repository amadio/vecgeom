// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of the specialized elliptical tube volume
/// @file volumes/SpecializedEllipticalTube.h
/// @author Raman Sehgal, Evgueni Tcherniaev

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

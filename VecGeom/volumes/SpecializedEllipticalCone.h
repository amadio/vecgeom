// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of the specialized elliptical cone volume.
/// @file volumes/SpecializedEllipticalCone.h
/// @author Raman Sehgal, Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_SPECIALIZEDELLIPTICALCONE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDELLIPTICALCONE_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/EllipticalConeImplementation.h"
#include "VecGeom/volumes/PlacedEllipticalCone.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedEllipticalCone.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedEllipticalCone = SIMDSpecializedVolImplHelper<EllipticalConeImplementation, transCodeT, rotCodeT>;

using SimpleEllipticalCone = SpecializedEllipticalCone<translation::kGeneric, rotation::kGeneric>;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDELLIPTICALCONE_H_

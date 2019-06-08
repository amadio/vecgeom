// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Includes all headers related to the EllipticalCone volume
/// @file volumes/SpecializedEllipticalCone.h
/// @author Raman Sehgal (raman.sehgal@cern.ch), Evgueni Tcherniaev (evgueni.tcherniaev@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDELLIPTICALCONE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDELLIPTICALCONE_H_

#include "base/Global.h"

#include "volumes/kernel/EllipticalConeImplementation.h"
#include "volumes/PlacedEllipticalCone.h"
#include "volumes/SpecializedPlacedVolImplHelper.h"
#include "volumes/UnplacedEllipticalCone.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedEllipticalCone = SIMDSpecializedVolImplHelper<EllipticalConeImplementation, transCodeT, rotCodeT>;

using SimpleEllipticalCone = SpecializedEllipticalCone<translation::kGeneric, rotation::kGeneric>;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDELLIPTICALCONE_H_

// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// \brief Declaration of the Specialized Orb volume
/// \file volumes/SpecializedOrb.h
/// \author Raman Sehgal

#ifndef VECGEOM_VOLUMES_SPECIALIZEDORB_H_
#define VECGEOM_VOLUMES_SPECIALIZEDORB_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/OrbImplementation.h"
#include "VecGeom/volumes/PlacedOrb.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedOrb.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedOrb = SpecializedVolImplHelper<OrbImplementation, transCodeT, rotCodeT>;

using SimpleOrb = SpecializedOrb<translation::kGeneric, rotation::kGeneric>;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDORB_H_

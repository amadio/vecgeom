// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of the specialized elliptical tube volume
/// @file volumes/SpecializedEllipticalTube.h
/// @author Raman Sehgal, Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_SPECIALIZEDELLIPTICALTUBE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDELLIPTICALTUBE_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/EllipticalTubeImplementation.h"
#include "VecGeom/volumes/PlacedEllipticalTube.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedEllipticalTube.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

using SpecializedEllipticalTube = SpecializedVolImplHelper<EllipticalTubeImplementation>;

using SimpleEllipticalTube = SpecializedEllipticalTube;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDELLIPTICALTUBE_H_

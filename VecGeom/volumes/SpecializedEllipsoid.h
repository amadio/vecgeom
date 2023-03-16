// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of the specialized ellipsoid volume
/// @file volumes/SpecializedEllipsoid.h
/// @author Evgueni Tcherniaev

#ifndef VECGEOM_VOLUMES_SPECIALIZEDELLIPSOID_H_
#define VECGEOM_VOLUMES_SPECIALIZEDELLIPSOID_H_

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/EllipsoidImplementation.h"
#include "VecGeom/volumes/PlacedEllipsoid.h"
#include "VecGeom/volumes/SpecializedPlacedVolImplHelper.h"
#include "VecGeom/volumes/UnplacedEllipsoid.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

using SpecializedEllipsoid = SpecializedVolImplHelper<EllipsoidImplementation>;

using SimpleEllipsoid = SpecializedEllipsoid;
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPECIALIZEDELLIPSOID_H_

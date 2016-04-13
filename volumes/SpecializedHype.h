/// @file SpecializedHype.h
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_SPECIALIZEDHYPE_H_
#define VECGEOM_VOLUMES_SPECIALIZEDHYPE_H_

#include "base/Global.h"
#include "backend/Backend.h"
#include "volumes/kernel/HypeImplementation.h"
#include "volumes/PlacedHype.h"
#include "volumes/ShapeImplementationHelper.h"

#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedHype = ShapeImplementationHelper<HypeImplementation<transCodeT, rotCodeT>>;
using SimpleHype = SpecializedHype<translation::kGeneric, rotation::kGeneric>;
}
} // End global namespace

#endif // VECGEOM_VOLUMES_SPECIALIZEDHYPE_H_

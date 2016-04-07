/*
 * SpecializedGenTrap.h
 *
 *  Created on: Aug 3, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_SPECIALIZEDGENTRAP_H_
#define VECGEOM_VOLUMES_SPECIALIZEDGENTRAP_H_


#include "base/Global.h"
#include "volumes/kernel/GenTrapImplementation.h"
#include "volumes/PlacedGenTrap.h"
#include "volumes/ShapeImplementationHelper.h"

#include <stdio.h>

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

template <TranslationCode transCodeT, RotationCode rotCodeT>
using SpecializedGenTrap = ShapeImplementationHelper<GenTrapImplementation<transCodeT, rotCodeT> >;

using SimpleGenTrap = SpecializedGenTrap<translation::kGeneric, rotation::kGeneric>;

} } // End global namespace

/*
class SpecializedGenTrap
    : public ShapeImplementationHelper<GenTrapImplementation<transCodeT, rotCodeT> > {

  typedef ShapeImplementationHelper<PlacedGenTrap,
                                    GenTrapImplementation<
                                        transCodeT, rotCodeT> > Helper;

public:

#ifndef VECGEOM_NVCC

  SpecializedGenTrap(char const *const label,
                 LogicalVolume const *const logical_volume,
                 Transformation3D const *const transformation)
      : Helper(label, logical_volume, transformation, NULL) {}

  SpecializedGenTrap(LogicalVolume const *const logical_volume,
                 Transformation3D const *const transformation)
      : SpecializedGenTrap("", logical_volume, transformation) {}

  virtual ~SpecializedGenTrap() {}

  // SpecializedGenTrap(char const *const label,
  //                Vector3D<Precision> const &dim)
  //     : SpecializedGenTrap(label, new LogicalVolume(new UnplacedGenTrap(dim)),
  //                      Transformation3D::kIdentity) {}

  // A standard USolid like constructor
  //SpecializedGenTrap(char const *const label,
  //               const Precision dX, const Precision dY, const Precision dZ)
  //    : SpecializedGenTrap(label, new LogicalVolume(new UnplacedGenTrap(dX, dY, dZ)),
  //                     &Transformation3D::kIdentity) {}

#else

  __device__
  SpecializedGenTrap(LogicalVolume const *const logical_volume,
                 Transformation3D const *const transformation,
                 const int id)
      : Helper(logical_volume, transformation, this, id) {}

#endif

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const;

};

typedef SpecializedGenTrap<translation::kGeneric, rotation::kGeneric> SimpleGenTrap;

template <TranslationCode transCodeT, RotationCode rotCodeT>
void SpecializedGenTrap<transCodeT, rotCodeT>::PrintType() const {
  printf("SpecializedGenTrap<%i, %i>", transCodeT, rotCodeT);
}

} } // End global namespace
*/
#endif /* VECGEOM_VOLUMES_SPECIALIZEDGENTRAP_H_ */

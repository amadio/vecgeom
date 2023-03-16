/// \file VolumeFactory.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_MANAGEMENT_VOLUMEFACTORY_H_
#define VECGEOM_MANAGEMENT_VOLUMEFACTORY_H_

#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/volumes/LogicalVolume.h"

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

class VolumeFactory {

public:
  static VolumeFactory &Instance()
  {
    static VolumeFactory instance;
    return instance;
  }

  // takes an existing placed volume and creates new one (changes its type) to a given Helper and ImplKernel
  // but keeps the transformation specialization properties
  // (used only internally)
  template <template <typename> class SpecializedHelper, typename ImplKernel>
  static VPlacedVolume *ChangeTypeKeepTransformation(VPlacedVolume const *);

#ifndef VECCORE_CUDA

  template <typename VolumeType>
  static VPlacedVolume *CreateByTransformation(LogicalVolume const *const logical_volume,
                                               Transformation3D const *const transformation,
                                               VPlacedVolume *const placement = NULL);

#else

  template <typename VolumeType>
  VECCORE_ATT_DEVICE static VPlacedVolume *CreateByTransformation(LogicalVolume const *const logical_volume,
                                                                  Transformation3D const *const transformation,
                                                                  const int id, const int copy_no, const int child_id,
                                                                  VPlacedVolume *const placement = NULL);

#endif

private:
  VolumeFactory() {}
  VolumeFactory(VolumeFactory const &);
  VolumeFactory &operator=(VolumeFactory const &);
};

template <typename VolumeType>
VECCORE_ATT_DEVICE VPlacedVolume *VolumeFactory::CreateByTransformation(LogicalVolume const *const logical_volume,
                                                                        Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                                                                        const int id, const int copy_no,
                                                                        const int child_id,
#endif
                                                                        VPlacedVolume *const placement)
{
  return VolumeType::Create(logical_volume, transformation,
#ifdef VECCORE_CUDA
                            id, copy_no, child_id,
#endif
                            placement);
}

template <template <typename> class SpecializedHelper, typename ImplKernel>
VPlacedVolume *VolumeFactory::ChangeTypeKeepTransformation(VPlacedVolume const *pvol)
{
  return new SpecializedHelper<ImplKernel>(pvol);
}
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_MANAGEMENT_VOLUMEFACTORY_H_

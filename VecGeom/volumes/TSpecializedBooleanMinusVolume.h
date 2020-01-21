#ifndef VECGEOM_VOLUMES_TSPECIALIZEDBOOLEANMINUS_H
#define VECGEOM_VOLUMES_TSPECIALIZEDBOOLEANMINUS_H

#include "VecGeom/base/Global.h"

#include "VecGeom/volumes/kernel/TBooleanMinusImplementation.h"
#include "VecGeom/volumes/TUnplacedBooleanMinusVolume.h"
#include "VecGeom/volumes/TPlacedBooleanMinusVolume.h"
#include "VecGeom/volumes/ShapeImplementationHelper.h"

namespace VECGEOM_NAMESPACE {

template <typename LeftUnplacedVolume_t, typename RightPlacedVolume_t, TranslationCode transCodeT,
          RotationCode rotCodeT>
class TSpecializedBooleanMinusVolume
    : public ShapeImplementationHelper<
          TPlacedBooleanMinusVolume,
          TBooleanMinusImplementation<LeftUnplacedVolume_t, RightPlacedVolume_t, transCodeT, rotCodeT>> {

  typedef ShapeImplementationHelper<
      TPlacedBooleanMinusVolume,
      TBooleanMinusImplementation<LeftUnplacedVolume_t, RightPlacedVolume_t, transCodeT, rotCodeT>>
      Helper;

  // typedef TUnplacedBooleanMinusVolume<LeftUnplacedVolume_t, RightPlacedVolume_t> UnplacedVol_t;
  typedef TUnplacedBooleanMinusVolume UnplacedVol_t;

public:
#ifndef VECCORE_CUDA

  TSpecializedBooleanMinusVolume(char const *const label, LogicalVolume const *const logical_volume,
                                 Transformation3D const *const transformation)
      : Helper(label, logical_volume, transformation, NULL)
  {
  }

  TSpecializedBooleanMinusVolume(LogicalVolume const *const logical_volume,
                                 Transformation3D const *const transformation)
      : TSpecializedBooleanMinusVolume("", logical_volume, transformation)
  {
  }

  virtual ~TSpecializedBooleanMinusVolume() {}

#else

  VECCORE_ATT_DEVICE TSpecializedBooleanMinusVolume(LogicalVolume const *const logical_volume,
                                                    Transformation3D const *const transformation,
                                                    PlacedBox const *const boundingBox, const int id)
      : Helper(logical_volume, transformation, boundingBox, id)
  {
  }

#endif

  virtual int MemorySize() const { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void PrintType() const { printf("NOT IMPLEMENTED"); };

}; // endofclassdefinition

} // End global namespace

#endif // VECGEOM_VOLUMES_TSPECIALIZEDBOOLEANMINUS_H

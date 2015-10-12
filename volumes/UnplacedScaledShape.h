/// @file UnplacedScaledShape.h
/// @author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDSCALEDSHAPE_H_
#define VECGEOM_VOLUMES_UNPLACEDSCALEDSHAPE_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "base/Vector3D.h"
#include "base/Scale3D.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/PlacedVolume.h"
#include "volumes/LogicalVolume.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class UnplacedScaledShape; )
VECGEOM_DEVICE_DECLARE_CONV( UnplacedScaledShape )

inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedScaledShape : public VUnplacedVolume, public AlignedBase {

public:

  VPlacedVolume const *fPlaced;  /// Need a placed volue for the navigation interface
  Scale3D              fScale;   /// Scale object

private:

/// Copy constructor - not to use
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedScaledShape(UnplacedScaledShape const &other) : fPlaced(other.fPlaced), fScale(other.fScale) { }

/// Assignment operator - not to use
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedScaledShape &operator=(UnplacedScaledShape const &other) {
    if (&other != this) {
      fPlaced = other.fPlaced;
      fScale = other.fScale;
    }
    return *this;
  }    

public:

/// Dummy constructor
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedScaledShape() : fPlaced(0), fScale() { }

/// Constructor
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedScaledShape(VUnplacedVolume const *shape, Precision sx, Precision sy, Precision sz) : fPlaced(0), fScale(sx,sy,sz) {
    // We need to create a placement with identity transformation from the unplaced version
    // Hopefully we don't need to create a logical volume 
    LogicalVolume *lvol = new LogicalVolume("", shape);
    //   we would have to replace nullptr by lvol below...
    fPlaced = shape->PlaceVolume(lvol, &Transformation3D::kIdentity);
  } 
  
/// Destructor
  ~UnplacedScaledShape() {
    delete fPlaced;
  }    
    
  virtual int memory_size() const { return sizeof(*this); } // should add size of *fPlaced ??

  #ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedScaledShape>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
  #endif

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  const VUnplacedVolume *UnscaledShape() const { return fPlaced->GetLogicalVolume()->GetUnplacedVolume(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Volume() const {
    Precision capacity = ((VPlacedVolume*)fPlaced)->Capacity();
    const Vector3D<Precision> &scl = fScale.Scale();
    capacity *=scl[0]*scl[1]*scl[2];
    return capacity;
  }  
  	
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision Capacity() { return Volume(); }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision SurfaceArea() const {
  /// To do - not so easy as for the capacity...
    Precision area = ((VPlacedVolume*)fPlaced)->SurfaceArea();
    return area;	
  }

  void Extent( Vector3D<Precision> &min, Vector3D<Precision> &max) const;
    	
  Vector3D<Precision> GetPointOnSurface() const;

#if !defined(VECGEOM_NVCC)
  virtual std::string GetEntityType() const { return "ScaledShape";}
#endif // !VECGEOM_NVCC

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const;

  virtual void Print(std::ostream &os) const;

#ifndef VECGEOM_NVCC

  template <TranslationCode trans_code, RotationCode rot_code>
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
                               VPlacedVolume *const placement = NULL);

  static VPlacedVolume* CreateSpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      VPlacedVolume *const placement = NULL);

#else

  template <TranslationCode trans_code, RotationCode rot_code>
  __device__
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
                               const int id,
                               VPlacedVolume *const placement = NULL);

  __device__
  static VPlacedVolume* CreateSpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      const int id, VPlacedVolume *const placement = NULL);

#endif
  
private:

#ifndef VECGEOM_NVCC

  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const lvolume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      VPlacedVolume *const placement = NULL) const {
    return CreateSpecializedVolume(lvolume, transformation, trans_code, rot_code,
                                   placement);
  }

#else

  __device__
  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
      const int id, VPlacedVolume *const placement = NULL) const {
    return CreateSpecializedVolume(volume, transformation, trans_code, rot_code,
                                   id, placement);
  }

#endif
  
};

} } // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACESCALEDSHAPE_H_

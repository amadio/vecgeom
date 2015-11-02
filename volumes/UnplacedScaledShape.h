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

public:
/// Dummy ctor
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedScaledShape() : fPlaced(nullptr), fScale() { }
  
/// Constructor based on placed volume
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedScaledShape(VPlacedVolume const* placed, Precision sx, Precision sy, Precision sz) : 
     fPlaced(placed), fScale(sx,sy,sz) {/* assert(placed->GetTransformation()->IsIdentity());*/ }

/// Constructor based on unplaced volume
#if !defined(VECGEOM_NVCC) 
  UnplacedScaledShape(VUnplacedVolume const *shape, Precision sx, Precision sy, Precision sz) : fPlaced(0), fScale(sx,sy,sz) {
    // We need to create a placement with identity transformation from the unplaced version
    // Hopefully we don't need to create a logical volume 
    LogicalVolume *lvol = new LogicalVolume("", shape);
    fPlaced = lvol->Place();    
  }
#endif  

/// Copy constructor 
//  VECGEOM_CUDA_HEADER_BOTH
  UnplacedScaledShape(UnplacedScaledShape const &other) : fPlaced(other.fPlaced->GetLogicalVolume()->Place()), fScale(other.fScale) { }

/// Assignment operator 
//  VECGEOM_CUDA_HEADER_BOTH
  UnplacedScaledShape &operator=(UnplacedScaledShape const &other) {
    if (&other != this) {
      fPlaced = other.fPlaced->GetLogicalVolume()->Place();
      fScale = other.fScale;
    }
    return *this;
  }    
  
/// Destructor
  VECGEOM_CUDA_HEADER_BOTH
  virtual ~UnplacedScaledShape() {
    delete fPlaced;
  }    
    
  virtual int memory_size() const { return ( sizeof(*this) + fPlaced->memory_size() ); } 

  #ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedScaledShape>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
  #endif

  VECGEOM_INLINE
  const VUnplacedVolume *UnscaledShape() const { return fPlaced->GetLogicalVolume()->GetUnplacedVolume(); }

  VECGEOM_INLINE
  VPlacedVolume const* GetPlaced() const { return fPlaced; }

  VECGEOM_INLINE
  Scale3D const  &GetScale() const { return fScale; }

#ifndef VECGEOM_NVCC
  VECGEOM_INLINE
  Precision Volume() const {
    Precision capacity = ((VPlacedVolume*)fPlaced)->Capacity();
    const Vector3D<Precision> &scl = fScale.Scale();
    capacity *=scl[0]*scl[1]*scl[2];
    return capacity;
  }  
  	
  VECGEOM_INLINE
  Precision Capacity() { return Volume(); }

  VECGEOM_INLINE
  Precision SurfaceArea() const {
  /// To do - not so easy as for the capacity...
    Precision area = ((VPlacedVolume*)fPlaced)->SurfaceArea();
    return area;	
  }
#endif // !VECGEOM_NVCC

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool IsConvex() const override {return fPlaced->GetUnplacedVolume()->IsConvex();}

  void Extent( Vector3D<Precision> &min, Vector3D<Precision> &max) const;
    	
  Vector3D<Precision> GetPointOnSurface() const;

  virtual std::string GetEntityType() const { return "ScaledShape";}

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const;

  virtual void Print(std::ostream &os) const;


  template <TranslationCode trans_code, RotationCode rot_code>
  VECGEOM_CUDA_HEADER_DEVICE
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
 #ifdef VECGEOM_NVCC
                               const int id,
 #endif// !VECGEOM_NVCC
                               VPlacedVolume *const placement = NULL);

  VECGEOM_CUDA_HEADER_DEVICE
  static VPlacedVolume* CreateSpecializedVolume(
                               LogicalVolume const *const volume,
                               Transformation3D const *const transformation,
                               const TranslationCode trans_code, const RotationCode rot_code,
 #ifdef VECGEOM_NVCC
                               const int id, 
 #endif// !VECGEOM_NVCC
                               VPlacedVolume *const placement = NULL);

  
private:

  VECGEOM_CUDA_HEADER_DEVICE
  virtual VPlacedVolume* SpecializedVolume(
       LogicalVolume const *const volume,
       Transformation3D const *const transformation,
       const TranslationCode trans_code, const RotationCode rot_code,
 #ifdef VECGEOM_NVCC
       const int id,
 #endif
       VPlacedVolume *const placement = NULL) const {
    return CreateSpecializedVolume(volume, transformation, trans_code, rot_code,
 #ifdef VECGEOM_NVCC
                                   id,
 #endif
                                   placement); } 
  void SetPlaced(VPlacedVolume const *pvol) { fPlaced = pvol; }
  void SetScale(Scale3D const &scale) { fScale = scale; }
   
  friend class GeoManager;
};

} } // End global namespace

#endif // VECGEOM_VOLUMES_UNPLACESCALEDSHAPE_H_

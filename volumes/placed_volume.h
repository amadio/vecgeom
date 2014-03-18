/**
 * @file placed_volume.h
 * @author Johannes de Fine Licht (johannes.definelicht@cern.ch)
 */

#ifndef VECGEOM_VOLUMES_PLACEDVOLUME_H_
#define VECGEOM_VOLUMES_PLACEDVOLUME_H_

#include "base/global.h"

#include "base/transformation_matrix.h"
#include "management/geo_manager.h"
#include "volumes/logical_volume.h"

namespace vecgeom {

class PlacedBox;

class VPlacedVolume {

protected:

  LogicalVolume const *logical_volume_;
  TransformationMatrix const *matrix_;
  PlacedBox const *bounding_box_;

  VECGEOM_CUDA_HEADER_BOTH
  VPlacedVolume(LogicalVolume const *const logical_volume,
                TransformationMatrix const *const matrix,
                PlacedBox const *const bounding_box)
      : logical_volume_(logical_volume), matrix_(matrix),
        bounding_box_(bounding_box) {}

public:

  virtual ~VPlacedVolume() {}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  PlacedBox const* bounding_box() const { return bounding_box_; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  LogicalVolume const* logical_volume() const {
    return logical_volume_;
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector<Daughter> const& daughters() const {
    return logical_volume_->daughters();
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  VUnplacedVolume const* unplaced_volume() const {
    return logical_volume_->unplaced_volume();
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  TransformationMatrix const* matrix() const {
    return matrix_;
  }

  VECGEOM_CUDA_HEADER_BOTH
  void set_logical_volume(LogicalVolume const *const logical_volume) {
    logical_volume_ = logical_volume;
  }

  VECGEOM_CUDA_HEADER_BOTH
  void set_matrix(TransformationMatrix const *const matrix) {
    matrix_ = matrix;
  }

  VECGEOM_CUDA_HEADER_HOST
  friend std::ostream& operator<<(std::ostream& os, VPlacedVolume const &vol);

  virtual int memory_size() const =0;

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual bool Inside(Vector3D<Precision> const &point) const =0;


  virtual void Inside(SOA3D<Precision> const &point,
                      bool *const output) const =0;

  virtual void Inside(AOS3D<Precision> const &point,
                      bool *const output) const =0;

  /** an inside function that gives back the localpoint in the reference frame of the callee
   * this is useful for the locate function
   **/
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  virtual bool Inside(Vector3D<Precision> const &point,
		  	  	  	  Vector3D<Precision> & localpoint) const =0;


  /** an inside function where we know that localpoint is already in the reference frame of the callee
    **/
   VECGEOM_CUDA_HEADER_BOTH
   VECGEOM_INLINE
   virtual bool UnplacedInside(Vector3D<Precision> const &localpoint) const =0;



  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToIn(Vector3D<Precision> const &position,
                                 Vector3D<Precision> const &direction,
                                 const Precision step_max = kInfinity) const =0;

  virtual void DistanceToIn(SOA3D<Precision> const &position,
                            SOA3D<Precision> const &direction,
                            Precision const *const step_max,
                            Precision *const output) const =0;

  virtual void DistanceToIn(AOS3D<Precision> const &position,
                            AOS3D<Precision> const &direction,
                            Precision const *const step_max,
                            Precision *const output) const =0;

  virtual void DistanceToOut(SOA3D<Precision> const &position,
                              SOA3D<Precision> const &direction,
                              Precision const *const step_max,
                              Precision *const output) const =0;

  virtual void DistanceToOut(AOS3D<Precision> const &position,
                              AOS3D<Precision> const &direction,
                              Precision const *const step_max,
                              Precision *const output) const =0;


  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToOut(Vector3D<Precision> const &position,
		  	  	  	  	  	  	 Vector3D<Precision> const &direction,
                                 Precision const step_max = kInfinity) const =0;


protected:

  template <TranslationCode trans_code, RotationCode rot_code,
            typename VolumeType, typename ContainerType>
  VECGEOM_INLINE
  static void InsideBackend(VolumeType const &volume,
                            ContainerType const &points,
                            bool *const output);

  template <TranslationCode trans_code, RotationCode rot_code,
            typename VolumeType, typename ContainerType>
  VECGEOM_INLINE
  static void DistanceToInBackend(VolumeType const &volume,
                                  ContainerType const &positions,
                                  ContainerType const &directions,
                                  Precision const *const step_max,
                                  Precision *const output);

public:

  #ifdef VECGEOM_CUDA
  virtual VPlacedVolume* CopyToGpu(LogicalVolume const *const logical_volume,
                                   TransformationMatrix const *const matrix,
                                   VPlacedVolume *const gpu_ptr) const =0;
  virtual VPlacedVolume* CopyToGpu(
      LogicalVolume const *const logical_volume,
      TransformationMatrix const *const matrix) const =0;
  #endif

  #ifdef VECGEOM_BENCHMARK
  virtual VPlacedVolume const* ConvertToUnspecialized() const =0;
  virtual TGeoShape const* ConvertToRoot() const =0;
  virtual ::VUSolid const* ConvertToUSolids() const =0;
  #endif

};

} // End namespace vecgeom

#endif // VECGEOM_VOLUMES_PLACEDVOLUME_H_]

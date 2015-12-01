/// \file ShapeImplementationHelper.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_SHAPEIMPLEMENTATIONHELPER_H_
#define VECGEOM_VOLUMES_SHAPEIMPLEMENTATIONHELPER_H_

#include "base/Global.h"

#include "backend/scalar/Backend.h"
#include "backend/Backend.h"
#include "base/SOA3D.h"
#include "base/AOS3D.h"
#include "volumes/PlacedBox.h"

#include <algorithm>

#ifdef VECGEOM_DISTANCE_DEBUG
#include "volumes/utilities/ResultComparator.h"
#endif

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(ShapeImplementationHelper,class)

inline namespace VECGEOM_IMPL_NAMESPACE {

template <class Specialization>
class ShapeImplementationHelper : public Specialization::PlacedShape_t {

using PlacedShape_t = typename Specialization::PlacedShape_t;
using UnplacedShape_t = typename Specialization::UnplacedShape_t;
using Helper_t = ShapeImplementationHelper<Specialization>;
//using Implementation_t = Specialization;  // not used

public:

#ifndef VECGEOM_NVCC

  ShapeImplementationHelper(char const *const label,
                            LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox)
      : PlacedShape_t(label, logical_volume, transformation, boundingBox) {}

  ShapeImplementationHelper(char const *const label,
                            LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
     : ShapeImplementationHelper(label, logical_volume, transformation, details::UseIfSameType<PlacedShape_t,PlacedBox>::Get(this)) {}

  ShapeImplementationHelper(char const *const label,
                            LogicalVolume *const logical_volume,
                            Transformation3D const*const transformation,
                            PlacedBox const*const boundingBox)
      : PlacedShape_t(label, logical_volume, transformation, boundingBox) {}

  ShapeImplementationHelper(char const *const label,
                            LogicalVolume *const logical_volume,
                            Transformation3D const*const transformation)
      : ShapeImplementationHelper(label, logical_volume, transformation, details::UseIfSameType<PlacedShape_t,PlacedBox>::Get(this)) {}

  ShapeImplementationHelper(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox)
      : ShapeImplementationHelper("", logical_volume, transformation,
                                  boundingBox) {}


  ShapeImplementationHelper(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : ShapeImplementationHelper("", logical_volume, transformation) {}

  // this constructor mimics the constructor from the Unplaced solid
  // it ensures that placed volumes can be constructed just like ordinary Geant4/ROOT/USolids solids
  template <typename... ArgTypes>
  ShapeImplementationHelper(char const *const label, ArgTypes... params)
      : ShapeImplementationHelper(label,
                                  new LogicalVolume(new UnplacedShape_t(params...)),
                                  &Transformation3D::kIdentity) {}


#else // Compiling for CUDA

  __device__
  ShapeImplementationHelper(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox,
                            const int id)
      : PlacedShape_t(logical_volume, transformation, boundingBox, id) {}


  __device__
  ShapeImplementationHelper(LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation,
                            const int id)
      : PlacedShape_t(logical_volume, transformation, details::UseIfSameType<PlacedShape_t,PlacedBox>::Get(this), id) {}

#endif

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void PrintType() const { Specialization::PrintType(); }

#ifdef VECGEOM_CUDA_INTERFACE

  virtual size_t DeviceSizeOf() const { return DevicePtr<CudaType_t<Helper_t> >::SizeOf(); }

  DevicePtr<cuda::VPlacedVolume> CopyToGpu(
    DevicePtr<cuda::LogicalVolume> const logical_volume,
    DevicePtr<cuda::Transformation3D> const transform,
    DevicePtr<cuda::VPlacedVolume> const in_gpu_ptr) const
 {
     DevicePtr<CudaType_t<Helper_t> > gpu_ptr(in_gpu_ptr);
     gpu_ptr.Construct(logical_volume, transform, DevicePtr<cuda::PlacedBox>(), this->id());
     CudaAssertError();
     // Need to go via the void* because the regular c++ compilation
     // does not actually see the declaration for the cuda version
     // (and thus can not determine the inheritance).
     return DevicePtr<cuda::VPlacedVolume>((void*)gpu_ptr);
 }

 DevicePtr<cuda::VPlacedVolume> CopyToGpu(
    DevicePtr<cuda::LogicalVolume> const logical_volume,
    DevicePtr<cuda::Transformation3D> const transform) const
 {
     DevicePtr<CudaType_t<Helper_t> > gpu_ptr;
     gpu_ptr.Allocate();
     return CopyToGpu(logical_volume,transform,
                      DevicePtr<cuda::VPlacedVolume>((void*)gpu_ptr));
 }

#endif // VECGEOM_CUDA_INTERFACE

  VECGEOM_CUDA_HEADER_BOTH
  virtual EnumInside Inside(Vector3D<Precision> const &point) const {
    Inside_t output = EInside::kOutside;
    Specialization::template Inside<kScalar>(
      *this->GetUnplacedVolume(),
      *this->GetTransformation(),
      point,
      output
    );
    // we need to convert the output from int to an enum
    // necessary because Inside kernels operate on ints to be able to vectorize operations
    return (EnumInside) output;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Contains(Vector3D<Precision> const &point) const {
    bool output = false;
    Vector3D<Precision> localPoint;
    Specialization::template Contains<kScalar>(
      *this->GetUnplacedVolume(),
      *this->GetTransformation(),
      point,
      localPoint,
      output
    );
    return output;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Contains(Vector3D<Precision> const &point,
                        Vector3D<Precision> &localPoint) const {
    bool output = false;
    Specialization::template Contains<kScalar>(
      *this->GetUnplacedVolume(),
      *this->GetTransformation(),
      point,
      localPoint,
      output
    );


#ifdef VECGEOM_DISTANCE_DEBUG
    DistanceComparator::CompareUnplacedContains( this, output, localPoint );
#endif

    return output;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool UnplacedContains(Vector3D<Precision> const &point) const {
    bool output = false;
    Specialization::template UnplacedContains<kScalar>(
      *this->GetUnplacedVolume(),
      point,
      output
    );

#ifdef VECGEOM_DISTANCE_DEBUG
    DistanceComparator::CompareUnplacedContains( this, output, point );
#endif

    return output;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToIn(Vector3D<Precision> const &point,
                                 Vector3D<Precision> const &direction,
                                 const Precision stepMax = kInfinity) const {
#ifndef VECGEOM_NVCC
      assert( direction.IsNormalized() && " direction not normalized in call to  DistanceToIn " );
#endif
      Precision output = kInfinity;
    Specialization::template DistanceToIn<kScalar>(
      *this->GetUnplacedVolume(),
      *this->GetTransformation(),
      point,
      direction,
      stepMax,
      output
    );

#ifdef VECGEOM_REPLACE_USOLIDS
    // avoid distance values within kTolerance
    MaskedAssign(Abs(output)<kTolerance, 0., &distance);
#endif

#ifdef VECGEOM_DISTANCE_DEBUG
    DistanceComparator::CompareDistanceToIn( this, output, point, direction, stepMax );
#endif

    return output;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToOut(Vector3D<Precision> const &point,
                                  Vector3D<Precision> const &direction,
                                  const Precision stepMax = kInfinity) const {
#ifndef VECGEOM_NVCC
      assert( direction.IsNormalized() && " direction not normalized in call to  DistanceToOut " );
#endif
    Precision output = kInfinity;
    Specialization::template DistanceToOut<kScalar>(
      *this->GetUnplacedVolume(),
      point,
      direction,
      stepMax,
      output
    );

#ifdef VECGEOM_REPLACE_USOLIDS
    // avoid distance values within kTolerance
    MaskedAssign(Abs(output)<kTolerance, 0., &distance);
#endif

#ifdef VECGEOM_DISTANCE_DEBUG
    DistanceComparator::CompareDistanceToOut( this, output, point, direction, stepMax );
#endif

    // detect -inf responses which are often an indication for a real bug
#ifndef VECGEOM_NVCC
    assert( ! ( (output < 0.) && std::isinf(output) ) );
#endif

    return output;
  }


  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision PlacedDistanceToOut(Vector3D<Precision> const &point,
                                        Vector3D<Precision> const &direction,
                                        const Precision stepMax = kInfinity) const {
#ifndef VECGEOM_NVCC
      assert( direction.IsNormalized() && " direction not normalized in call to  PlacedDistanceToOut " );
#endif
     Precision output = kInfinity;
     Transformation3D const * t = this->GetTransformation();
     Specialization::template DistanceToOut<kScalar>(
        *this->GetUnplacedVolume(),
        t->Transform< Specialization::transC, Specialization::rotC, Precision>(point),
        t->TransformDirection< Specialization::rotC, Precision>(direction),
        stepMax,
        output
      );

  #ifdef VECGEOM_DISTANCE_DEBUG
      DistanceComparator::CompareDistanceToOut(
              this,
              output,
              this->GetTransformation()->Transform(point),
              this->GetTransformation()->TransformDirection(direction),
              stepMax );
  #endif
      return output;
    }



#ifdef VECGEOM_USOLIDS
  /*
   * WARNING: Trivial implementation for standard USolids interface
   * for DistanceToOut. The value for convex might be wrong
   */
  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision DistanceToOut(Vector3D<Precision> const &point,
                                  Vector3D<Precision> const &direction,
                                  Vector3D<Precision> &normal,
                                  bool &convex, Precision step = kInfinity ) const {
    double d = DistanceToOut(point, direction, step);
    Vector3D<double> hitpoint = point + d * direction;
    PlacedShape_t::Normal(hitpoint, normal);

    // Lets the shape tell itself whether it is convex or not.
    // convex = PlacedShape_t::IsConvex();

    // Now Convexity is defined only for UnplacedVolume, not required for PlacedVolume
    convex = this->GetUnplacedVolume()->UnplacedShape_t::IsConvex();

    return d;
  }
#endif

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision SafetyToIn(Vector3D<Precision> const &point) const {
    Precision output = kInfinity;
    Specialization::template SafetyToIn<kScalar>(
      *this->GetUnplacedVolume(),
      *this->GetTransformation(),
      point,
      output
    );
#ifdef VECGEOM_REPLACE_USOLIDS
    if(output < 0.0 && output > -kTolerance) output = 0.0;
#endif
    return output;
  }

  VECGEOM_CUDA_HEADER_BOTH
  virtual Precision SafetyToOut(Vector3D<Precision> const &point) const {
    Precision output = kInfinity;
    Specialization::template SafetyToOut<kScalar>(
      *this->GetUnplacedVolume(),
      point,
      output
    );
#ifdef VECGEOM_REPLACE_USOLIDS
    if(output < 0.0) output = 0.0;
#endif
    return output;
  }

  // virtual void Contains(AOS3D<Precision> const &points,
  //                       bool *const output) const {
  //   ContainsTemplate(points, output);
  // }

  virtual void Contains(SOA3D<Precision> const &points,
                        bool *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(
        VECGEOM_BACKEND_PRECISION(points.x()+i),
        VECGEOM_BACKEND_PRECISION(points.y()+i),
        VECGEOM_BACKEND_PRECISION(points.z()+i)
      );
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> localPoint;
      VECGEOM_BACKEND_BOOL result(false);
      Specialization::template Contains<VECGEOM_BACKEND_TYPE>(
        *this->GetUnplacedVolume(),
        *this->GetTransformation(),
        point,
        localPoint,
        result
      );
      StoreTo(result, output+i);
    }
  }

  // virtual void Inside(AOS3D<Precision> const &points,
  //                     Inside_t *const output) const {
  //   InsideTemplate(points, output);
  // }

  virtual void Inside(SOA3D<Precision> const &points,
                      Inside_t *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(
        VECGEOM_BACKEND_PRECISION(points.x()+i),
        VECGEOM_BACKEND_PRECISION(points.y()+i),
        VECGEOM_BACKEND_PRECISION(points.z()+i)
      );
      VECGEOM_BACKEND_INSIDE result = VECGEOM_BACKEND_INSIDE(EInside::kOutside);
      Specialization::template Inside<VECGEOM_BACKEND_TYPE>(
        *this->GetUnplacedVolume(),
        *this->GetTransformation(),
        point,
        result
      );
#ifdef VECGEOM_VC
      // Vc breaks VecGeom when using the StoreTo operation or its own store operation:
      //StoreTo(result, output+i);
      //result.store(output+i);
      for(unsigned j=0; j<kVectorSize; j++)
        output[i+j] = result[j];
#else
      StoreTo(result, output+i);
#endif
    }
  }

  // virtual void DistanceToIn(AOS3D<Precision> const &points,
  //                           AOS3D<Precision> const &directions,
  //                           Precision const *const stepMax,
  //                           Precision *const output) const {
  //   DistanceToInTemplate(points, directions, stepMax, output);
  // }

  virtual void DistanceToIn(SOA3D<Precision> const &points,
                            SOA3D<Precision> const &directions,
                            Precision const *const stepMax,
                            Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(
        VECGEOM_BACKEND_PRECISION(points.x()+i),
        VECGEOM_BACKEND_PRECISION(points.y()+i),
        VECGEOM_BACKEND_PRECISION(points.z()+i)
      );
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> direction(
        VECGEOM_BACKEND_PRECISION(directions.x()+i),
        VECGEOM_BACKEND_PRECISION(directions.y()+i),
        VECGEOM_BACKEND_PRECISION(directions.z()+i)
      );
      VECGEOM_BACKEND_TYPE::precision_v stepMaxBackend = VECGEOM_BACKEND_PRECISION(&stepMax[i]);
      VECGEOM_BACKEND_TYPE::precision_v result = kInfinity;
      Specialization::template DistanceToIn<VECGEOM_BACKEND_TYPE>(
        *this->GetUnplacedVolume(),
        *this->GetTransformation(),
        point,
        direction,
        stepMaxBackend,
        result
      );
      StoreTo(result, output+i);
    }
  }


#if !defined(__clang__) && !defined(VECGEOM_INTEL) && defined(VECGEOM_VC)
  #pragma GCC push_options
  #pragma GCC optimize ("unroll-loops")
#endif
  virtual void DistanceToInMinimize(SOA3D<Precision> const &points,
                                    SOA3D<Precision> const &directions,
                                    int daughterId,
                                    Precision *const currentDistance,
                                    int *const nextDaughterIdList) const {
    unsigned safesize = points.size() - points.size() % kVectorSize;
    for (int i = 0; i < safesize; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(
        VECGEOM_BACKEND_PRECISION(points.x()+i),
        VECGEOM_BACKEND_PRECISION(points.y()+i),
        VECGEOM_BACKEND_PRECISION(points.z()+i)
      );
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> direction(
        VECGEOM_BACKEND_PRECISION(directions.x()+i),
        VECGEOM_BACKEND_PRECISION(directions.y()+i),
        VECGEOM_BACKEND_PRECISION(directions.z()+i)
      );
      // currentDistance is also estimate for stepMax
      VECGEOM_BACKEND_TYPE::precision_v stepMaxBackend = VECGEOM_BACKEND_PRECISION(&currentDistance[i]);
      VECGEOM_BACKEND_TYPE::precision_v result = kInfinity;
      Specialization::template DistanceToIn<VECGEOM_BACKEND_TYPE>(
        *this->GetUnplacedVolume(),
        *this->GetTransformation(),
        point,
        direction,
        stepMaxBackend,
        result
      );
      // now we have distance and we can compare it to old distance step
      // and update it if necessary
      // -1E20 used here as Vc does not have a check for minus infinity
      VECGEOM_BACKEND_BOOL valid = result < stepMaxBackend && result > -1E20;
      MaskedAssign(!valid, stepMaxBackend, &result);
      StoreTo(result, currentDistance+i); // go back to previous result if we don't get better

/*
 * Keeping the original comments:
      // currently do not know how to do this better (can do it when Vc offers long ints )
#ifdef VECGEOM_INTEL
#pragma unroll
#endif
      for(unsigned int j=0;j<kVectorSize;++j) {
        nextDaughterIdList[i+j] = (valid[j]) ? daughterId : nextDaughterIdList[i+j];
      }
*/
      MaskedAssign(valid, daughterId, nextDaughterIdList+i);
    }
    // treat the tail:
    unsigned tailsize = points.size() - safesize;
    for (unsigned i = 0; i < tailsize; ++i) {
      unsigned track = safesize + i;
      Precision result(kInfinity);
      Specialization::template DistanceToIn<kScalar>(*this->GetUnplacedVolume(), *this->GetTransformation(), points[track], directions[track],
                                                      currentDistance[track], result);
      //bool valid = result < stepMax[track] && ! IsInf(result);
      if (result < currentDistance[i] && !IsInf(result)) {
               currentDistance[i] = result;
               nextDaughterIdList[i] = daughterId;
      }
    }
  }
#if !defined(__clang__) && !defined(VECGEOM_INTEL) && defined(VECGEOM_VC)
#pragma GCC pop_options
#endif

  // virtual void DistanceToOut(AOS3D<Precision> const &points,
  //                            AOS3D<Precision> const &directions,
  //                            Precision const *const stepMax,
  //                            Precision *const output) const {
  //   DistanceToOutTemplate(points, directions, stepMax, output);
  // }

  virtual void DistanceToOut(SOA3D<Precision> const &points,
                             SOA3D<Precision> const &directions,
                             Precision const *const stepMax,
                             Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(
        VECGEOM_BACKEND_PRECISION(points.x()+i),
        VECGEOM_BACKEND_PRECISION(points.y()+i),
        VECGEOM_BACKEND_PRECISION(points.z()+i)
      );
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> direction(
        VECGEOM_BACKEND_PRECISION(directions.x()+i),
        VECGEOM_BACKEND_PRECISION(directions.y()+i),
        VECGEOM_BACKEND_PRECISION(directions.z()+i)
      );
      VECGEOM_BACKEND_TYPE::precision_v stepMaxBackend = VECGEOM_BACKEND_PRECISION(&stepMax[i]);
      VECGEOM_BACKEND_TYPE::precision_v result = kInfinity;
      Specialization::template DistanceToOut<VECGEOM_BACKEND_TYPE>(
        *this->GetUnplacedVolume(),
        point,
        direction,
        stepMaxBackend,
        result
      );
      StoreTo(result,output+i);
    }
  }

  virtual void DistanceToOut(SOA3D<Precision> const &points,
                             SOA3D<Precision> const &directions,
                             Precision const *const stepMax,
                             Precision *const output,
                             int *const nextNodeIndex) const {
    unsigned safesize = points.size() - points.size() % kVectorSize;
    for (int i = 0; i < safesize; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(
        VECGEOM_BACKEND_PRECISION(points.x()+i),
        VECGEOM_BACKEND_PRECISION(points.y()+i),
        VECGEOM_BACKEND_PRECISION(points.z()+i)
      );
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> direction(
        VECGEOM_BACKEND_PRECISION(directions.x()+i),
        VECGEOM_BACKEND_PRECISION(directions.y()+i),
        VECGEOM_BACKEND_PRECISION(directions.z()+i)
      );
      VECGEOM_BACKEND_TYPE::precision_v stepMaxBackend = VECGEOM_BACKEND_PRECISION(&stepMax[i]);
      VECGEOM_BACKEND_TYPE::precision_v result = kInfinity;
      Specialization::template DistanceToOut<VECGEOM_BACKEND_TYPE>(
        *this->GetUnplacedVolume(),
        point,
        direction,
        stepMaxBackend,
        result
      );
      MaskedAssign(result < 0., kInfinity, &result);
      StoreTo(result, output+i);
      // -1: physics step is longer than geometry
      // -2: particle may stay inside volume
      CondAssign(result < stepMaxBackend, -1, -2, nextNodeIndex+i);
    }
    // treat the tail:
    unsigned tailsize = points.size() - safesize;
    for (unsigned i = 0; i < tailsize; ++i){
        unsigned track = safesize + i;
        Precision result(vecgeom::kInfinity);
        Specialization::template DistanceToOut<kScalar>(*this->GetUnplacedVolume(), points[track], directions[track], stepMax[track], result);
        result = (result<0.)? kInfinity : result;
        output[track] = result;
        // -1: physics step is longer than geometry
        // -2: particle may stay inside volume
        nextNodeIndex[track] = ( result < stepMax[track] )? -1 : -2;
    }
  }

  virtual void SafetyToIn(SOA3D<Precision> const &points,
                          Precision *const output) const {
    for (int i = 0, i_max = points.size(); i < i_max; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(
        VECGEOM_BACKEND_PRECISION(points.x()+i),
        VECGEOM_BACKEND_PRECISION(points.y()+i),
        VECGEOM_BACKEND_PRECISION(points.z()+i)
      );
      VECGEOM_BACKEND_TYPE::precision_v result = kInfinity;
      Specialization::template SafetyToIn<VECGEOM_BACKEND_TYPE>(
        *this->GetUnplacedVolume(),
        *this->GetTransformation(),
        point,
        result
      );
      StoreTo(result, output+i);
    }
  }

  // virtual void SafetyToIn(AOS3D<Precision> const &points,
  //                         Precision *const output) const {
  //   SafetyToInTemplate(points, output);
  // }

  virtual void SafetyToInMinimize(SOA3D<Precision> const &points,
                                  Precision *const safeties) const {
    unsigned safesize = points.size() - points.size() % kVectorSize;
    for (int i = 0; i < safesize; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(
        VECGEOM_BACKEND_PRECISION(points.x()+i),
        VECGEOM_BACKEND_PRECISION(points.y()+i),
        VECGEOM_BACKEND_PRECISION(points.z()+i)
      );
      VECGEOM_BACKEND_TYPE::precision_v estimate = VECGEOM_BACKEND_PRECISION(&safeties[i]);
      VECGEOM_BACKEND_TYPE::precision_v result = kInfinity;
      Specialization::template SafetyToIn<VECGEOM_BACKEND_TYPE>(
        *this->GetUnplacedVolume(),
        *this->GetTransformation(),
        point,
        result
      );
      MaskedAssign(estimate < result, estimate, &result);
      StoreTo(result, safeties+i);
    }
    unsigned tailsize = points.size() - safesize;
    for (unsigned int i=0; i < tailsize; ++i){
        unsigned int track = safesize + i;
        Precision result = kInfinity;
        Specialization::template SafetyToIn<kScalar>(*this->GetUnplacedVolume(), *this->GetTransformation(), points[track], result);
        safeties[track] = (result < safeties[track])? result : safeties[track];
    }
  }

  virtual void SafetyToOut(SOA3D<Precision> const &points,
                          Precision *const output) const {
    unsigned safesize = points.size() - points.size() % kVectorSize;
    for (unsigned int i = 0; i < safesize; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(
        VECGEOM_BACKEND_PRECISION(points.x()+i),
        VECGEOM_BACKEND_PRECISION(points.y()+i),
        VECGEOM_BACKEND_PRECISION(points.z()+i)
      );
      VECGEOM_BACKEND_TYPE::precision_v result = kInfinity;
      Specialization::template SafetyToOut<VECGEOM_BACKEND_TYPE>(
        *this->GetUnplacedVolume(),
        point,
        result
      );
      StoreTo(result, output+i);
    }
    // tail treatment
    unsigned tailsize = points.size() - safesize;
    for (unsigned int i = 0; i < tailsize; ++i) {
        Precision result = kInfinity;
        unsigned int track = safesize + i;
        Specialization::template SafetyToOut<kScalar>(*this->GetUnplacedVolume(), points[track], result);
        output[track] = result;
    }
  }

  // virtual void SafetyToOut(AOS3D<Precision> const &points,
  //                         Precision *const output) const {
  //   SafetyToOutTemplate(points, output);
  // }

  virtual void SafetyToOutMinimize(SOA3D<Precision> const &points,
                                   Precision *const safeties) const {
    for (int i = 0, iMax = points.size(); i < iMax; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(
        VECGEOM_BACKEND_PRECISION(points.x()+i),
        VECGEOM_BACKEND_PRECISION(points.y()+i),
        VECGEOM_BACKEND_PRECISION(points.z()+i)
      );
      VECGEOM_BACKEND_TYPE::precision_v estimate = VECGEOM_BACKEND_PRECISION(&safeties[i]);
      VECGEOM_BACKEND_TYPE::precision_v result = kInfinity;
      Specialization::template SafetyToOut<VECGEOM_BACKEND_TYPE>(
        *this->GetUnplacedVolume(),
        point,
        result
      );
      MaskedAssign(estimate < result, estimate, &result);
      StoreTo(result, safeties+i);
    }
  }

}; // End class ShapeImplementationHelper

} } // End global namespace

#endif // VECGEOM_VOLUMES_SHAPEIMPLEMENTATIONHELPER_H_

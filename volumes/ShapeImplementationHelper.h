/// \file ShapeImplementationHelper.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_VOLUMES_SHAPEIMPLEMENTATIONHELPER_H_
#define VECGEOM_VOLUMES_SHAPEIMPLEMENTATIONHELPER_H_

#include "base/Global.h"

#include "backend/scalar/Backend.h"
#include "backend/Backend.h"
#include "base/SOA3D.h"
#include "volumes/PlacedBox.h"

#include <algorithm>

#ifdef VECGEOM_DISTANCE_DEBUG
#include "volumes/utilities/ResultComparator.h"
#endif

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE(class, ShapeImplementationHelper, typename);

inline namespace VECGEOM_IMPL_NAMESPACE {

template <class Specialization>
class ShapeImplementationHelper : public Specialization::PlacedShape_t {

  using PlacedShape_t   = typename Specialization::PlacedShape_t;
  using UnplacedShape_t = typename Specialization::UnplacedShape_t;
  using Helper_t        = ShapeImplementationHelper<Specialization>;
  // using Implementation_t = Specialization;  // not used

public:
#ifndef VECCORE_CUDA

  ShapeImplementationHelper(char const *const label, LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation, PlacedBox const *const boundingBox)
      : PlacedShape_t(label, logical_volume, transformation, boundingBox)
  {
  }

  ShapeImplementationHelper(char const *const label, LogicalVolume const *const logical_volume,
                            Transformation3D const *const transformation)
      : ShapeImplementationHelper(label, logical_volume, transformation,
                                  details::UseIfSameType<PlacedShape_t, PlacedBox>::Get(this))
  {
  }

  ShapeImplementationHelper(char const *const label, LogicalVolume *const logical_volume,
                            Transformation3D const *const transformation, PlacedBox const *const boundingBox)
      : PlacedShape_t(label, logical_volume, transformation, boundingBox)
  {
  }

  ShapeImplementationHelper(char const *const label, LogicalVolume *const logical_volume,
                            Transformation3D const *const transformation)
      : ShapeImplementationHelper(label, logical_volume, transformation,
                                  details::UseIfSameType<PlacedShape_t, PlacedBox>::Get(this))
  {
  }

  ShapeImplementationHelper(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                            PlacedBox const *const boundingBox)
      : ShapeImplementationHelper("", logical_volume, transformation, boundingBox)
  {
  }

  ShapeImplementationHelper(LogicalVolume const *const logical_volume, Transformation3D const *const transformation)
      : ShapeImplementationHelper("", logical_volume, transformation)
  {
  }

  // this constructor mimics the constructor from the Unplaced solid
  // it ensures that placed volumes can be constructed just like ordinary Geant4/ROOT/USolids solids
  template <typename... ArgTypes>
  ShapeImplementationHelper(char const *const label, ArgTypes... params)
      : ShapeImplementationHelper(label, new LogicalVolume(new UnplacedShape_t(params...)),
                                  &Transformation3D::kIdentity)
  {
  }

#else // Compiling for CUDA

  VECCORE_ATT_DEVICE ShapeImplementationHelper(LogicalVolume const *const logical_volume,
                                               Transformation3D const *const transformation,
                                               PlacedBox const *const boundingBox, const int id)
      : PlacedShape_t(logical_volume, transformation, boundingBox, id)
  {
  }

  VECCORE_ATT_DEVICE ShapeImplementationHelper(LogicalVolume const *const logical_volume,
                                               Transformation3D const *const transformation, const int id)
      : PlacedShape_t(logical_volume, transformation, details::UseIfSameType<PlacedShape_t, PlacedBox>::Get(this), id)
  {
  }

#endif
  using PlacedShape_t::SafetyToIn;
  using PlacedShape_t::SafetyToOut;
  using PlacedShape_t::DistanceToIn;
  using PlacedShape_t::DistanceToOut;

  virtual int MemorySize() const override { return sizeof(*this); }

  VECCORE_ATT_HOST_DEVICE
  virtual void PrintType() const override { Specialization::PrintType(); }

  virtual void PrintType(std::ostream &os) const override { Specialization::PrintType(os); }
  virtual void PrintImplementationType(std::ostream &os) const override { Specialization::PrintImplementationType(os); }
  virtual void PrintUnplacedType(std::ostream &os) const override { Specialization::PrintUnplacedType(os); }

#ifdef VECGEOM_CUDA_INTERFACE

  virtual size_t DeviceSizeOf() const override { return DevicePtr<CudaType_t<Helper_t>>::SizeOf(); }

  DevicePtr<cuda::VPlacedVolume> CopyToGpu(DevicePtr<cuda::LogicalVolume> const logical_volume,
                                           DevicePtr<cuda::Transformation3D> const transform,
                                           DevicePtr<cuda::VPlacedVolume> const in_gpu_ptr) const override
  {
    DevicePtr<CudaType_t<Helper_t>> gpu_ptr(in_gpu_ptr);
    gpu_ptr.Construct(logical_volume, transform, DevicePtr<cuda::PlacedBox>(), this->id());
    CudaAssertError();
    // Need to go via the void* because the regular c++ compilation
    // does not actually see the declaration for the cuda version
    // (and thus can not determine the inheritance).
    return DevicePtr<cuda::VPlacedVolume>((void *)gpu_ptr);
  }

  DevicePtr<cuda::VPlacedVolume> CopyToGpu(DevicePtr<cuda::LogicalVolume> const logical_volume,
                                           DevicePtr<cuda::Transformation3D> const transform) const override
  {
    DevicePtr<CudaType_t<Helper_t>> gpu_ptr;
    gpu_ptr.Allocate();
    return CopyToGpu(logical_volume, transform, DevicePtr<cuda::VPlacedVolume>((void *)gpu_ptr));
  }

#endif // VECGEOM_CUDA_INTERFACE

  VECCORE_ATT_HOST_DEVICE
  virtual EnumInside Inside(Vector3D<Precision> const &point) const override
  {
    Inside_t output = EInside::kOutside;
    Specialization::template Inside<kScalar>(*this->GetUnplacedVolume(), *this->GetTransformation(), point, output);
    // we need to convert the output from int to an enum
    // necessary because Inside kernels operate on ints to be able to vectorize operations
    return (EnumInside)output;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual bool Contains(Vector3D<Precision> const &point) const override
  {
    bool output = false;
    Vector3D<Precision> localPoint;
    Specialization::template Contains<kScalar>(*this->GetUnplacedVolume(), *this->GetTransformation(), point,
                                               localPoint, output);
    return output;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual bool Contains(Vector3D<Precision> const &point, Vector3D<Precision> &localPoint) const override
  {
    bool output = false;
    Specialization::template Contains<kScalar>(*this->GetUnplacedVolume(), *this->GetTransformation(), point,
                                               localPoint, output);

#ifdef VECGEOM_DISTANCE_DEBUG
    DistanceComparator::CompareUnplacedContains(this, output, localPoint);
#endif

    return output;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual bool UnplacedContains(Vector3D<Precision> const &point) const override
  {
    bool output = false;
    Specialization::template UnplacedContains<kScalar>(*this->GetUnplacedVolume(), point, output);

#ifdef VECGEOM_DISTANCE_DEBUG
    DistanceComparator::CompareUnplacedContains(this, output, point);
#endif

    return output;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision DistanceToIn(Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
                                 const Precision stepMax = kInfLength) const override
  {
#ifndef VECCORE_CUDA
    assert(direction.IsNormalized() && " direction not normalized in call to  DistanceToIn ");
#endif
    Precision output = kInfLength;
    Specialization::template DistanceToIn<kScalar>(*this->GetUnplacedVolume(), *this->GetTransformation(), point,
                                                   direction, stepMax, output);

#ifdef VECGEOM_REPLACE_USOLIDS
    // apply USolids convention: convert negative values to zero
    vecCore::MaskedAssign(output, output < kHalfTolerance, 0.);
#else
    // avoid distance values within tolerance
    vecCore::MaskedAssign(output, Abs(output) < kHalfTolerance, 0.);
#endif

#ifdef VECGEOM_DISTANCE_DEBUG
    DistanceComparator::CompareDistanceToIn(this, output, point, direction, stepMax);
#endif

    return output;
  }

  virtual VECGEOM_BACKEND_PRECISION_TYPE DistanceToInVec(Vector3D<VECGEOM_BACKEND_PRECISION_TYPE> const &point,
                                                         Vector3D<VECGEOM_BACKEND_PRECISION_TYPE> const &direction,
                                                         const VECGEOM_BACKEND_PRECISION_TYPE stepMax) const override
  {
    //#ifndef VECCORE_CUDA
    //    assert(direction.IsNormalized() && " direction not normalized in call to  DistanceToIn ");
    //#endif
    VECGEOM_BACKEND_PRECISION_TYPE output = kInfLength;
    Specialization::template DistanceToIn<VECGEOM_BACKEND_TYPE>(*this->GetUnplacedVolume(), *this->GetTransformation(),
                                                                point, direction, stepMax, output);

    // vecCore::MaskedAssign(output, Abs(output)<kHalfTolerance, 0.);

    //#ifdef VECGEOM_DISTANCE_DEBUG
    //    DistanceComparator::CompareDistanceToIn(this, output, point, direction, stepMax);
    //#endif

    return output;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision DistanceToOut(Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
                                  const Precision stepMax = kInfLength) const override
  {
    //#ifndef VECCORE_CUDA
    //      assert( direction.IsNormalized() && " direction not normalized in call to  DistanceToOut " );
    //#endif
    Precision output = kInfLength;
    Specialization::template DistanceToOut<kScalar>(*this->GetUnplacedVolume(), point, direction, stepMax, output);

#ifdef VECGEOM_REPLACE_USOLIDS
    // apply USolids convention: convert negative values to zero
    vecCore::MaskedAssign(output, output < kHalfTolerance, 0.);
#else
    // avoid distance values within tolerance
    vecCore::MaskedAssign(output, Abs(output) < kHalfTolerance, 0.);
#endif

#ifdef VECGEOM_DISTANCE_DEBUG
    DistanceComparator::CompareDistanceToOut(this, output, point, direction, stepMax);
#endif

    // detect -inf responses which are often an indication for a real bug
    //#ifndef VECCORE_CUDA
    //    assert( ! ( (output < 0.) && std::isinf(output) ) );
    //#endif

    return output;
  }

  virtual VECGEOM_BACKEND_PRECISION_TYPE DistanceToOutVec(Vector3D<VECGEOM_BACKEND_PRECISION_TYPE> const &point,
                                                          Vector3D<VECGEOM_BACKEND_PRECISION_TYPE> const &direction,
                                                          const VECGEOM_BACKEND_PRECISION_TYPE stepMax) const override
  {
    //#ifndef VECCORE_CUDA
    //    assert(direction.IsNormalized() && " direction not normalized in call to  DistanceToOut ");
    //#endif
    VECGEOM_BACKEND_PRECISION_TYPE output = kInfLength;
    Specialization::template DistanceToOut<VECGEOM_BACKEND_TYPE>(*this->GetUnplacedVolume(), point, direction, stepMax,
                                                                 output);
    // avoid distance values within tolerance
    // vecCore::MaskedAssign(output, Abs(output)<kHalfTolerance, 0.);

    // TODO: provide CompareDistance check for vector interface
    //#ifdef VECGEOM_DISTANCE_DEBUG
    //    DistanceComparator::CompareDistanceToOut(this, output, point, direction, stepMax);
    //#endif

    // detect -inf responses which are often an indication for a real bug
    //#ifndef VECCORE_CUDA
    // assert(!((output < 0.) && std::isinf(output)));
    //#endif

    return output;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision PlacedDistanceToOut(Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
                                        const Precision stepMax = kInfLength) const override
  {
#ifndef VECCORE_CUDA
    assert(direction.IsNormalized() && " direction not normalized in call to  PlacedDistanceToOut ");
#endif
    Precision output          = kInfLength;
    Transformation3D const *t = this->GetTransformation();
    Specialization::template DistanceToOut<kScalar>(
        *this->GetUnplacedVolume(), t->Transform<Specialization::transC, Specialization::rotC, Precision>(point),
        t->TransformDirection<Specialization::rotC, Precision>(direction), stepMax, output);

#ifdef VECGEOM_DISTANCE_DEBUG
    DistanceComparator::CompareDistanceToOut(this, output, this->GetTransformation()->Transform(point),
                                             this->GetTransformation()->TransformDirection(direction), stepMax);
#endif
    return output;
  }

#ifdef VECGEOM_USOLIDS
  /*
   * WARNING: Trivial implementation for standard USolids interface
   * for DistanceToOut. The value for convex might be wrong
   */
  VECCORE_ATT_HOST_DEVICE
  virtual Precision DistanceToOut(Vector3D<Precision> const &point, Vector3D<Precision> const &direction,
                                  Vector3D<Precision> &normal, bool &convex, Precision step = kInfLength) const override
  {
    double d                  = DistanceToOut(point, direction, step);
    Vector3D<double> hitpoint = point + d * direction;
    PlacedShape_t::Normal(hitpoint, normal);

    // Now Convexity is defined only for UnplacedVolume, not required for PlacedVolume
    convex = this->GetUnplacedVolume()->UnplacedShape_t::IsConvex();

    return d;
  }
#endif

  VECCORE_ATT_HOST_DEVICE
  virtual Precision SafetyToIn(Vector3D<Precision> const &point) const override
  {
    Precision output = kInfLength;
    Specialization::template SafetyToIn<kScalar>(*this->GetUnplacedVolume(), *this->GetTransformation(), point, output);

#ifdef VECGEOM_REPLACE_USOLIDS
    // apply USolids convention: convert negative values to zero
    vecCore::MaskedAssign(output, output < kHalfTolerance, 0.);
#else
    // avoid distance values within tolerance
    vecCore::MaskedAssign(output, Abs(output) < kHalfTolerance, 0.);
#endif

    return output;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual Precision SafetyToOut(Vector3D<Precision> const &point) const override
  {
    Precision output = kInfLength;
    Specialization::template SafetyToOut<kScalar>(*this->GetUnplacedVolume(), point, output);

#ifdef VECGEOM_REPLACE_USOLIDS
    // apply USolids convention: convert negative values to zero
    vecCore::MaskedAssign(output, output < kHalfTolerance, 0.);
#else
    // avoid distance values within tolerance
    vecCore::MaskedAssign(output, Abs(output) < kHalfTolerance, 0.);
#endif

    return output;
  }

  VECGEOM_FORCE_INLINE
  virtual VECGEOM_BACKEND_PRECISION_TYPE SafetyToInVec(
      Vector3D<VECGEOM_BACKEND_PRECISION_TYPE> const &position) const override
  {
    VECGEOM_BACKEND_PRECISION_TYPE output(kInfLength);
    Specialization::template SafetyToIn<VECGEOM_BACKEND_TYPE>(*this->GetUnplacedVolume(), *this->GetTransformation(),
                                                              position, output);

    // avoid distance values within tolerance
    // vecCore::MaskedAssign(output, Abs(output)<kHalfTolerance, 0.);

    return output;
  }

  VECGEOM_FORCE_INLINE
  virtual VECGEOM_BACKEND_PRECISION_TYPE SafetyToOutVec(
      Vector3D<VECGEOM_BACKEND_PRECISION_TYPE> const &position) const override
  {
    VECGEOM_BACKEND_PRECISION_TYPE output(kInfLength);
    Specialization::template SafetyToOut<VECGEOM_BACKEND_TYPE>(*this->GetUnplacedVolume(), position, output);

    // avoid distance values within tolerance
    // vecCore::MaskedAssign(output, Abs(output)<kHalfTolerance, 0.);

    return output;
  }

  virtual void Contains(SOA3D<Precision> const &points, bool *const output) const override
  {
    for (int i = 0, i_max = points.size(); i < i_max; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(VECGEOM_BACKEND_PRECISION_FROM_PTR(points.x() + i),
                                                        VECGEOM_BACKEND_PRECISION_FROM_PTR(points.y() + i),
                                                        VECGEOM_BACKEND_PRECISION_FROM_PTR(points.z() + i));
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> localPoint;
      VECGEOM_BACKEND_BOOL result(false);
      Specialization::template Contains<VECGEOM_BACKEND_TYPE>(*this->GetUnplacedVolume(), *this->GetTransformation(),
                                                              point, localPoint, result);
      vecCore::Store(result, output + i);
    }
  }

  virtual void Inside(SOA3D<Precision> const &points, Inside_t *const output) const override
  {
    for (int i = 0, i_max = points.size(); i < i_max; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(VECGEOM_BACKEND_PRECISION_FROM_PTR(points.x() + i),
                                                        VECGEOM_BACKEND_PRECISION_FROM_PTR(points.y() + i),
                                                        VECGEOM_BACKEND_PRECISION_FROM_PTR(points.z() + i));
      VECGEOM_BACKEND_INSIDE result = VECGEOM_BACKEND_INSIDE(EInside::kOutside);
      Specialization::template Inside<VECGEOM_BACKEND_TYPE>(*this->GetUnplacedVolume(), *this->GetTransformation(),
                                                            point, result);
#ifdef VECGEOM_VC
      // Vc breaks VecGeom when using the StoreTo operation or its own store operation:
      // StoreTo(result, output+i);
      // result.store(output+i);
      for (unsigned j = 0; j < kVectorSize; j++)
        output[i + j] = result[j];
#else
      vecCore::Store(result, output + i);
#endif
    }
  }

  virtual void DistanceToIn(SOA3D<Precision> const &points, SOA3D<Precision> const &directions,
                            Precision const *const stepMax, Precision *const output) const override
  {
    for (int i = 0, i_max = points.size(); i < i_max; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(VECGEOM_BACKEND_PRECISION_FROM_PTR(points.x() + i),
                                                        VECGEOM_BACKEND_PRECISION_FROM_PTR(points.y() + i),
                                                        VECGEOM_BACKEND_PRECISION_FROM_PTR(points.z() + i));
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> direction(VECGEOM_BACKEND_PRECISION_FROM_PTR(directions.x() + i),
                                                            VECGEOM_BACKEND_PRECISION_FROM_PTR(directions.y() + i),
                                                            VECGEOM_BACKEND_PRECISION_FROM_PTR(directions.z() + i));
      VECGEOM_BACKEND_TYPE::precision_v stepMaxBackend = VECGEOM_BACKEND_PRECISION_FROM_PTR(&stepMax[i]);
      VECGEOM_BACKEND_TYPE::precision_v result         = kInfLength;
      Specialization::template DistanceToIn<VECGEOM_BACKEND_TYPE>(
          *this->GetUnplacedVolume(), *this->GetTransformation(), point, direction, stepMaxBackend, result);
      vecCore::Store(result, output + i);
    }
  }

#if !defined(__clang__) && !defined(VECGEOM_INTEL) && defined(VECGEOM_VC)
#pragma GCC push_options
#pragma GCC optimize("unroll-loops")
#endif
  virtual void DistanceToInMinimize(SOA3D<Precision> const &points, SOA3D<Precision> const &directions, int daughterId,
                                    Precision *const currentDistance, int *const nextDaughterIdList) const override
  {
    unsigned safesize = points.size() - points.size() % kVectorSize;
    for (unsigned int i = 0; i < safesize; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(VECGEOM_BACKEND_PRECISION_FROM_PTR(points.x() + i),
                                                        VECGEOM_BACKEND_PRECISION_FROM_PTR(points.y() + i),
                                                        VECGEOM_BACKEND_PRECISION_FROM_PTR(points.z() + i));
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> direction(VECGEOM_BACKEND_PRECISION_FROM_PTR(directions.x() + i),
                                                            VECGEOM_BACKEND_PRECISION_FROM_PTR(directions.y() + i),
                                                            VECGEOM_BACKEND_PRECISION_FROM_PTR(directions.z() + i));
      // currentDistance is also estimate for stepMax
      VECGEOM_BACKEND_TYPE::precision_v stepMaxBackend = VECGEOM_BACKEND_PRECISION_FROM_PTR(&currentDistance[i]);
      VECGEOM_BACKEND_TYPE::precision_v result         = kInfLength;
      Specialization::template DistanceToIn<VECGEOM_BACKEND_TYPE>(
          *this->GetUnplacedVolume(), *this->GetTransformation(), point, direction, stepMaxBackend, result);
      // now we have distance and we can compare it to old distance step
      // and update it if necessary
      // -1E20 used here as Vc does not have a check for minus infinity
      VECGEOM_BACKEND_BOOL valid = result < stepMaxBackend && result > -1E20;
      vecCore::MaskedAssign(result, !valid, stepMaxBackend);
      vecCore::Store(result, currentDistance + i); // go back to previous result if we don't get better

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
      for (size_t j = 0; j < vecCore::VectorSize(result); j++)
        if (vecCore::Get(valid, j)) nextDaughterIdList[i + j] = vecCore::Get(valid, j);
    }
    // treat the tail:
    unsigned tailsize = points.size() - safesize;
    for (unsigned i = 0; i < tailsize; ++i) {
      unsigned track = safesize + i;
      Precision result(kInfLength);
      Specialization::template DistanceToIn<kScalar>(*this->GetUnplacedVolume(), *this->GetTransformation(),
                                                     points[track], directions[track], currentDistance[track], result);
      // bool valid = result < stepMax[track] && ! IsInf(result);
      if (result < currentDistance[i] && !IsInf(result)) {
        currentDistance[i]    = result;
        nextDaughterIdList[i] = daughterId;
      }
    }
  }
#if !defined(__clang__) && !defined(VECGEOM_INTEL) && defined(VECGEOM_VC)
#pragma GCC pop_options
#endif

  virtual void DistanceToOut(SOA3D<Precision> const &points, SOA3D<Precision> const &directions,
                             Precision const *const stepMax, Precision *const output) const override
  {
    for (unsigned int i = 0, i_max = points.size(); i < i_max; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(VECGEOM_BACKEND_PRECISION_FROM_PTR(points.x() + i),
                                                        VECGEOM_BACKEND_PRECISION_FROM_PTR(points.y() + i),
                                                        VECGEOM_BACKEND_PRECISION_FROM_PTR(points.z() + i));
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> direction(VECGEOM_BACKEND_PRECISION_FROM_PTR(directions.x() + i),
                                                            VECGEOM_BACKEND_PRECISION_FROM_PTR(directions.y() + i),
                                                            VECGEOM_BACKEND_PRECISION_FROM_PTR(directions.z() + i));
      VECGEOM_BACKEND_TYPE::precision_v stepMaxBackend = VECGEOM_BACKEND_PRECISION_FROM_PTR(&stepMax[i]);
      VECGEOM_BACKEND_TYPE::precision_v result         = kInfLength;
      Specialization::template DistanceToOut<VECGEOM_BACKEND_TYPE>(*this->GetUnplacedVolume(), point, direction,
                                                                   stepMaxBackend, result);
      vecCore::Store(result, output + i);
    }
  }

  virtual void DistanceToOut(SOA3D<Precision> const &points, SOA3D<Precision> const &directions,
                             Precision const *const stepMax, Precision *const output,
                             int *const nextNodeIndex) const override
  {
    unsigned safesize = points.size() - points.size() % kVectorSize;
    for (unsigned int i = 0; i < safesize; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(VECGEOM_BACKEND_PRECISION_FROM_PTR(points.x() + i),
                                                        VECGEOM_BACKEND_PRECISION_FROM_PTR(points.y() + i),
                                                        VECGEOM_BACKEND_PRECISION_FROM_PTR(points.z() + i));
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> direction(VECGEOM_BACKEND_PRECISION_FROM_PTR(directions.x() + i),
                                                            VECGEOM_BACKEND_PRECISION_FROM_PTR(directions.y() + i),
                                                            VECGEOM_BACKEND_PRECISION_FROM_PTR(directions.z() + i));
      VECGEOM_BACKEND_TYPE::precision_v stepMaxBackend = VECGEOM_BACKEND_PRECISION_FROM_PTR(&stepMax[i]);
      VECGEOM_BACKEND_TYPE::precision_v result         = kInfLength;
      Specialization::template DistanceToOut<VECGEOM_BACKEND_TYPE>(*this->GetUnplacedVolume(), point, direction,
                                                                   stepMaxBackend, result);

      vecCore__MaskedAssignFunc(result, result < 0.0, InfinityLength<decltype(result)>());
      vecCore::Store(result, output + i);
      // -1: physics step is longer than geometry
      // -2: particle may stay inside volume
      VECGEOM_BACKEND_TYPE::int_v vnegone(-1), vnegtwo(-2);
      vecCore::Mask<VECGEOM_BACKEND_TYPE::int_v> mask(result < stepMaxBackend);
      VECGEOM_BACKEND_TYPE::int_v vIndex = vecCore::Blend(mask, vnegone, vnegtwo);
      vecCore::Store(vIndex, nextNodeIndex + i);
    }
    // treat the tail:
    unsigned tailsize = points.size() - safesize;
    for (unsigned i = 0; i < tailsize; ++i) {
      unsigned track = safesize + i;
      Precision result(vecgeom::kInfLength);
      Specialization::template DistanceToOut<kScalar>(*this->GetUnplacedVolume(), points[track], directions[track],
                                                      stepMax[track], result);
      result        = (result < 0.) ? kInfLength : result;
      output[track] = result;
      // -1: physics step is longer than geometry
      // -2: particle may stay inside volume
      nextNodeIndex[track] = (result < stepMax[track]) ? -1 : -2;
    }
  }

  virtual void SafetyToIn(SOA3D<Precision> const &points, Precision *const output) const override
  {
    for (int i = 0, i_max = points.size(); i < i_max; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(VECGEOM_BACKEND_PRECISION_FROM_PTR(points.x() + i),
                                                        VECGEOM_BACKEND_PRECISION_FROM_PTR(points.y() + i),
                                                        VECGEOM_BACKEND_PRECISION_FROM_PTR(points.z() + i));
      VECGEOM_BACKEND_TYPE::precision_v result = kInfLength;
      Specialization::template SafetyToIn<VECGEOM_BACKEND_TYPE>(*this->GetUnplacedVolume(), *this->GetTransformation(),
                                                                point, result);

      vecCore::MaskedAssign(result, Abs(result) < kHalfTolerance, VECGEOM_BACKEND_PRECISION_TYPE(0.0));

      vecCore::Store(result, output + i);
    }
  }

  virtual void SafetyToInMinimize(SOA3D<Precision> const &points, Precision *const safeties) const override
  {
    unsigned safesize = points.size() - points.size() % kVectorSize;
    for (unsigned int i = 0; i < safesize; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(VECGEOM_BACKEND_PRECISION_FROM_PTR(points.x() + i),
                                                        VECGEOM_BACKEND_PRECISION_FROM_PTR(points.y() + i),
                                                        VECGEOM_BACKEND_PRECISION_FROM_PTR(points.z() + i));
      VECGEOM_BACKEND_TYPE::precision_v estimate = VECGEOM_BACKEND_PRECISION_FROM_PTR(&safeties[i]);
      VECGEOM_BACKEND_TYPE::precision_v result   = kInfLength;
      Specialization::template SafetyToIn<VECGEOM_BACKEND_TYPE>(*this->GetUnplacedVolume(), *this->GetTransformation(),
                                                                point, result);
      vecCore::MaskedAssign(result, estimate < result, estimate);
      vecCore::Store(result, safeties + i);
    }
    unsigned tailsize = points.size() - safesize;
    for (unsigned int i = 0; i < tailsize; ++i) {
      unsigned int track = safesize + i;
      Precision result   = kInfLength;
      Specialization::template SafetyToIn<kScalar>(*this->GetUnplacedVolume(), *this->GetTransformation(),
                                                   points[track], result);
      safeties[track] = (result < safeties[track]) ? result : safeties[track];
    }
  }

  virtual void SafetyToOut(SOA3D<Precision> const &points, Precision *const output) const override
  {
    unsigned safesize = points.size() - points.size() % kVectorSize;
    for (unsigned int i = 0; i < safesize; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(VECGEOM_BACKEND_PRECISION_FROM_PTR(points.x() + i),
                                                        VECGEOM_BACKEND_PRECISION_FROM_PTR(points.y() + i),
                                                        VECGEOM_BACKEND_PRECISION_FROM_PTR(points.z() + i));
      VECGEOM_BACKEND_TYPE::precision_v result = kInfLength;
      Specialization::template SafetyToOut<VECGEOM_BACKEND_TYPE>(*this->GetUnplacedVolume(), point, result);
      vecCore::Store(result, output + i);
    }
    // tail treatment
    unsigned tailsize = points.size() - safesize;
    for (unsigned int i = 0; i < tailsize; ++i) {
      Precision result   = kInfLength;
      unsigned int track = safesize + i;
      Specialization::template SafetyToOut<kScalar>(*this->GetUnplacedVolume(), points[track], result);
      output[track] = result;
    }
  }

  virtual void SafetyToOutMinimize(SOA3D<Precision> const &points, Precision *const safeties) const override
  {
    for (int i = 0, iMax = points.size(); i < iMax; i += kVectorSize) {
      Vector3D<VECGEOM_BACKEND_TYPE::precision_v> point(VECGEOM_BACKEND_PRECISION_FROM_PTR(points.x() + i),
                                                        VECGEOM_BACKEND_PRECISION_FROM_PTR(points.y() + i),
                                                        VECGEOM_BACKEND_PRECISION_FROM_PTR(points.z() + i));
      VECGEOM_BACKEND_TYPE::precision_v estimate = VECGEOM_BACKEND_PRECISION_FROM_PTR(&safeties[i]);
      VECGEOM_BACKEND_TYPE::precision_v result   = kInfLength;
      Specialization::template SafetyToOut<VECGEOM_BACKEND_TYPE>(*this->GetUnplacedVolume(), point, result);
      vecCore::MaskedAssign(result, estimate < result, estimate);
      vecCore::Store(result, safeties + i);
    }
  }

}; // End class ShapeImplementationHelper
}
} // End global namespace

#endif // VECGEOM_VOLUMES_SHAPEIMPLEMENTATIONHELPER_H_

/// \file AlignedBase.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BASE_ALIGNEDBASE_H_
#define VECGEOM_BASE_ALIGNEDBASE_H_

#include "base/Global.h"
#ifdef VECGEOM_VC
#include <Vc/Vc>
#endif

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class AlignedBase; )
VECGEOM_DEVICE_DECLARE_CONV( AlignedBase )

inline namespace VECGEOM_IMPL_NAMESPACE {

#ifdef VECGEOM_VC
  // unfortunately the version macros have changed in Vc over time
  // so I am checking which one exist
#ifdef Vc_VERSION_NUMBER
#if Vc_VERSION_NUMBER >= Vc_VERSION_CHECK(0,99,71) & Vc_VERSION_NUMBER < Vc_VERSION_CHECK(1,0,79)
    class AlignedBase : public Vc::VectorAlignedBase<Vc::Vector<double> > {
#else
    class AlignedBase : public Vc::VectorAlignedBase {
#endif
#endif
#ifdef VC_VERSION_NUMBER
#if VC_VERSION_NUMBER >= VC_VERSION_CHECK(0,99,71)
    class AlignedBase : public Vc::VectorAlignedBase<Vc::Vector<double> > {
#else
    class AlignedBase : public Vc::VectorAlignedBase {
#endif
#endif
    public:
      virtual ~AlignedBase() {}
};
#elif !defined(VECGEOM_NVCC)
class AlignedBase {

public:

  VECGEOM_INLINE
  void *operator new(size_t size) {
    return _mm_malloc(size, kAlignmentBoundary);
  }

  VECGEOM_INLINE
  void *operator new(size_t, void *p) {
    return p;
  }

  VECGEOM_INLINE
  void *operator new[](size_t size) {
    return _mm_malloc(size, kAlignmentBoundary);
  }

  VECGEOM_INLINE
  void *operator new[](size_t , void *p) {
    return p;
  }
  
  VECGEOM_INLINE
  void operator delete(void *ptr, size_t) {
    _mm_free(ptr);
  }

  VECGEOM_INLINE
  void operator delete(void *, void *) {}

  VECGEOM_INLINE
  void operator delete[](void *ptr, size_t) {
    _mm_free(ptr);
  }

  VECGEOM_INLINE
  void operator delete[](void *, void *) {}

};
#else
class AlignedBase {};
#endif

} } // End global namespace

#endif // VECGEOM_BASE_ALIGNEDBASE_H_

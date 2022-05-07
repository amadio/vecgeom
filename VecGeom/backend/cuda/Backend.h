/// \file cuda/Backend.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_BACKEND_CUDABACKEND_H_
#define VECGEOM_BACKEND_CUDABACKEND_H_

#include "VecGeom/base/Config.h"
#include "VecGeom/base/Global.h"

#include "VecGeom/backend/scalar/Backend.h"
#include "VecGeom/backend/cuda/Interface.h"

namespace vecgeom {
#ifdef VECCORE_CUDA
inline
#endif
    namespace cuda {

struct kCuda {
  typedef int int_v;
  typedef Precision precision_v;
  typedef bool bool_v;
  typedef Inside_t inside_v;
  static constexpr precision_v kOne  = 1.0;
  static constexpr precision_v kZero = 0.0;
  const static bool_v kTrue          = true;
  const static bool_v kFalse         = false;
  // alternative typedefs ( might supercede above typedefs )
  typedef int Int_t;
  typedef Precision Double_t;
  typedef bool Bool_t;
  typedef int Index_t;
};

typedef kCuda::int_v CudaInt;
typedef kCuda::precision_v CudaPrecision;
typedef kCuda::bool_v CudaBool;

#if defined(VECGEOM_ENABLE_CUDA) && !defined(VECGEOM_BACKEND_TYPE)
constexpr size_t kVectorSize = 1;
#define VECGEOM_BACKEND_TYPE vecgeom::kScalar
#define VECGEOM_BACKEND_PRECISION_FROM_PTR(P) (*(P))
#define VECGEOM_BACKEND_PRECISION_TYPE Precision
#define VECGEOM_BACKEND_PRECISION_TYPE_SIZE 1
//#define VECGEOM_BACKEND_PRECISION_NOT_SCALAR
#define VECGEOM_BACKEND_BOOL vecgeom::ScalarBool
#define VECGEOM_BACKEND_INSIDE vecgeom::kScalar::inside_v
#endif

static const unsigned kThreadsPerBlock = 256;

// Auxiliary GPU functions
#ifdef VECCORE_CUDA

VECCORE_ATT_DEVICE
VECGEOM_FORCE_INLINE
int ThreadIndex()
{
  return blockDim.x * blockIdx.x + threadIdx.x;
}

VECCORE_ATT_DEVICE
VECGEOM_FORCE_INLINE
int ThreadOffset()
{
  return blockDim.x * gridDim.x;
}

#endif

/**
 * Initialize with the number of threads required to construct the necessary
 * block and grid dimensions to accommodate all threads.
 */
struct LaunchParameters {
  dim3 block_size;
  dim3 grid_size;
  LaunchParameters(const unsigned threads)
  {
    // Blocks always one dimensional
    block_size.x                                 = kThreadsPerBlock;
    if (threads < kThreadsPerBlock) block_size.x = threads;
    block_size.y                                 = 1;
    block_size.z                                 = 1;
    // Grid becomes two dimensional at large sizes
    const unsigned blocks = 1 + (threads - 1) / kThreadsPerBlock;
    grid_size.z           = 1;
    if (blocks <= 1 << 16) {
      grid_size.x = blocks;
      grid_size.y = 1;
    } else {
      int dim     = static_cast<int>(sqrt(static_cast<double>(blocks)) + 0.5);
      grid_size.x = dim;
      grid_size.y = dim;
    }
  }
};
}
} // End global namespace

#endif // VECGEOM_BACKEND_CUDABACKEND_H_

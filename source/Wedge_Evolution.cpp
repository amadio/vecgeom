/*
 * Wedge.cpp
 *
 *  Created on: 28.03.2015
 *      Author: swenzel
 */
#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/Wedge_Evolution.h"
#include <iostream>
#include <iomanip>

namespace vecgeom {
namespace evolution {
inline namespace VECGEOM_IMPL_NAMESPACE {

VECCORE_ATT_HOST_DEVICE
Wedge::Wedge(Precision angle, Precision zeroangle /* = 0 */) : fDPhi(angle), fAlongVector1(), fAlongVector2()
{
  // Note: This routine is outlined to work around a deficiency in NVCC's device code last checked
  // with nvcc 9.1.85, gcc 6.4.0 CentOS Linux release 7.4.1708.
  // When the constructor is included in the header file, the device crash during the
  // execution of the function cos as long as it takes input related to zeroangle
  // eg. any of this variation will crash:
  //
  // The crash stack trace is:
  // #0  0x0000000002c32e00 in __internal_trig_reduction_slowpathd ()
  // #1  0x0000000002c616e0 in cos ()
  // #2  0x0000000002c94168 in vecgeom::evolution::cuda::Wedge::Wedge (this=<optimized out>, angle=<optimized out>,
  // zeroangle=<optimized out>)
  //     at
  //     /data/sftnight/workspace/VecGeom-gitlab-CUDA_2/BUILDTYPE/Debug/COMPILER/gcc49/LABEL/continuos-cc7/VecGeom/volumes/Wedge_Evolution.h:84
  // #3  0x0000000002d9d768 in vecgeom::cuda::TubeStruct<double>::TubeStruct (this=<optimized out>, _rmin=<optimized
  // out>, _rmax=<optimized out>, _z=<optimized out>, _sphi=<optimized out>, _dphi=<optimized out>)
  //     at
  //     /data/sftnight/workspace/VecGeom-gitlab-CUDA_2/BUILDTYPE/Debug/COMPILER/gcc49/LABEL/continuos-cc7/VecGeom/volumes/TubeStruct.h:166
  // #4  0x0000000003c04990 in vecgeom::cuda::CutTubeStruct<double>::CutTubeStruct (this=<optimized out>,
  // rmin=<optimized out>, rmax=<optimized out>, z=<optimized out>, sphi=0.35355339059327384, dphi=<optimized out>,
  // bottomNormal=..., topNormal=...)
  //     at
  //     /data/sftnight/workspace/VecGeom-gitlab-CUDA_2/BUILDTYPE/Debug/COMPILER/gcc49/LABEL/continuos-cc7/VecGeom/volumes/CutTubeStruct.h:40
  // #5  0x0000000003c00970 in vecgeom::cuda::UnplacedCutTube::UnplacedCutTube (this=<optimized out>, rmin=<optimized
  // out>, rmax=<optimized out>, z=<optimized out>, sphi=<optimized out>, dphi=<optimized out>, bx=<optimized out>,
  // by=<optimized out>,
  //     bz=<optimized out>, tx=<optimized out>, ty=<optimized out>, tz=<optimized out>) at
  //     /data/sftnight/workspace/VecGeom-gitlab-CUDA_2/BUILDTYPE/Debug/COMPILER/gcc49/LABEL/continuos-cc7/VecGeom/volumes/UnplacedCutTube.h:55
  // #6  0x0000000003bfc758 in vecgeom::cuda::ConstructOnGpu<vecgeom::cuda::UnplacedCutTube, double, double, double,
  // double, double, double, double, double, double, double, double><<<(1,1,1),(1,1,1)>>> (gpu_ptr=0x4201700200,
  // params=3.3155729693438206e-316,
  //     params=3.3155729693438206e-316, params=3.3155729693438206e-316, params=3.3155729693438206e-316,
  //     params=3.3155729693438206e-316, params=3.3155729693438206e-316, params=3.3155729693438206e-316,
  //     params=3.3155729693438206e-316,
  //     params=3.3155729693438206e-316, params=3.3155729693438206e-316, params=3.3155729693438206e-316) at
  //     /data/sftnight/workspace/VecGeom-gitlab-CUDA_2/BUILDTYPE/Debug/COMPILER/gcc49/LABEL/continuos-cc7/VecGeom/backend/cuda/Interface.h:25
  //
  // cuda-memcheck says:
  //
  // ========= Invalid __global__ read of size 8
  // =========     at 0x000004f0 in __internal_trig_reduction_slowpathd
  // =========     by thread (0,0,0) in block (0,0,0)
  // =========     Address 0x41e5300350 is out of bounds
  // =========     Device Frame:cos (cos : 0x460)
  // =========     Device Frame:vecgeom::evolution::cuda::Wedge::__complete_object_constructor__(double, double)
  // (vecgeom::evolution::cuda::Wedge::__complete_object_constructor__(double, double) : 0x6e8)
  // =========     Device Frame:vecgeom::cuda::TubeStruct<double>::__complete_object_constructor__(double, double,
  // double, double, double) (vecgeom::cuda::TubeStruct<double>::__complete_object_constructor__(double, double, double,
  // double, double) : 0x7a8)
  // =========     Device Frame:vecgeom::cuda::CutTubeStruct<double>::__complete_object_constructor__(double, double,
  // double, double, double, vecgeom::cuda::Vector3D<double>, vecgeom::cuda::Vector3D<double>)
  // (vecgeom::cuda::CutTubeStruct<double>::__complete_object_constructor__(double, double, double, double, double,
  // vecgeom::cuda::Vector3D<double>, vecgeom::cuda::Vector3D<double>) : 0x610)
  // =========     Device Frame:vecgeom::cuda::UnplacedCutTube::__complete_object_constructor__(double const &, double
  // const &, double const &, double const &, double const &, double const &, double const &, double const &, double
  // const &, double const &, double const &) (vecgeom::cuda::UnplacedCutTube::__complete_object_constructor__(double
  // const &, double const &, double const &, double const &, double const &, double const &, double const &, double
  // const &, double const &, double const &, double const &) : 0xb70)
  // =========     Device Frame:_ZN7vecgeom4cuda14ConstructOnGpuINS0_15UnplacedCutTubeEJdddddddddddEEEvPT_DpT0_
  // (_ZN7vecgeom4cuda14ConstructOnGpuINS0_15UnplacedCutTubeEJdddddddddddEEEvPT_DpT0_ : 0x818)
  // =========     Saved host backtrace up to driver entry point at kernel launch time
  // =========     Host Frame:/lib64/libcuda.so.1 (cuLaunchKernel + 0x2cd) [0x23c06d]
  // =========     Host Frame:/usr/local/cuda/lib64/libcudart.so.9.1 [0x15f70]
  // =========     Host Frame:/usr/local/cuda/lib64/libcudart.so.9.1 (cudaLaunch + 0x14e) [0x347be]
  // =========     Host
  // Frame:/data/sftnight/workspace/VecGeom-gitlab-CUDA_2/BUILDTYPE/Debug/COMPILER/gcc49/LABEL/continuos-cc7/build.gcc.6.4.0/libvecgeomcuda.so
  // [0x170964]
  // =========     Host
  // Frame:/data/sftnight/workspace/VecGeom-gitlab-CUDA_2/BUILDTYPE/Debug/COMPILER/gcc49/LABEL/continuos-cc7/build.gcc.6.4.0/libvecgeomcuda.so
  // [0x170388]
  // =========     Host
  // Frame:/data/sftnight/workspace/VecGeom-gitlab-CUDA_2/BUILDTYPE/Debug/COMPILER/gcc49/LABEL/continuos-cc7/build.gcc.6.4.0/libvecgeomcuda.so
  // [0x170463]
  // =========     Host
  // Frame:/data/sftnight/workspace/VecGeom-gitlab-CUDA_2/BUILDTYPE/Debug/COMPILER/gcc49/LABEL/continuos-cc7/build.gcc.6.4.0/libvecgeomcuda.so
  // (_ZN7vecgeom4cuda14ConstructOnGpuINS0_15UnplacedCutTubeEJdddddddddddEEEvPT_DpT0_ + 0x72) [0x1728d6]
  // =========     Host
  // Frame:/data/sftnight/workspace/VecGeom-gitlab-CUDA_2/BUILDTYPE/Debug/COMPILER/gcc49/LABEL/continuos-cc7/build.gcc.6.4.0/libvecgeomcuda.so
  // (_ZNK7vecgeom3cxx9DevicePtrINS_4cuda15UnplacedCutTubeEE9ConstructIJdddddddddddEEEvDpT_ + 0xfe) [0x1726fa]
  // =========     Host Frame:./CutTubeBenchmark
  // (_ZNK7vecgeom3cxx15VUnplacedVolume13CopyToGpuImplINS0_15UnplacedCutTubeEJdddddddddddEEENS0_9DevicePtrINS_4cuda15VUnplacedVolumeEEES7_DpT0_
  // + 0xb1) [0x10170f]
  // =========     Host Frame:./CutTubeBenchmark
  // (_ZNK7vecgeom3cxx15UnplacedCutTube9CopyToGpuENS0_9DevicePtrINS_4cuda15VUnplacedVolumeEEE + 0x24e) [0xfaa6a]
  // =========     Host Frame:./CutTubeBenchmark (_ZN7vecgeom3cxx11CudaManager11SynchronizeEv + 0x3c9) [0x126c5d]
  // =========     Host Frame:./CutTubeBenchmark
  // (_ZN7vecgeom11Benchmarker17GetVolumePointersERNSt7__cxx114listINS_3cxx9DevicePtrINS_4cuda13VPlacedVolumeEEESaIS7_EEE
  // + 0x47) [0x11d09d]
  // =========     Host
  // Frame:/data/sftnight/workspace/VecGeom-gitlab-CUDA_2/BUILDTYPE/Debug/COMPILER/gcc49/LABEL/continuos-cc7/build.gcc.6.4.0/libvecgeomcuda.so
  // (_ZN7vecgeom11Benchmarker13RunInsideCudaEPdS1_S1_PbPi + 0x81) [0x1c4a61]
  // =========     Host Frame:./CutTubeBenchmark (_ZN7vecgeom11Benchmarker18RunInsideBenchmarkEv + 0x450) [0x114d12]
  // =========     Host Frame:./CutTubeBenchmark (_ZN7vecgeom11Benchmarker12RunBenchmarkEv + 0x4a) [0x114854]
  // =========     Host Frame:./CutTubeBenchmark (_Z9benchmarkdddddddddii + 0x3ce) [0x83392]
  // =========     Host Frame:./CutTubeBenchmark (main + 0x687) [0x83b45]
  // =========     Host Frame:/lib64/libc.so.6 (__libc_start_main + 0xf5) [0x21c05]
  // =========     Host Frame:./CutTubeBenchmark [0x81db9]

  // check input
  // assert(angle > 0.0 && angle <= kTwoPi);

  // initialize angles
  fAlongVector1.x() = std::cos(zeroangle);
  fAlongVector1.y() = std::sin(zeroangle);
  fAlongVector2.x() = std::cos(zeroangle + angle);
  fAlongVector2.y() = std::sin(zeroangle + angle);

  fNormalVector1.x() = -std::sin(zeroangle);
  fNormalVector1.y() = std::cos(zeroangle); // not the + sign
  fNormalVector2.x() = std::sin(zeroangle + angle);
  fNormalVector2.y() = -std::cos(zeroangle + angle); // note the - sign
}

} // VECGEOM_IMPL_NAMESPACE
} // namespace evolution
} // namespace vecgeom

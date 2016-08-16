///===-- volumes/UnplacedTrapezoid.h ----------------------------------*- C++ -*-===//
///
/// \file volumes/UnplacedTrapezoid.h
/// \author Guilherme Lima (lima@fnal.gov)
/// \brief This file contains the declaration of the UnplacedTrapezoid class
///
/// _____________________________________________________________________________
/// A trapezoid is the solid bounded by the following surfaces:
/// - 2 XY-parallel quadrilaterals (trapezoids) cutting the Z axis at Z=-dz and Z=+dz
/// - 4 additional *planar* quadrilaterals intersecting the faces at +/-dz at their edges
///
/// The easiest way to think about the trapezoid is starting from a box with faces at +/-dz,
/// and moving its eight corners in such a way as to define the two +/-dz trapezoids, without
/// destroying the coplanarity of the other four faces.  Then the (x,y,z) coordinates of the
/// eight corners can be used to build this trapezoid.
///
/// If the four side faces are not coplanar, a generic trapezoid must be used (GenTrap).
//===------------------------------------------------------------------------===//
///
/// 140520 G. Lima   Created based on USolids algorithms and vectorized types
/// 160722 G. Lima   Migration to new helpers and VecCore-based scheme

#ifndef VECGEOM_VOLUMES_UNPLACEDTRAPEZOID_H_
#define VECGEOM_VOLUMES_UNPLACEDTRAPEZOID_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedVolumeImplHelper.h"

#include "volumes/TrapezoidStruct.h" // the pure Trapezoid struct
#include "volumes/kernel/TrapezoidImplementation.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedTrapezoid;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedTrapezoid);

inline namespace VECGEOM_IMPL_NAMESPACE {

typedef Vector3D<Precision> TrapCorners[8];

class UnplacedTrapezoid : public SIMDUnplacedVolumeImplHelper<TrapezoidImplementation>, public AlignedBase {

private:
  TrapezoidStruct<Precision> fTrap;

  // variables to store cached values for Capacity and SurfaceArea
  // Precision fCubicVolume, fSurfaceArea;

public:
  // full constructor
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrapezoid(const Precision dz, const Precision theta, const Precision phi, const Precision dy1,
                    const Precision dx1, const Precision dx2, const Precision tanAlpha1, const Precision dy2,
                    const Precision dx3, const Precision dx4, const Precision tanAlpha2)
      : fTrap(dz, theta, phi, dy1, dx1, dx2, tanAlpha1, dy2, dx3, dx4, tanAlpha2)
  {
    fGlobalConvexity = true;
    MakePlanes();
  }

  // default constructor
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrapezoid() : fTrap(0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.) { fGlobalConvexity = true; }

  /// \brief Fast constructor: all parameters from one array
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrapezoid(Precision const *params)
      : UnplacedTrapezoid(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7],
                          params[8], params[9], params[10])
  {
  }

  /// \brief Constructor based on 8 corner points
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrapezoid(TrapCorners const corners);

  /// \brief Constructor for masquerading a box (test purposes)
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrapezoid(Precision xbox, Precision ybox, Precision zbox);

  /// \brief Constructor required by Geant4
  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrapezoid(double dx, double dy, double dz, double);

  /// \brief Constructor for a Trd-like trapezoid
  UnplacedTrapezoid(double dx1, double dx2, double dy1, double dy2, double dz)
      : UnplacedTrapezoid(dz, 0., 0., dy1, dx1, dx2, 0., dy2, dx1, dx2, 0.)
  {
  }

  /// \brief Constructor for a Parallelepiped-like trapezoid (Note: still to be validated)
  UnplacedTrapezoid(double dx, double dy, double dz, double alpha, double theta, double phi)
      : fTrap(dz, theta, phi, dy, dx, dx, 0., dy, dx, dx, 0.)
  {
    // TODO: validate alpha usage here
    fTrap.fTanAlpha1 = std::tan(alpha);
    fTrap.fTanAlpha2 = fTrap.fTanAlpha1;
    fGlobalConvexity = true;
    MakePlanes();
  }

  /// \brief Accessors
  /// @{
  // VECGEOM_CUDA_HEADER_BOTH
  // TrapParameters const& GetParameters() const { return _params; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision dz() const { return fTrap.fDz; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision theta() const { return fTrap.fTheta; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision phi() const { return fTrap.fPhi; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision dy1() const { return fTrap.fDy1; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision dy2() const { return fTrap.fDy2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision dx1() const { return fTrap.fDx1; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision dx2() const { return fTrap.fDx2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision dx3() const { return fTrap.fDx3; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision dx4() const { return fTrap.fDx4; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision tanAlpha1() const { return fTrap.fTanAlpha1; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision tanAlpha2() const { return fTrap.fTanAlpha2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision alpha1() const { return GetAlpha1(); } // note: slow, avoid using it

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision alpha2() const { return GetAlpha2(); } // note: slow, avoid using it

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision tanThetaCosPhi() const { return fTrap.fTthetaCphi; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision tanThetaSinPhi() const { return fTrap.fTthetaSphi; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision GetDz() const { return fTrap.fDz; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision GetTheta() const { return fTrap.fTheta; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision GetPhi() const { return fTrap.fPhi; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision GetDy1() const { return fTrap.fDy1; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision GetDx1() const { return fTrap.fDx1; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision GetDx2() const { return fTrap.fDx2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision GetTanAlpha1() const { return fTrap.fTanAlpha1; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision GetDy2() const { return fTrap.fDy2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision GetDx3() const { return fTrap.fDx3; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision GetDx4() const { return fTrap.fDx4; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision GetTanAlpha2() const { return fTrap.fTanAlpha2; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision GetTanThetaSinPhi() const { return fTrap.fTthetaSphi; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision GetTanThetaCosPhi() const { return fTrap.fTthetaCphi; }

  /// @}

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  void SetDz(Precision val) { fTrap.fDz = val; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  void SetTheta(Precision val)
  {
    fTrap.fTheta = val;
    fTrap.CalculateCached();
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  void SetPhi(Precision val)
  {
    fTrap.fPhi = val;
    fTrap.CalculateCached();
  }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  void SetDy1(Precision val) { fTrap.fDy1 = val; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  void SetDy2(Precision val) { fTrap.fDy2 = val; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  void SetDx1(Precision val) { fTrap.fDx1 = val; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  void SetDx2(Precision val) { fTrap.fDx2 = val; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  void SetDx3(Precision val) { fTrap.fDx3 = val; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  void SetDx4(Precision val) { fTrap.fDx4 = val; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  void SetTanAlpha1(Precision val) { fTrap.fTanAlpha1 = val; }

  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  void SetTanAlpha2(Precision val) { fTrap.fTanAlpha2 = val; }

  VECGEOM_CUDA_HEADER_BOTH
  void Extent(Vector3D<Precision> &, Vector3D<Precision> &) const override;

  // VECGEOM_CUDA_HEADER_BOTH
  // void CalcCapacity();

  // VECGEOM_CUDA_HEADER_BOTH
  // void CalcSurfaceArea();

  // VECGEOM_CUDA_HEADER_BOTH
  Precision Capacity() const;
  // {
  //   Assert(!fOutdated);
  //   return fCubicVolume;
  // }

  // VECGEOM_CUDA_HEADER_BOTH
  Precision SurfaceArea() const;
  // {
  //   Assert(!fOutdated);
  //   return fSurfaceArea;
  // }

  Vector3D<Precision> GetPointOnSurface() const override;

  Vector3D<Precision> GetPointOnPlane(Vector3D<Precision> const &p0, Vector3D<Precision> const &p1,
                                      Vector3D<Precision> const &p2, Vector3D<Precision> const &p3) const;

  VECGEOM_CUDA_HEADER_BOTH
  virtual bool Normal(Vector3D<Precision> const &point, Vector3D<Precision> &normal) const override
  {
    bool valid = false;
    normal     = TrapezoidImplementation::NormalKernel(fTrap, point, valid);
    return valid;
  }

  std::string GetEntityType() const { return "Trapezoid"; }

#if defined(VECGEOM_USOLIDS)
  template <typename T>
  VECGEOM_CUDA_HEADER_BOTH
  void GetParametersList(int aNumber, T *aArray) const;

  VECGEOM_CUDA_HEADER_BOTH
  UnplacedTrapezoid *Clone() const;

  std::ostream &StreamInfo(std::ostream &os) const;
#endif

public:
  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const override;

  virtual void Print(std::ostream &os) const override;

#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const override { return DevicePtr<cuda::UnplacedTrapezoid>::SizeOf(); }

  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const override;

  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const override;
#endif

#ifndef VECGEOM_NVCC
  // this is the function called from the VolumeFactory, it may be specific to the trapezoid
  template <TranslationCode trans_code, RotationCode rot_code>
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               VPlacedVolume *const placement = NULL);

  VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume, Transformation3D const *const transformation,
                                   const TranslationCode trans_code, const RotationCode rot_code,
                                   VPlacedVolume *const placement) const override;

#else

  template <TranslationCode trans_code, RotationCode rot_code>
  __device__
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
                               const int id, VPlacedVolume *const placement = NULL);

  __device__ VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                              Transformation3D const *const transformation,
                                              const TranslationCode trans_code, const RotationCode rot_code,
                                              const int id, VPlacedVolume *const placement) const override;
#endif

  // Note: use of ATan() makes this one slow -- to be avoided
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision GetAlpha1() const { return vecCore::math::ATan(fTrap.fTanAlpha1); }

  // Note: use of Atan() makes this one slow -- to be avoided
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_FORCE_INLINE
  Precision GetAlpha2() const { return vecCore::math::ATan(fTrap.fTanAlpha2); }

  // The next functions force upon the user insider knowledge about how the side planes should be used
  VECGEOM_CUDA_HEADER_BOTH
  TrapezoidStruct<Precision> const &GetStruct() const { return fTrap; }

  // #ifndef VECGEOM_PLANESHELL_DISABLE
  //   VECGEOM_CUDA_HEADER_BOTH
  //   VECGEOM_FORCE_INLINE
  //   PlaneShell<4,Precision> const *GetPlanes() const { return fTrap.GetPlanes(); }

  // #else
  //   using TrapSidePlane = TrapezoidStruct<double>::TrapSidePlane;
  //   VECGEOM_CUDA_HEADER_BOTH
  //   TrapSidePlane const *GetPlanes() const { return fTrap.GetPlanes(); }
  // #endif

private:
  /// \brief Calculate trapezoid parameters when user provides the 8 corners
  VECGEOM_CUDA_HEADER_BOTH
  void fromCornersToParameters(TrapCorners const pt);

  /// \brief Calculate the 8 corner points using pre-stored parameters
  VECGEOM_CUDA_HEADER_BOTH
  void fromParametersToCorners(TrapCorners pt) const;

  /// \brief Construct the four side planes from input corner points
  VECGEOM_CUDA_HEADER_BOTH
  bool MakePlanes(TrapCorners const corners);

  /// \brief Construct the four side planes by converting stored parameters into TrapCorners object
  VECGEOM_CUDA_HEADER_BOTH
  bool MakePlanes();

/// \brief Construct the four side planes from input corner points
#ifndef VECGEOM_PLANESHELL_DISABLE
  VECGEOM_CUDA_HEADER_BOTH
  bool MakeAPlane(Vector3D<Precision> const &p1, Vector3D<Precision> const &p2, Vector3D<Precision> const &p3,
                  Vector3D<Precision> const &p4, unsigned int iplane);
#else
  VECGEOM_CUDA_HEADER_BOTH
  bool MakeAPlane(Vector3D<Precision> const &p1, Vector3D<Precision> const &p2, Vector3D<Precision> const &p3,
                  Vector3D<Precision> const &p4, TrapezoidStruct<double>::TrapSidePlane &plane);
#endif
};

} // inline NS
} // vecgeom NS

#endif

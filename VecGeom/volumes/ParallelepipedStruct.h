// This file is part of VecGeom and is distributed under the
// conditions in the file LICENSE.txt in the top directory.
// For the full list of authors see CONTRIBUTORS.txt and `git log`.

/// Declaration of a struct with data members for the UnplacedParallelepiped class
/// @file volumes/ParallelepipedStruct.h
/// @author First version created by Mihaela Gheata

#ifndef VECGEOM_VOLUMES_PARALLELEPIPEDSTRUCT_H_
#define VECGEOM_VOLUMES_PARALLELEPIPEDSTRUCT_H_
#include "VecGeom/base/Global.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

/// Struct encapsulating data members of the unplaced parallelepiped
template <typename T = double>
struct ParallelepipedStruct {
  Vector3D<T> fDimensions; ///< Dimensions dx, dy, dx
  T fAlpha;                ///< Angle dx versus dy
  T fTheta;                ///< Theta angle of parallelepiped axis
  T fPhi;                  ///< Phi angle of parallelepiped axis
  T fCtx;                  ///< Scale factor for safety distances in X
  T fCty;                  ///< Scale factor for safety distances in Y
  T fAreas[3];             ///< Facet areas
  Vector3D<T> fNormals[3]; ///< Precomputed normals

  // Precomputed values computed from parameters
  T fTanAlpha;       ///< Tangent of alpha angle
  T fTanThetaSinPhi; ///< tan(theta)*sin(phi)
  T fTanThetaCosPhi; ///< tan(theta)*cos(phi)
  T fCosTheta;       ///< cos(theta)

  /// Constructor from a vector of dimensions and three angles
  /// @param dim 3D vector with dx, dy, dz
  /// @param alpha Angle between y-axis and the line joining centres of the faces at +/- dy
  /// @param theta Polar angle
  /// @param phi Azimuthal angle
  VECCORE_ATT_HOST_DEVICE
  ParallelepipedStruct(Vector3D<T> const &dim, const T alpha, const T theta, const T phi)
      : fDimensions(dim), fAlpha(0), fTheta(0), fPhi(0), fCtx(0), fCty(0), fTanAlpha(0), fTanThetaSinPhi(0),
        fTanThetaCosPhi(0)
  {
    SetAlpha(alpha);
    SetThetaAndPhi(theta, phi);
  }

  /// Constructor from three dimensions and three angles
  /// @param dx Half length in x
  /// @param dy Half length in y
  /// @param dz Half length in z
  /// @param alpha Angle between y-axis and the line joining centres of the faces at +/- dy
  /// @param theta Polar angle
  /// @param phi Azimuthal angle
  VECCORE_ATT_HOST_DEVICE
  ParallelepipedStruct(const T dx, const T dy, const T dz, const T alpha, const T theta, const T phi)
      : fDimensions(dx, dy, dz), fAlpha(0), fTheta(0), fPhi(0), fCtx(0), fCty(0), fTanAlpha(0), fTanThetaSinPhi(0),
        fTanThetaCosPhi(0)
  {
    SetAlpha(alpha);
    SetThetaAndPhi(theta, phi);
  }

  /// Setter for alpha angle
  /// @param alpha angle between Y and the axis of symmetry of the base
  VECCORE_ATT_HOST_DEVICE
  void SetAlpha(const T alpha)
  {
    fAlpha    = alpha;
    fTanAlpha = tan(alpha);
    ComputeNormals();
  }

  /// Setter for theta angle
  /// @param theta Polar angle
  VECCORE_ATT_HOST_DEVICE
  void SetTheta(const T theta) { SetThetaAndPhi(theta, fPhi); }

  /// Setter for phi angle
  /// @param phi Azimuthal angle
  VECCORE_ATT_HOST_DEVICE
  void SetPhi(const T phi) { SetThetaAndPhi(fTheta, phi); }

  /// Setter for theta and phi
  /// @param theta Polar angle
  /// @param phi Azimuthal angle
  VECCORE_ATT_HOST_DEVICE
  void SetThetaAndPhi(const T theta, const T phi)
  {
    fTheta          = theta;
    fPhi            = phi;
    fTanThetaCosPhi = vecCore::math::Tan(fTheta) * vecCore::math::Cos(fPhi);
    fTanThetaSinPhi = vecCore::math::Tan(fTheta) * vecCore::math::Sin(fPhi);
    fCosTheta       = vecCore::math::Cos(fTheta);
    ComputeNormals();
  }

  /// Compute auxiliary data members: normals, areas, scale factors
  VECCORE_ATT_HOST_DEVICE
  void ComputeNormals()
  {
    Vector3D<T> vx(1., 0., 0.);
    Vector3D<T> vy(fTanAlpha, 1., 0.);
    Vector3D<T> vz(fTanThetaCosPhi, fTanThetaSinPhi, 1.);
    fNormals[0] = vy.Cross(vz);
    fNormals[1] = vz.Cross(vx);
    fNormals[2].Set(0., 0., 1.);
    fAreas[0] = 4. * fDimensions.y() * fDimensions.z() * fNormals[0].Mag();
    fAreas[1] = 4. * fDimensions.z() * fDimensions.x() * fNormals[1].Mag();
    fAreas[2] = 4. * fDimensions.x() * fDimensions.y();
    fNormals[0].Normalize();
    fNormals[1].Normalize();
    fCtx = vecCore::math::Abs(fNormals[0].x());
    fCty = vecCore::math::Abs(fNormals[1].y());
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif

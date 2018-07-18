/*
 * TetStruct.h
 *
 *  Created on: 10.07.2018
 *      Author: Raman Sehgal, Evgueni Tcherniaev
 */

#ifndef VECGEOM_VOLUMES_TETSTRUCT_H_
#define VECGEOM_VOLUMES_TETSTRUCT_H_
#include "base/Global.h"
#include "base/Vector3D.h"

namespace vecgeom {

inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename T = double>
struct TetStruct {

  Vector3D<T> fVertex[4];
  struct {
    Vector3D<T> n;
    T d;
  } fPlane[4];

  /* Add whatever data member you want to be cached
  ** like Volume and Surface Area, etc..
  */
  Precision fCubicVolume, fSurfaceArea;

  VECCORE_ATT_HOST_DEVICE
  TetStruct() {}

  VECCORE_ATT_HOST_DEVICE
  TetStruct(const T p0[], const T p1[], const T p2[], const T p3[])
  {
    Vector3D<T> vertices[4];
    vertices[0].Set(p0[0], p0[1], p0[2]);
    vertices[1].Set(p1[0], p1[1], p1[2]);
    vertices[2].Set(p2[0], p2[1], p2[2]);
    vertices[3].Set(p3[0], p3[1], p3[2]);

    CalculateCached(vertices[0], vertices[1], vertices[2], vertices[3]);
  }

  VECCORE_ATT_HOST_DEVICE
  TetStruct(const Vector3D<T> p0, const Vector3D<T> p1, const Vector3D<T> p2, const Vector3D<T> p3)
      : fCubicVolume(0.), fSurfaceArea(0.)
  {

    CalculateCached(p0, p1, p2, p3);
  }

  VECCORE_ATT_HOST_DEVICE
  void CalculateCached(const Vector3D<T> p0, const Vector3D<T> p1, const Vector3D<T> p2, const Vector3D<T> p3)
  {
    // Fill all the cached values
    fVertex[0] = p0;
    fVertex[1] = p1;
    fVertex[2] = p2;
    fVertex[3] = p3;

    // if (CheckDegeneracy()) std::cout << "DeGenerate Tetrahedron not allowed" << std::endl;
    CheckDegeneracy();

    Vector3D<Precision> n0 = (fVertex[1] - fVertex[0]).Cross(fVertex[2] - fVertex[0]).Unit();
    Vector3D<Precision> n1 = (fVertex[2] - fVertex[1]).Cross(fVertex[3] - fVertex[1]).Unit();
    Vector3D<Precision> n2 = (fVertex[3] - fVertex[2]).Cross(fVertex[0] - fVertex[2]).Unit();
    Vector3D<Precision> n3 = (fVertex[0] - fVertex[3]).Cross(fVertex[1] - fVertex[3]).Unit();

    if (n0.Dot(fVertex[3] - fVertex[0]) > 0) n0 = -n0;
    if (n1.Dot(fVertex[0] - fVertex[1]) > 0) n1 = -n1;
    if (n2.Dot(fVertex[1] - fVertex[2]) > 0) n2 = -n2;
    if (n3.Dot(fVertex[2] - fVertex[3]) > 0) n3 = -n3;

    fPlane[0].n = n0;
    fPlane[0].d = -n0.Dot(fVertex[0]);
    // fPlane[0].d = n0.Dot(fVertex[0]);

    fPlane[1].n = n1;
    fPlane[1].d = -n1.Dot(fVertex[1]);
    // fPlane[1].d = n1.Dot(fVertex[1]);

    fPlane[2].n = n2;
    fPlane[2].d = -n2.Dot(fVertex[2]);
    // fPlane[2].d = n2.Dot(fVertex[2]);

    fPlane[3].n = n3;
    fPlane[3].d = -n3.Dot(fVertex[3]);
    // fPlane[3].d = n3.Dot(fVertex[3]);

    for (int i = 0; i < 4; i++) {
      // std::cout << "Plane[" << i << "] = " << fPlane[i].n << "  " << fPlane[i].d << std::endl;
    }
  }

  VECCORE_ATT_HOST_DEVICE
  void CalcCapacity()
  {
    // fCubicVolume = <Logic to calculate Capacity>
    fCubicVolume =
        vecCore::math::Abs((fVertex[1] - fVertex[0]).Dot((fVertex[2] - fVertex[0]).Cross(fVertex[3] - fVertex[0]))) /
        6.;
  }

  VECCORE_ATT_HOST_DEVICE
  void CalcSurfaceArea()
  {
    // fSurfaceArea = <Logic to calculate SurfaceArea>
    fSurfaceArea = ((fVertex[1] - fVertex[0]).Cross(fVertex[2] - fVertex[0]).Mag() +
                    (fVertex[2] - fVertex[1]).Cross(fVertex[3] - fVertex[1]).Mag() +
                    (fVertex[3] - fVertex[2]).Cross(fVertex[0] - fVertex[2]).Mag() +
                    (fVertex[0] - fVertex[3]).Cross(fVertex[1] - fVertex[3]).Mag()) *
                   0.5;
  }

  // Function to Set the parameters
  VECCORE_ATT_HOST_DEVICE
  void SetParameters(const Vector3D<T> p0, const Vector3D<T> p1, const Vector3D<T> p2, const Vector3D<T> p3)
  {
    CalculateCached(p0, p1, p2, p3);
  }

  VECCORE_ATT_HOST_DEVICE
  bool CheckDegeneracy()
  {
    CalcCapacity();
    CalcSurfaceArea();
    if (fCubicVolume < kTolerance * kTolerance * kTolerance) return true;
    Precision s0 = (fVertex[1] - fVertex[0]).Cross(fVertex[2] - fVertex[0]).Mag() * 0.5;
    Precision s1 = (fVertex[2] - fVertex[1]).Cross(fVertex[3] - fVertex[1]).Mag() * 0.5;
    Precision s2 = (fVertex[3] - fVertex[2]).Cross(fVertex[0] - fVertex[2]).Mag() * 0.5;
    Precision s3 = (fVertex[0] - fVertex[3]).Cross(fVertex[1] - fVertex[3]).Mag() * 0.5;
    return (vecCore::math::Max(vecCore::math::Max(vecCore::math::Max(s0, s1), s2), s3) < 2 * kTolerance);
  }
};

} /* end of IMPL namespace */
} /* end of vecgeom namespace */

#endif

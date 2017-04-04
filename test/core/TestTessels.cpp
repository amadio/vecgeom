#include <VecCore/VecCore>

#include "base/Global.h"
#include "base/Vector3D.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

//* Simple test demonstrating that we can create/use a structure having SIMD data members
//  to represent the tesselated cluster of triangles. Tested features:
//  - discovery of vector size, Real_v type
//  - filling the Real_v data members from scalar triangle data
//  - Vectorizing scalar queries using multiplexing of the input data to Real_v types, then
//    backend vector operations
constexpr size_t kVecSize = vecCore::VectorSize<vecgeom::VectorBackend::Real_v>();

template <typename Real_v>
struct TessellatedStruct {
  vecgeom::Vector3D<Real_v> fNormals;
  vecgeom::Vector3D<double> fNormal[kVecSize];

  void AddNormal(size_t i, vecgeom::Vector3D<double> const &normal)
  {
    using vecCore::FromPtr;
    assert(i < kVecSize);
    fNormal[i] = normal;
    std::cout << "normal[" << i << "] = " << fNormal[i] << std::endl;

    fNormals.x()[i] = normal.x();
    fNormals.y()[i] = normal.y();
    fNormals.z()[i] = normal.z();

    double arrayx[kVecSize];
    double arrayy[kVecSize];
    double arrayz[kVecSize];
    if (i == (kVecSize - 1)) {
      /*
            for (size_t j = 0; j < kVecSize; ++j) {
              std::cout << "normal[" << j << "] = " << fNormal[j] << std::endl;
              arrayx[j] = fNormal[j].x();
              arrayy[j] = fNormal[j].y();
              arrayz[j] = fNormal[j].z();
            }
            fNormals.Set(FromPtr<Real_v>(arrayx), FromPtr<Real_v>(arrayy),FromPtr<Real_v>(arrayz));
      */
      std::cout << "fNormals = " << fNormals << std::endl;
    }
  }

  void DotProduct(vecgeom::Vector3D<double> const &point, Real_v &result)
  {
    vecgeom::Vector3D<Real_v> vpoint(Real_v(point.x()), Real_v(point.y()), Real_v(point.z()));
    std::cout << "point = " << point << std::endl;
    std::cout << "vpoint = " << vpoint << std::endl;
    result = fNormals.Dot(vpoint);
  }
};

int main()
{
  using Real_v = vecgeom::VectorBackend::Real_v;
  TessellatedStruct<Real_v> tessels;
  for (size_t i = 0; i < kVecSize; ++i) {
    double phi   = i * vecgeom::kPi / 10.;
    double theta = phi;
    vecgeom::Vector3D<double> normal(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
    tessels.AddNormal(i, normal);
  }
  vecgeom::Vector3D<double> point(1., 2., 3.);
  Real_v dot;
  tessels.DotProduct(point, dot);
  std::cout << "point = " << point << "  dot = " << dot << std::endl;
  return 0;
}

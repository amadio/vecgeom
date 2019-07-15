#include "volumes/UnplacedBox.h"
#include "management/VolumeFactory.h"
#include "volumes/SpecializedBox.h"
#include "base/RNG.h"

#include "base/Utils3D.h"
#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECCORE_CUDA
SolidMesh *UnplacedBox::CreateMesh3D(Transformation3D const &trans, const size_t nFaces) const
{
  SolidMesh *sm = new SolidMesh();
  sm->ResetMesh(8, 6);
  Vector3D<Precision> const &box  = fBox.fDimensions;
  const Utils3D::Vec_t vertices[] = {{-box[0], -box[1], -box[2]}, {-box[0], box[1], -box[2]}, {box[0], box[1], -box[2]},
                                     {box[0], -box[1], -box[2]},  {-box[0], -box[1], box[2]}, {-box[0], box[1], box[2]},
                                     {box[0], box[1], box[2]},    {box[0], -box[1], box[2]}};
  sm->SetVertices(vertices, 8);
  sm->TransformVertices(trans);
  sm->InitConvexHexahedron();

  return sm;
}
#endif

VECCORE_ATT_HOST_DEVICE
void UnplacedBox::Print() const
{
  printf("UnplacedBox {%.2f, %.2f, %.2f}", x(), y(), z());
}

void UnplacedBox::Print(std::ostream &os) const
{
  os << "UnplacedBox {" << x() << ", " << y() << ", " << z() << "}";
}

Vector3D<Precision> UnplacedBox::SamplePointOnSurface() const
{
  Vector3D<Precision> p(dimensions());

  double S[3] = {p[1] * p[2], p[0] * p[2], p[0] * p[1]};

  double rand = (S[0] + S[1] + S[2]) * RNG::Instance().uniform(-1.0, 1.0);

  int axis = 0, direction = rand < 0.0 ? -1 : 1;

  rand = std::abs(rand);

  // rand is guaranteed (by the contract of RNG::Instance().uniform(-1.0, 1.0);)
  // to be less than one of the S[axis] since it starts
  // at a random number between 0 and (S[0] + S[1] + S[2])
  // and each iteration exits or substracts, respectively, S[0] and then S[1]
  // so on the 3rd iteration we have axis==2 and rand <= S[2]
  // Note that an automated tools (clang-tidy for example) will not be
  // able to detect this guarantee and complains about a possible out of
  // bound access (or garbage value).
  while (rand > S[axis])
    rand -= S[axis], axis++;

  p[0] = (axis == 0) ? direction * x() : p[0] * RNG::Instance().uniform(-1.0, 1.0);
  p[1] = (axis == 1) ? direction * y() : p[1] * RNG::Instance().uniform(-1.0, 1.0);
  p[2] = (axis == 2) ? direction * z() : p[2] * RNG::Instance().uniform(-1.0, 1.0);
  return p;
}

#ifndef VECCORE_CUDA
template <TranslationCode trans_code, RotationCode rot_code>
VPlacedVolume *UnplacedBox::Create(LogicalVolume const *const logical_volume,
                                   Transformation3D const *const transformation, VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedBox<trans_code, rot_code>(logical_volume, transformation);
    return placement;
  }
  return new SpecializedBox<trans_code, rot_code>(logical_volume, transformation);
}

VPlacedVolume *UnplacedBox::SpecializedVolume(LogicalVolume const *const volume,
                                              Transformation3D const *const transformation,
                                              const TranslationCode trans_code, const RotationCode rot_code,
                                              VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedBox>(volume, transformation, trans_code, rot_code, placement);
}
#else

template <TranslationCode trans_code, RotationCode rot_code>
VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedBox::Create(LogicalVolume const *const logical_volume,
                                   Transformation3D const *const transformation, const int id,
                                   VPlacedVolume *const placement)
{
  if (placement) {
    new (placement) SpecializedBox<trans_code, rot_code>(logical_volume, transformation, id);
    return placement;
  }
  return new SpecializedBox<trans_code, rot_code>(logical_volume, transformation, id);
}

VECCORE_ATT_DEVICE
VPlacedVolume *UnplacedBox::SpecializedVolume(LogicalVolume const *const volume,
                                              Transformation3D const *const transformation,
                                              const TranslationCode trans_code, const RotationCode rot_code,
                                              const int id, VPlacedVolume *const placement) const
{
  return VolumeFactory::CreateByTransformation<UnplacedBox>(volume, transformation, trans_code, rot_code, id,
                                                            placement);
}

#endif

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::VUnplacedVolume> UnplacedBox::CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const in_gpu_ptr) const
{
  return CopyToGpuImpl<UnplacedBox>(in_gpu_ptr, x(), y(), z());
}

DevicePtr<cuda::VUnplacedVolume> UnplacedBox::CopyToGpu() const
{
  return CopyToGpuImpl<UnplacedBox>();
}

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::UnplacedBox>::SizeOf();
template void DevicePtr<cuda::UnplacedBox>::Construct(const Precision x, const Precision y, const Precision z) const;

} // namespace cxx

#endif

} // namespace vecgeom

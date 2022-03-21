/// \file Transformation3D.cpp
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)
#include "VecGeom/base/Transformation3D.h"

#ifdef VECGEOM_CUDA_INTERFACE
#include "VecGeom/backend/cuda/Interface.h"
#endif
#include "VecGeom/base/SpecializedTransformation3D.h"

#ifdef VECGEOM_ROOT
#include "TGeoMatrix.h"
#endif

#include <sstream>
#include <stdio.h>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

const Transformation3D Transformation3D::kIdentity = Transformation3D();

// Transformation3D::Transformation3D(const Precision tx,
//                                   const Precision ty,
//                                   const Precision tz) :
//   fIdentity(false), fHasRotation(true), fHasTranslation(true)
//{
//  SetTranslation(tx, ty, tz);
//  SetRotation(1, 0, 0, 0, 1, 0, 0, 0, 1);
//  SetProperties();
//}

VECCORE_ATT_HOST_DEVICE
Transformation3D::Transformation3D(const Precision tx, const Precision ty, const Precision tz, const Precision phi,
                                   const Precision theta, const Precision psi)
    : fIdentity(false), fHasRotation(true), fHasTranslation(true)
{
  SetTranslation(tx, ty, tz);
  SetRotation(phi, theta, psi);
  SetProperties();
}

VECCORE_ATT_HOST_DEVICE
Transformation3D::Transformation3D(const Precision tx, const Precision ty, const Precision tz, const Precision r0,
                                   const Precision r1, const Precision r2, const Precision r3, const Precision r4,
                                   const Precision r5, const Precision r6, const Precision r7, const Precision r8)
    : fIdentity(false), fHasRotation(true), fHasTranslation(true)
{
  SetTranslation(tx, ty, tz);
  SetRotation(r0, r1, r2, r3, r4, r5, r6, r7, r8);
  SetProperties();
}

VECCORE_ATT_HOST_DEVICE
Transformation3D::Transformation3D(const Vector3D<Precision> &axis, bool inverse)
{
  SetTranslation(0, 0, 0);
  if (inverse)
    SetRotation(axis.Phi() * kRadToDeg - 90, -axis.Theta() * kRadToDeg, 0);
  else
    SetRotation(0, axis.Theta() * kRadToDeg, 90 - axis.Phi() * kRadToDeg);
  SetProperties();
}

void Transformation3D::Print(std::ostream &s) const
{
  s << "Transformation3D {{" << fTranslation[0] << "," << fTranslation[1] << "," << fTranslation[2] << "}";
  s << "{" << fRotation[0] << "," << fRotation[1] << "," << fRotation[2] << "," << fRotation[3] << "," << fRotation[4]
    << "," << fRotation[5] << "," << fRotation[6] << "," << fRotation[7] << "," << fRotation[8] << "}}\n";
}

VECCORE_ATT_HOST_DEVICE
void Transformation3D::Print() const
{
  printf("Transformation3D {{%.2f, %.2f, %.2f}, ", fTranslation[0], fTranslation[1], fTranslation[2]);
  printf("{%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f}}", fRotation[0], fRotation[1], fRotation[2],
         fRotation[3], fRotation[4], fRotation[5], fRotation[6], fRotation[7], fRotation[8]);
}

VECCORE_ATT_HOST_DEVICE
void Transformation3D::SetTranslation(const Precision tx, const Precision ty, const Precision tz)
{
  fTranslation[0] = tx;
  fTranslation[1] = ty;
  fTranslation[2] = tz;
}

VECCORE_ATT_HOST_DEVICE
void Transformation3D::SetTranslation(Vector3D<Precision> const &vec)
{
  SetTranslation(vec[0], vec[1], vec[2]);
}

VECCORE_ATT_HOST_DEVICE
void Transformation3D::SetProperties()
{
  fHasTranslation =
      (fabs(fTranslation[0]) > kTolerance || fabs(fTranslation[1]) > kTolerance || fabs(fTranslation[2]) > kTolerance)
          ? true
          : false;
  fHasRotation = (GenerateRotationCode() == rotation::kIdentity) ? false : true;
  fIdentity    = !fHasTranslation && !fHasRotation;
}

VECCORE_ATT_HOST_DEVICE
void Transformation3D::SetRotation(const Precision phi, const Precision theta, const Precision psi)
{

  const Precision sinphi = sin(kDegToRad * phi);
  const Precision cosphi = cos(kDegToRad * phi);
  const Precision sinthe = sin(kDegToRad * theta);
  const Precision costhe = cos(kDegToRad * theta);
  const Precision sinpsi = sin(kDegToRad * psi);
  const Precision cospsi = cos(kDegToRad * psi);

  fRotation[0] = cospsi * cosphi - costhe * sinphi * sinpsi;
  fRotation[1] = -sinpsi * cosphi - costhe * sinphi * cospsi;
  fRotation[2] = sinthe * sinphi;
  fRotation[3] = cospsi * sinphi + costhe * cosphi * sinpsi;
  fRotation[4] = -sinpsi * sinphi + costhe * cosphi * cospsi;
  fRotation[5] = -sinthe * cosphi;
  fRotation[6] = sinpsi * sinthe;
  fRotation[7] = cospsi * sinthe;
  fRotation[8] = costhe;
}

VECCORE_ATT_HOST_DEVICE
void Transformation3D::SetRotation(Vector3D<Precision> const &vec)
{
  SetRotation(vec[0], vec[1], vec[2]);
}

VECCORE_ATT_HOST_DEVICE
void Transformation3D::SetRotation(const Precision rot0, const Precision rot1, const Precision rot2,
                                   const Precision rot3, const Precision rot4, const Precision rot5,
                                   const Precision rot6, const Precision rot7, const Precision rot8)
{

  fRotation[0] = rot0;
  fRotation[1] = rot1;
  fRotation[2] = rot2;
  fRotation[3] = rot3;
  fRotation[4] = rot4;
  fRotation[5] = rot5;
  fRotation[6] = rot6;
  fRotation[7] = rot7;
  fRotation[8] = rot8;
}

VECCORE_ATT_HOST_DEVICE
RotationCode Transformation3D::GenerateRotationCode() const
{
  int code = 0;
  for (int i = 0; i < 9; ++i) {
    // Assign each bit
    code |= (1 << i) * (fabs(fRotation[i]) > kTolerance);
  }
  if (code == rotation::kDiagonal && (fRotation[0] == 1. && fRotation[4] == 1. && fRotation[8] == 1.)) {
    code = rotation::kIdentity;
  }
  return code;
}

/**
 * Very simple translation code. Kept as an integer in case other cases are to
 * be implemented in the future.
 * /return The transformation's translation code, which is 0 for transformations
 *         without translation and 1 otherwise.
 */
VECCORE_ATT_HOST_DEVICE
TranslationCode Transformation3D::GenerateTranslationCode() const
{
  return (fHasTranslation) ? translation::kGeneric : translation::kIdentity;
}

#ifdef VECGEOM_ROOT
// function to convert this transformation to a TGeo transformation
// mainly used for the benchmark comparisons with ROOT
TGeoMatrix *Transformation3D::ConvertToTGeoMatrix() const
{
  double rotd[9];
  if (fHasRotation) {
    for (auto i = 0; i < 9; ++i)
      rotd[i] = Rotation()[i];
  }

  if (fIdentity) {
    return new TGeoIdentity();
  }
  if (fHasTranslation && !fHasRotation) {
    return new TGeoTranslation(fTranslation[0], fTranslation[1], fTranslation[2]);
  }
  if (fHasRotation && !fHasTranslation) {
    TGeoRotation *tmp = new TGeoRotation();
    tmp->SetMatrix(rotd);
    return tmp;
  }
  if (fHasTranslation && fHasRotation) {
    TGeoRotation *tmp = new TGeoRotation();
    tmp->SetMatrix(rotd);
    return new TGeoCombiTrans(fTranslation[0], fTranslation[1], fTranslation[2], tmp);
  }
  return 0;
}
#endif

std::ostream &operator<<(std::ostream &os, Transformation3D const &transformation)
{
  os << "Transformation {" << transformation.Translation() << ", "
     << "(" << transformation.Rotation(0) << ", " << transformation.Rotation(1) << ", " << transformation.Rotation(2)
     << ", " << transformation.Rotation(3) << ", " << transformation.Rotation(4) << ", " << transformation.Rotation(5)
     << ", " << transformation.Rotation(6) << ", " << transformation.Rotation(7) << ", " << transformation.Rotation(8)
     << ")}"
     << "; identity(" << transformation.IsIdentity() << "); rotation(" << transformation.HasRotation() << ")";
  return os;
}

#ifdef VECGEOM_CUDA_INTERFACE

DevicePtr<cuda::Transformation3D> Transformation3D::CopyToGpu(DevicePtr<cuda::Transformation3D> const gpu_ptr) const
{

  gpu_ptr.Construct(fTranslation[0], fTranslation[1], fTranslation[2], fRotation[0], fRotation[1], fRotation[2],
                    fRotation[3], fRotation[4], fRotation[5], fRotation[6], fRotation[7], fRotation[8]);
  CudaAssertError();
  return gpu_ptr;
}

DevicePtr<cuda::Transformation3D> Transformation3D::CopyToGpu() const
{

  DevicePtr<cuda::Transformation3D> gpu_ptr;
  gpu_ptr.Allocate();
  return this->CopyToGpu(gpu_ptr);
}

/**
 * Copy a large number of transformation instances to the GPU.
 * \param trafos Host instances to copy.
 * \param gpu_ptrs Device pointers to indicate where the transformations should be placed.
 * The device memory must have been allocated before copying.
 */
void Transformation3D::CopyManyToGpu(const std::vector<Transformation3D const *>& trafos,
                                     const std::vector<DevicePtr<cuda::Transformation3D>>& gpu_ptrs)
{
  assert(trafos.size() == gpu_ptrs.size());

  // Memory for constructor data
  // Store it as
  // tx0, tx1, tx2, ...,
  // ty0, ty1, ty2, ...,
  // ...
  // rot0_0, rot0_1, rot0_2, ...,
  // ...
  std::vector<Precision> trafoData(12 * trafos.size());

  std::size_t trafoCounter = 0;
  for (Transformation3D const * trafo : trafos) {
    for (unsigned int i = 0; i < 3; ++i) trafoData[trafoCounter +  i    * trafos.size()] = trafo->Translation(i);
    for (unsigned int i = 0; i < 9; ++i) trafoData[trafoCounter + (i+3) * trafos.size()] = trafo->Rotation(i);
    ++trafoCounter;
  }

  ConstructManyOnGpu<cuda::Transformation3D>(trafos.size(), gpu_ptrs.data(),
      trafoData.data(),                     trafoData.data() +  1 * trafos.size(), trafoData.data() +  2 * trafos.size(), // translations
      trafoData.data() + 3 * trafos.size(), trafoData.data() +  4 * trafos.size(), trafoData.data() +  5 * trafos.size(), // rotations
      trafoData.data() + 6 * trafos.size(), trafoData.data() +  7 * trafos.size(), trafoData.data() +  8 * trafos.size(),
      trafoData.data() + 9 * trafos.size(), trafoData.data() + 10 * trafos.size(), trafoData.data() + 11 * trafos.size()
  );
}

#endif // VECGEOM_CUDA_INTERFACE

} // namespace VECGEOM_IMPL_NAMESPACE

#ifdef VECCORE_CUDA

namespace cxx {

template size_t DevicePtr<cuda::Transformation3D>::SizeOf();
template void DevicePtr<cuda::Transformation3D>::Construct(const Precision tx, const Precision ty, const Precision tz,
                                                           const Precision r0, const Precision r1, const Precision r2,
                                                           const Precision r3, const Precision r4, const Precision r5,
                                                           const Precision r6, const Precision r7,
                                                           const Precision r8) const;
template void ConstructManyOnGpu<Transformation3D>(std::size_t, DevicePtr<cuda::Transformation3D> const *,
                                                   Precision const * tx, Precision const * ty, Precision const * tz,
                                                   Precision const * r0, Precision const * r1, Precision const * r2,
                                                   Precision const * r3, Precision const * r4, Precision const * r5,
                                                   Precision const * r6, Precision const * r7, Precision const * r8);

} // namespace cxx

#endif // VECCORE_CUDA

} // namespace vecgeom

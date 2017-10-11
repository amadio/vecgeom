/// @file UnplacedExtruded.h
/// @author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_VOLUMES_UNPLACEDEXTRUDED_H_
#define VECGEOM_VOLUMES_UNPLACEDEXTRUDED_H_

#include "UnplacedTessellated.h"
#include "volumes/PlanarPolygon.h"

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(class UnplacedExtruded;);
VECGEOM_DEVICE_DECLARE_CONV(class, UnplacedExtruded);

inline namespace VECGEOM_IMPL_NAMESPACE {

struct XtruVertex2 {
  double x;
  double y;
};

struct XtruSection {
  Vector3D<double> fOrigin; // Origin of the section
  double fScale;
};

class UnplacedExtruded : public UnplacedTessellated {

template <typename U>
using vector_t = vecgeom::Vector<U>;

private:
  vector_t<XtruVertex2> fVertices;      ///< Polygone vertices
  vector_t<XtruSection> fSections;      ///< Vector of sections
  PlanarPolygon *fPolygon = nullptr;            ///< Planar polygon

public:
  /** @brief Constructor providing polygone vertices and sections */
  VECCORE_ATT_HOST_DEVICE
  UnplacedExtruded(int nvertices, XtruVertex2 const *vertices,
                   int nsections, XtruSection const *sections);

  /** @brief Check if point i is inside triangle (i1, i2, i3) defined clockwise. */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsPointInside(size_t i, size_t i1, size_t i2, size_t i3) {
    if (!IsConvexSide(i1, i2, i) || !IsConvexSide(i2, i3, i) || !IsConvexSide(i3, i1, i))
      return false;
    return true;
  } 

  /** @brief Check if the polygone segments (i0, i1) and (i1, i2) make a convex side */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  bool IsConvexSide(size_t i0, size_t i1, size_t i2) {
    const double *x = fPolygon->GetVertices().x(); 
    const double *y = fPolygon->GetVertices().y();
    double cross = (x[i1] - x[i0]) * (y[i2] - y[i1]) - (x[i2] - x[i1]) * (y[i1] - y[i0]);
    return cross < 0.;
  } 

  /** @brief Returns the coordinates for a given vertex index at a given section */
  VECCORE_ATT_HOST_DEVICE
  VECGEOM_FORCE_INLINE
  Vector3D<double> VertexToSection(size_t ivert, size_t isect) {
    const double *x = fPolygon->GetVertices().x(); 
    const double *y = fPolygon->GetVertices().y();
    Vector3D<double> vert(fSections[isect].fOrigin[0] + fSections[isect].fScale * x[ivert],
                     fSections[isect].fOrigin[1] + fSections[isect].fScale * y[ivert],
                     fSections[isect].fOrigin[2]);
    return vert;
  }

  VECCORE_ATT_HOST_DEVICE
  virtual void Print() const final;

  virtual int memory_size() const final { return sizeof(*this); }
  std::string GetEntityType() const { return "Extruded"; }

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECCORE_ATT_DEVICE
  static VPlacedVolume *Create(LogicalVolume const *const logical_volume, Transformation3D const *const transformation,
#ifdef VECCORE_CUDA
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);
#ifdef VECGEOM_CUDA_INTERFACE
  virtual size_t DeviceSizeOf() const { return DevicePtr<cuda::UnplacedExtruded>::SizeOf(); }
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu() const;
  virtual DevicePtr<cuda::VUnplacedVolume> CopyToGpu(DevicePtr<cuda::VUnplacedVolume> const gpu_ptr) const;
#endif

#if defined(VECGEOM_USOLIDS)
  std::ostream &StreamInfo(std::ostream &os) const;
#endif

private:
  virtual void Print(std::ostream &os) const final;

  VECCORE_ATT_DEVICE
  virtual VPlacedVolume *SpecializedVolume(LogicalVolume const *const volume,
                                           Transformation3D const *const transformation,
                                           const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECCORE_CUDA
                                           const int id,
#endif
                                           VPlacedVolume *const placement = NULL) const final;

};
}
} // end global namespace

#endif // VECGEOM_VOLUMES_UNPLACEDEXTRUDED_H_

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

class UnplacedExtruded : public UnplacedExtruded {
public:
  struct Vertex2 {
    double x;
    double y;
  };
  
  struct Section {
    Vector3D<double> fOrigin; // Origin of the section
    double fScale;
  };
    
private:
  bool fConvex = false;        ///< Convexity of the polygone
  Vector_t<Vertex2> fVertices; ///< Polygone vertices
  Vector_t<Section> fSections; ///< Vector of sections
  PlanarPolygon *fPolygon = nullptr;     ///< Planar polygon
  
public:
  VECCORE_ATT_HOST_DEVICE
  UnplacedExtruded(int nvertices, Vertex2 const *vertices, int nsections, Section const *sections);

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

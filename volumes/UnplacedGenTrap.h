#ifndef VECGEOM_VOLUMES_UNPLACEDGENTRAP_H_
#define VECGEOM_VOLUMES_UNPLACEDGENTRAP_H_

#include "base/Global.h"
#include "base/AlignedBase.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/SecondOrderSurfaceShell.h"
#include "volumes/UnplacedBox.h" // for bounding box

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class UnplacedGenTrap; )
VECGEOM_DEVICE_DECLARE_CONV( UnplacedGenTrap )

inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * A generic trap:
 * see TGeoArb8 or UGenericTrap
 */
class UnplacedGenTrap : public VUnplacedVolume, public AlignedBase {

public:
  typedef Vector3D<Precision> VertexType;
  // the bounding box: half lengths and origin
  UnplacedBox fBoundingBox;
   Vector3D<Precision> fBoundingBoxOrig;

  // the eight points that define the Arb8
  // actually we will neglect the z coordinates of those
  Vector3D<Precision> fVertices[8];

  // we also store this in SOA form
  Precision fVerticesX[8];
  Precision fVerticesY[8];

  // the half-height of the GenTrap
  Precision fDz;
  Precision fInverseDz;
  Precision fHalfInverseDz;
  bool fIstwisted;


  // we store the connecting vectors in SOA Form
  // these vectors are used to calculate the polygon at a certain z-height
  // less elegant than UVector2 but enables vectorization
  // TODO: make this elegant
  // moreover: they can be precomputed !!
  // Compute intersection between Z plane containing point and the shape
  //
  Precision fConnectingComponentsX[4];
  Precision fConnectingComponentsY[4];

  Precision fDeltaX[8]; //int  j = (i + 1) % 4;
  Precision fDeltaY[8];

  // to be done
  SecondOrderSurfaceShell<4> fSurfaceShell;

public:

  VECGEOM_CUDA_HEADER_BOTH
  // constructor
  UnplacedGenTrap(Vector3D<Precision> vertices[],
                  Precision halfzheight) :
                  fBoundingBox(Vector3D<Precision>(0.,0.,0.)),
                  fBoundingBoxOrig(0.,0.,0.),
                  fVertices(), fVerticesX(),  fVerticesY(),
                  fDz(halfzheight), fInverseDz(1./halfzheight), fHalfInverseDz(0.5/halfzheight),
                  fIstwisted(true), 
                  fConnectingComponentsX(), fConnectingComponentsY(),
                  fDeltaX(), fDeltaY(),
                  fSurfaceShell(vertices, halfzheight)
  {
      for (int i=0;i<8;++i)
        fVertices[i]=vertices[i];

      // initialize the connecting components
      for (int i=0;i<4;++i){
        fConnectingComponentsX[i]=(fVertices[i]-fVertices[i+4]).x();
        fConnectingComponentsY[i]=(fVertices[i]-fVertices[i+4]).y();
        fVerticesX[i]=fVertices[i].x();
        fVerticesX[i+4]=fVertices[i+4].x();
        fVerticesY[i]=fVertices[i].y();
        fVerticesY[i+4]=fVertices[i+4].y();
      }
      for (int i=0;i<4;++i){
        int j = (i + 1) % 4;
        fDeltaX[i] = fVerticesX[j]-fVerticesX[i];
        fDeltaX[i+4] = fVerticesX[j+4]-fVerticesX[i+4];
        fDeltaY[i] = fVerticesY[j]-fVerticesY[i];
        fDeltaY[i+4] = fVerticesY[j+4]-fVerticesY[i+4];
      }

    ComputeBoundingBox();

  }

  virtual ~UnplacedGenTrap() {}

  VECGEOM_CUDA_HEADER_BOTH
  SecondOrderSurfaceShell<4> const& GetShell() const
  {
      return (fSurfaceShell);
  }

  VECGEOM_CUDA_HEADER_BOTH
  Precision GetDZ() const { return (fDz); }

  VECGEOM_CUDA_HEADER_BOTH
  VertexType const & GetVertex(int i) const
  {
    //  assert(i<8);
      return fVertices[i];
  }

  // computes if this gentrap is twisted
  // should be a private method?
  bool ComputeIsTwisted();

  // computes and sets the bounding box member of this class
  void ComputeBoundingBox();

  virtual int memory_size() const { return sizeof(*this); }

  VECGEOM_CUDA_HEADER_BOTH
  virtual void Print() const;

  template <TranslationCode transCodeT, RotationCode rotCodeT>
  VECGEOM_CUDA_HEADER_DEVICE
  static VPlacedVolume* Create(LogicalVolume const *const logical_volume,
                               Transformation3D const *const transformation,
#ifdef VECGEOM_NVCC
                               const int id,
#endif
                               VPlacedVolume *const placement = NULL);

#ifdef VECGEOM_CUDA_INTERFACE
  virtual VUnplacedVolume* CopyToGpu() const;
  virtual VUnplacedVolume* CopyToGpu(VUnplacedVolume *const gpu_ptr) const;
#endif

  Precision Capacity() { return volume(); }

  VECGEOM_INLINE
  Precision volume() const {
    int i,j;
    Precision capacity = 0;
    for (i=0; i<4; i++) {
      j = (i+1)%4;
      capacity += 0.25*fDz*((fVerticesX[i]+fVerticesX[i+4])*(fVerticesY[j]+fVerticesY[j+4]) -
                            (fVerticesX[j]+fVerticesX[j+4])*(fVerticesY[i]+fVerticesY[i+4]) +
                    (1./3)*((fVerticesX[i+4]-fVerticesX[i])*(fVerticesY[j+4]-fVerticesY[j]) -
                            (fVerticesX[j]-fVerticesX[j+4])*(fVerticesY[i]-fVerticesY[i+4])));
     }
     return Abs(capacity);
  }

  VECGEOM_INLINE
  Precision SurfaceArea() const {
    return 0.;
  }

  void Extent( Vector3D<Precision> &, Vector3D<Precision> &) const;

  Vector3D<Precision> GetPointOnSurface() const { return Vector3D<Precision>(); }

  virtual std::string GetEntityType() const { return "GenTrap";}

  virtual void Print(std::ostream &os) const;

  VECGEOM_CUDA_HEADER_DEVICE
  virtual VPlacedVolume* SpecializedVolume(
      LogicalVolume const *const volume,
      Transformation3D const *const transformation,
      const TranslationCode trans_code, const RotationCode rot_code,
#ifdef VECGEOM_NVCC
      const int id,
#endif
      VPlacedVolume *const placement = NULL) const;

}; // end of class declaration

} } // End global namespace

#endif

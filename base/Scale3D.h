/// \file Scale3D.h
/// \author Mihaela Gheata (mihaela.gheata@cern.ch)

#ifndef VECGEOM_BASE_SCALE3D_H_
#define VECGEOM_BASE_SCALE3D_H_

#include "base/Global.h"

#include "base/Vector3D.h"
#include "backend/Backend.h"

#include "backend/Backend.h"
#ifdef VECGEOM_CUDA_INTERFACE
#include "backend/cuda/Interface.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE( class Scale3D; )

inline namespace VECGEOM_IMPL_NAMESPACE {

#ifndef VECGEOM_NVCC
   } namespace cuda { class Scale3D; }
   inline namespace VECGEOM_IMPL_NAMESPACE {
   //class vecgeom::cuda::Scale3D;
#endif

class Scale3D {

private:
  Vector3D<Precision> fScale;        /// scale transformation 
  Vector3D<Precision> fInvScale;     /// inverse scale (avoid divisions)
  Precision           fSclLocal;     /// factor to apply to safety to convert to local frame
  Precision           fSclMaster;    /// factor to apply to safety to convert to master frame

public:

  /**
   * Default constructor
   */
  VECGEOM_CUDA_HEADER_BOTH
  Scale3D() : fScale(1.,1.,1.), fInvScale(1.,1.,1.), fSclLocal(1.), fSclMaster(1.) { }
    
  /**
   * Constructor with scale parameters on each axis
   * @param sx Scale value on x
   * @param sy Scale value on y
   * @param sz Scale value on z
   */
  VECGEOM_CUDA_HEADER_BOTH
  Scale3D(Precision sx, Precision sy, Precision sz);

  /**
   * Constructor with scale parameters in a Vector3D
   * @param scale Scale as Vector3D
   */
  VECGEOM_CUDA_HEADER_BOTH
  Scale3D(Vector3D<Precision> const &scale);
    
  /**
   * Copy constructor.
   */
  VECGEOM_CUDA_HEADER_BOTH
  Scale3D(Scale3D const &other);
  
  /**
   * Assignment operator
   */
  VECGEOM_CUDA_HEADER_BOTH
  Scale3D& operator=(Scale3D const &other);
          
  /**
   * Update the backed-up inverse scale and special conversion factors based
   * on the values of the scale. Needed whenever the scale has changed value.
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Update();

  /**
   * Get reference to the scale vector.
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  const Vector3D<Precision> &Scale() const { return fScale; }

  /**
   * Get reference to the inverse scale vector.
   */  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  const Vector3D<Precision> &InvScale() const { return fInvScale; }
    
  /**
   * Set scale based on vector.
   */  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void SetScale(Vector3D<Precision> const &scale);

  /**
   * Set scale based on values.
   */  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void SetScale(Precision sx, Precision sy, Precision sz);

  /**
   * Transform point from master to local frame
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void Transform(Vector3D<Precision> const &master,
                                Vector3D<Precision> &local) const {
    local.Set(master[0]*fInvScale[0], master[1]*fInvScale[1], master[2]*fInvScale[2]);      
  }
  
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Precision> Transform(Vector3D<Precision> const &master) const {
    Vector3D<Precision> local(master[0]*fInvScale[0], master[1]*fInvScale[1], master[2]*fInvScale[2]);
    return local;
  }     
    
  /**
   * Transform point from local to master frame
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  void InverseTransform(Vector3D<Precision> const &local, 
                                Vector3D<Precision> & master) const {
    master.Set(local[0]*fScale[0], local[1]*fScale[1], local[2]*fScale[2]);
  }


  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Vector3D<Precision> InverseTransform(Vector3D<Precision> const &local) const {
    Vector3D<Precision> master(local[0]*fScale[0], local[1]*fScale[1], local[2]*fScale[2]);
    return master;
  }

  /**
   * Transform distance along given direction from master to local frame
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision TransformDistance(Precision dist,
                                Vector3D<Precision> &dir) const;

  /**
   * Transform safe distance from master to local frame (conservative)
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision TransformSafety(Precision safety) const {
    return (safety * fSclLocal);  
  }
    
  /**
   * Transform distance along given direction from local to master frame
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision InverseTransformDistance(Precision dist,
                                Vector3D<Precision> &dir) const;

  /**
   * Transform safe distance from local to master frame (conservative)
   */
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  Precision InverseTransformSafety(Precision safety) const {
    return (safety * fSclMaster);  
  }

public:

  static const Scale3D kIdentity;

}; // End class Scale3D

std::ostream& operator<<(std::ostream& os, Scale3D const &scale);

} } // End global namespace

#endif // VECGEOM_BASE_SCALE3D_H_

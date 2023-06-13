//!    \file ReflFactory.h
//!    \brief Factory dealing with reflected volumes, heavily inspired by G4ReflectionFactory
//!
//!    \authors Author:  Andrei Gheata <andrei.gheata@cern.ch>
//!

#ifndef VECGEOM_MANAGEMENT_REFLFACTORY_H_
#define VECGEOM_MANAGEMENT_REFLFACTORY_H_

#include <string>
#include <map>
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"

namespace vecgeom {
inline namespace cxx {

// Class providing functions for volumes placements with a general
// transfomation that may contain reflection.
// Reflection is then applied to a solid: a new UnplacedScaledShape
// instance is created and is placed with a transformation containing
// pure rotation and translation only.
// The pair of constituent and reflected logical volumes is
// considered as a generalized logical volume that is addressed
// by user specifying the constituent logical volume.
//
// Decomposition of a general transformation that can include reflection
// in a "reflection-free" transformation:
//
// x(inM') = TG*x(inM)         TG - general transformation
//         = T*(R*x(inM))      T  - "reflection-free" transformation
//         = T* x(inReflM)
//
// Daughters transformation:
// When a volume V containing daughter D with transformation TD
// is placed in mother M with a general tranformation TGV,
// the TGV is decomposed. New reflected volume ReflV containing
// a new daughter ReflD with reflected transformation ReflTD is created:
//
// x(inV) = TD * x(inD);
// x(inM) = TGV * x(inV)
//        = TV * R * x(inV)
//        = TV * R * TD * x(inD)
//        = TV * R*TD*R-1 * R*x(inD)
//        = TV * ReflTD * x(inReflD)

class ReflFactory {

  using Vector3                   = vecgeom::Vector3D<vecgeom::Precision>;
  using Transformation3D          = vecgeom::Transformation3D;
  using LogicalVolume             = vecgeom::LogicalVolume;
  using VPlacedVolume             = vecgeom::VPlacedVolume;
  using ReflectedVolumesMap       = std::map<LogicalVolume *, LogicalVolume *>;
  using LogicalVolumesMapIterator = ReflectedVolumesMap::const_iterator;

public:
  ~ReflFactory() { Clean(); }

  static ReflFactory &Instance()
  {
    static ReflFactory instance;
    return instance;
  }

  // In case a reflection is detected, this creates new reflected solid and
  // logical volume (or retrieves them from a map if the reflected
  // objects were already created), transforms the daughters (if present)
  // and place it in the given mother.
  // The result is a pair of physical volumes;
  // the second physical volume is a placement in a reflected mother
  // or 0 if mother LV was not reflected.
  bool Place(Transformation3D const &transform3D, Vector3 const &scale, std::string const &name,
             vecgeom::LogicalVolume *LV, vecgeom::LogicalVolume *motherLV, int copyNo);

  void SetVerboseLevel(int verboseLevel) { fVerboseLevel = verboseLevel; }
  int GetVerboseLevel() const { return fVerboseLevel; }

  // Returns the consituent volume of the given reflected volume,
  // nullptr if the given reflected volume was not found.
  LogicalVolume *GetConstituentLV(LogicalVolume *reflLV) const;

  // Returns the reflected volume of the given consituent volume,
  // nullptr if the given volume was not reflected.
  LogicalVolume *GetReflectedLV(LogicalVolume *lv) const;

  // Returns true if the given volume has been already reflected
  // (is in the map of constituent volumes).
  bool IsConstituent(LogicalVolume const *lv) const;

  // Returns true if the given volume is a reflected volume
  // (is in the map reflected  volumes).
  bool IsReflected(LogicalVolume const *lv) const;

  // Returns a handle to the internal map of volumes which have
  // been reflected, after that placement or replication is performed.
  ReflectedVolumesMap const &GetReflectedVolumesMap() const { return fReflectedLVMap; }

  // Clear maps of constituent and reflected volumes.
  // To be used exclusively when volumes are removed from the stores.
  void Clean();

  // Disabled copy constructor and assignment operator.
  ReflFactory(const ReflFactory &)            = delete;
  ReflFactory &operator=(const ReflFactory &) = delete;

protected:
  ReflFactory();
  // Protected singleton constructor.

private:
  // Returns scale * pureTransform3D * scale.Inverse()
  Transformation3D ConvertScaledToPureTransformation(Transformation3D const &pureTransform3D, Vector3 const &scale);
  // Gets/creates the reflected solid and logical volume
  // and copies + transforms LV daughters.
  LogicalVolume *ReflectLV(LogicalVolume *LV, Vector3 const &scale);

  // Creates the reflected solid and logical volume
  // and add the logical volumes pair in the maps.
  LogicalVolume *CreateReflectedLV(LogicalVolume *LV, Vector3 const &scale);

  // Reflects daughters recursively.
  void ReflectDaughters(LogicalVolume *LV, LogicalVolume *refLV, Vector3 const &scale);

  // Copies and transforms daughter of PVPlacement type of
  // a constituent volume into a reflected volume.
  void ReflectPlacedVolume(VPlacedVolume const *PV, LogicalVolume *refLV, Vector3 const &scale);

  // Returns true if the scale is negative, false otherwise.
  bool IsReflection(Vector3 const &scale) const { return scale[0] * scale[1] * scale[2] < 0; }

private:
  int fVerboseLevel = 0;
  std::string fNameExtension;
  ReflectedVolumesMap fConstituentLVMap;
  ReflectedVolumesMap fReflectedLVMap;
};

} // namespace cxx
} // namespace vecgeom

#endif

//
// ********************************************************************
// * This Software is part of the AIDA Unified Solids Library package *
// * See: https://aidasoft.web.cern.ch/USolids                        *
// ********************************************************************
//
// $Id:$
//
// --------------------------------------------------------------------
//
// UTessellatedSolid
//
// Class description:
//
// UTessellatedSolid is a special Geant4 solid defined by a number of
// facets (UVFacet). It is important that the supplied facets shall form a
// fully enclose space which is the solid.
// Only two types of facet can be used for the construction of
// a UTessellatedSolid, i.e. the UTriangularFacet and UQuadrangularFacet.
//
// How to contruct a UTessellatedSolid:
//
// First declare a tessellated solid:
//
//      UTessellatedSolid* solidTarget = new UTessellatedSolid("Solid_name");
//
// Define the facets which form the solid
//
//      double targetSiz = 10*cm ;
//      UTriangularFacet *facet1 = new
//      UTriangularFacet (UVector3(-targetSize,-targetSize,        0.0),
//                         UVector3(+targetSize,-targetSize,        0.0),
//                         UVector3(        0.0,        0.0,+targetSize),
//                         ABSOLUTE);
//      UTriangularFacet *facet2 = new
//      UTriangularFacet (UVector3(+targetSize,-targetSize,        0.0),
//                         UVector3(+targetSize,+targetSize,        0.0),
//                         UVector3(        0.0,        0.0,+targetSize),
//                         ABSOLUTE);
//      UTriangularFacet *facet3 = new
//      UTriangularFacet (UVector3(+targetSize,+targetSize,        0.0),
//                         UVector3(-targetSize,+targetSize,        0.0),
//                         UVector3(        0.0,        0.0,+targetSize),
//                         ABSOLUTE);
//      UTriangularFacet *facet4 = new
//      UTriangularFacet (UVector3(-targetSize,+targetSize,        0.0),
//                         UVector3(-targetSize,-targetSize,        0.0),
//                         UVector3(        0.0,        0.0,+targetSize),
//                         ABSOLUTE);
//      UQuadrangularFacet *facet5 = new
//      UQuadrangularFacet (UVector3(-targetSize,-targetSize,      0.0),
//                           UVector3(-targetSize,+targetSize,      0.0),
//                           UVector3(+targetSize,+targetSize,      0.0),
//                           UVector3(+targetSize,-targetSize,      0.0),
//                           ABSOLUTE);
//
// Then add the facets to the solid:
//
//      solidTarget->AddFacet((UVFacet*) facet1);
//      solidTarget->AddFacet((UVFacet*) facet2);
//      solidTarget->AddFacet((UVFacet*) facet3);
//      solidTarget->AddFacet((UVFacet*) facet4);
//      solidTarget->AddFacet((UVFacet*) facet5);
//
// Finally declare the solid is complete:
//
//      solidTarget->SetSolidClosed(true);
//
// 11.07.12 Marek Gayer
//          Created from original implementation in Geant4
// --------------------------------------------------------------------

#ifndef UTessellatedSolid_hh
#define UTessellatedSolid_hh

#include "VUSolid.hh"
#include "UTriangularFacet.hh"
#include "UVoxelizer.hh"

#ifdef VECGEOM_REPLACE_USOLIDS
//============== here for VecGeom-based implementation

#include "volumes/LogicalVolume.h"
#include "base/Transformation3D.h"
#include "volumes/UnplacedTessellated.h"

#include "volumes/USolidsAdapter.h"

class UTessellatedSolid : public vecgeom::USolidsAdapter<vecgeom::UnplacedTessellated> {

  // just forwards UTessellatedSolid to vecgeom tessellated solid
  using Shape_t = vecgeom::UnplacedTessellated;
  using Base_t  = vecgeom::USolidsAdapter<vecgeom::UnplacedTessellated>;

  // inherit all constructors
  using Base_t::Base_t;

public:
  VECCORE_ATT_HOST_DEVICE
  UTessellatedSolid() : Base_t("") {}

  UTessellatedSolid(const std::string &pName) : Base_t(pName.c_str()) {}

  bool AddFacet(VUFacet *aFacet)
  {
    if (aFacet->GetNumberOfVertices() == 3)
      return AddTriangularFacet(aFacet->GetVertex(0), aFacet->GetVertex(1), aFacet->GetVertex(2), true);
    else
      return AddQuadrilateralFacet(aFacet->GetVertex(0), aFacet->GetVertex(1), aFacet->GetVertex(2),
                                   aFacet->GetVertex(3), true);
  }

  inline VUFacet *GetFacet(int i) const
  {
    vecgeom::TriangleFacet<double> *facet = Base_t::GetFacet(i);
    VUFacet *ufacet =
        new UTriangularFacet(UVector3(facet->fVertices[0].x(), facet->fVertices[0].y(), facet->fVertices[0].z()),
                             UVector3(facet->fVertices[1].x(), facet->fVertices[1].y(), facet->fVertices[1].z()),
                             UVector3(facet->fVertices[2].x(), facet->fVertices[2].y(), facet->fVertices[2].z()),
                             UFacetVertexType::UABSOLUTE);
    return ufacet;
  }

  int GetNumberOfFacets() const { return GetNFacets(); }

  UGeometryType GetEntityType() const override { return "UTessellatedSolid"; }

  void SetSolidClosed(const bool t)
  {
    if (t) Close();
  }
  bool GetSolidClosed() const { return IsClosed(); }

  inline virtual void GetParametersList(int /*aNumber*/, double * /*aArray*/) const override {}
  inline virtual void ComputeBBox(UBBox * /*aBox*/, bool /*aStore = false*/) override {}
  std::ostream &StreamInfo(std::ostream &os) const override
  {
    int oldprc = os.precision(16);
    os << "-----------------------------------------------------------\n"
       << "     *** Dump for solid - " << GetEntityType() << " ***\n"
       << "     ===================================================\n"
       << " Solid type: VecGeom tessellated solid\n"
       << " Parameters: \n";
    os.precision(oldprc);
    return os;
  }

  inline void SetMaxVoxels(int) {}

  /* This function from the UTessellatedSolid interface exposes internals not
     available in vecgeom tessellated solid and will crash if called */
  inline UVoxelizer &GetVoxels() { return *(UVoxelizer *)(nullptr); }

  virtual VUSolid *Clone() const override
  {
    UTessellatedSolid *clone = new UTessellatedSolid(GetName().c_str());
    // ... copy/construct data
    return clone;
  }

  double GetMinXExtent() const
  {
    vecgeom::Vector3D<double> aMin, aMax;
    Extent(aMin, aMax);
    return aMin.x();
  }
  double GetMaxXExtent() const
  {
    vecgeom::Vector3D<double> aMin, aMax;
    Extent(aMin, aMax);
    return aMax.x();
  }
  double GetMinYExtent() const
  {
    vecgeom::Vector3D<double> aMin, aMax;
    Extent(aMin, aMax);
    return aMin.y();
  }
  double GetMaxYExtent() const
  {
    vecgeom::Vector3D<double> aMin, aMax;
    Extent(aMin, aMax);
    return aMax.y();
  }
  double GetMinZExtent() const
  {
    vecgeom::Vector3D<double> aMin, aMax;
    Extent(aMin, aMax);
    return aMin.z();
  }
  double GetMaxZExtent() const
  {
    vecgeom::Vector3D<double> aMin, aMax;
    Extent(aMin, aMax);
    return aMax.z();
  }

  int AllocatedMemoryWithoutVoxels() { return 0; }
  int AllocatedMemory() { return 0; }
  void DisplayAllocatedMemory() {}
};
//============== end of VecGeom-based implementation

#else

//============== here for USolids-based implementation

#include <iostream>
#include <vector>
#include <set>
#include <map>

struct UVertexInfo {
  int id;
  double mag2;
};

class UVertexComparator {
public:
  bool operator()(const UVertexInfo &l, const UVertexInfo &r) const
  {
    return l.mag2 == r.mag2 ? l.id < r.id : l.mag2 < r.mag2;
  }
};

class UTessellatedSolid : public VUSolid {
public:
  UTessellatedSolid();
  virtual ~UTessellatedSolid();

  UTessellatedSolid(const std::string &name);

  UTessellatedSolid(__void__ &);
  // Fake default constructor for usage restricted to direct object
  // persistency for clients requiring preallocation of memory for
  // persistifiable objects.

  UTessellatedSolid(const UTessellatedSolid &s);
  UTessellatedSolid &operator=(const UTessellatedSolid &s);
  UTessellatedSolid &operator+=(const UTessellatedSolid &right);

  bool AddFacet(VUFacet *aFacet);
  inline VUFacet *GetFacet(int i) const { return fFacets[i]; }
  int GetNumberOfFacets() const;

  virtual double GetSurfaceArea();

  virtual VUSolid::EnumInside Inside(const UVector3 &p) const;

  virtual bool Normal(const UVector3 &p, UVector3 &aNormal) const;

  virtual double SafetyFromOutside(const UVector3 &p, bool aAccurate = false) const;

  virtual double SafetyFromInside(const UVector3 &p, bool aAccurate = false) const;
  virtual UGeometryType GetEntityType() const;

  void SetSolidClosed(const bool t);

  bool GetSolidClosed() const;

  virtual UVector3 GetPointOnSurface() const;

  virtual std::ostream &StreamInfo(std::ostream &os) const;

  virtual double Capacity();
  virtual double SurfaceArea() { return GetSurfaceArea(); }

  inline virtual void GetParametersList(int /*aNumber*/, double * /*aArray*/) const {}
  inline virtual void ComputeBBox(UBBox * /*aBox*/, bool /*aStore = false*/) {}

  inline void SetMaxVoxels(int max) { fVoxels.SetMaxVoxels(max); }

  inline UVoxelizer &GetVoxels() { return fVoxels; }

  virtual VUSolid *Clone() const;

  double GetMinXExtent() const;
  double GetMaxXExtent() const;
  double GetMinYExtent() const;
  double GetMaxYExtent() const;
  double GetMinZExtent() const;
  double GetMaxZExtent() const;

  virtual double DistanceToIn(const UVector3 &p, const UVector3 &v, double aPstep = UUtils::kInfinity) const
  {
    return DistanceToInCore(p, v, aPstep);
  }

  virtual double DistanceToOut(const UVector3 &p, const UVector3 &v, UVector3 &aNormalVector, bool &aConvex,
                               double aPstep = UUtils::kInfinity) const
  {
    return DistanceToOutCore(p, v, aNormalVector, aConvex, aPstep);
  }

  void Extent(UVector3 &aMin, UVector3 &aMax) const;

  int AllocatedMemoryWithoutVoxels();
  int AllocatedMemory();
  void DisplayAllocatedMemory();

private:
  double DistanceToOutNoVoxels(const UVector3 &p, const UVector3 &v, UVector3 &aNormalVector, bool &aConvex,
                               double aPstep = UUtils::kInfinity) const;

  double DistanceToInCandidates(const std::vector<int> &candidates, const UVector3 &aPoint,
                                const UVector3 &aDirection /*, double aPstep, const UBits &bits*/) const;
  void DistanceToOutCandidates(const std::vector<int> &candidates, const UVector3 &aPoint, const UVector3 &direction,
                               double &minDist, UVector3 &minNormal,
                               int &minCandidate /*, double aPstep*/ /*,  UBits &bits*/) const;
  double DistanceToInNoVoxels(const UVector3 &p, const UVector3 &v, double aPstep = UUtils::kInfinity) const;

  void SetExtremeFacets();

  VUSolid::EnumInside InsideNoVoxels(const UVector3 &p) const;
  VUSolid::EnumInside InsideVoxels(const UVector3 &aPoint) const;

  void Voxelize();

  void CreateVertexList();

  void PrecalculateInsides();

  void SetRandomVectors();

  double DistanceToInCore(const UVector3 &p, const UVector3 &v, double aPstep = UUtils::kInfinity) const;
  double DistanceToOutCore(const UVector3 &p, const UVector3 &v, UVector3 &aNormalVector, bool &aConvex,
                           double aPstep = UUtils::kInfinity) const;

  int SetAllUsingStack(const std::vector<int> &voxel, const std::vector<int> &max, bool status, UBits &checked);

  void DeleteObjects();
  void CopyObjects(const UTessellatedSolid &s);

  static bool CompareSortedVoxel(const std::pair<int, double> &l, const std::pair<int, double> &r);

  double MinDistanceFacet(const UVector3 &p, bool simple, VUFacet *&facet) const;

  inline bool OutsideOfExtent(const UVector3 &p, double tolerance = 0) const
  {
    return (p.x() < fMinExtent.x() - tolerance || p.x() > fMaxExtent.x() + tolerance ||
            p.y() < fMinExtent.y() - tolerance || p.y() > fMaxExtent.y() + tolerance ||
            p.z() < fMinExtent.z() - tolerance || p.z() > fMaxExtent.z() + tolerance);
  }

  void Initialize();

private:
  std::vector<VUFacet *> fFacets;
  std::set<VUFacet *> fExtremeFacets; // Does all other facets lie on or behind this surface?

  UGeometryType fGeometryType;
  double fCubicVolume;
  double fSurfaceArea;

  std::vector<UVector3> fVertexList;

  std::set<UVertexInfo, UVertexComparator> fFacetList;

  UVector3 fMinExtent, fMaxExtent;
  bool fSolidClosed;

  static const double dirTolerance;
  std::vector<UVector3> fRandir;

  double fgToleranceHalf;

  int fMaxTries;

  UVoxelizer fVoxels; // voxelized solid

  UBits fInsides;
};
//============== end of USolids-based implementation

#endif // VECGEOM_REPLACE_USOLIDS

#endif

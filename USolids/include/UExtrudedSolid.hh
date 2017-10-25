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
// UExtrudedSolid
//
// Class description:
//
// UExtrudedSolid is a solid which represents the extrusion of an arbitrary
// polygon with fixed outline in the defined Z sections.
// The z-sides of the solid are the scaled versions of the same polygon.
// The solid is implemented as a specification of UTessellatedSolid.
//
// Parameters in the constructor:
// const std::tring& pName             - solid name
// std::vector<UVector2> polygon       - the vertices of the outlined polygon
//                                       defined in clockwise or anti-clockwise
//                                       order
// std::vector<ZSection>               - the z-sections defined by
//                                       z position, offset and scale
//                                       in increasing z-position order
//
// Parameters in the special constructor (for solid with 2 z-sections:
// double halfZ                    - the solid half length in Z
// UVector2 off1                   - offset of the side in -halfZ
// double scale1                   - scale of the side in -halfZ
// UVector2 off2                   - offset of the side in +halfZ
// double scale2                   - scale of the side in -halfZ
//
// 13.08.13 Tatiana Nikitina
//          Created from original implementation in Geant4
// --------------------------------------------------------------------

#ifndef USOLIDS_UExtrudedSolid_HH
#define USOLIDS_UExtrudedSolid_HH

#include <vector>
#include "UVector2.hh"

#ifdef VECGEOM_REPLACE_USOLIDS
//============== here for VecGeom-based implementation
#include "volumes/LogicalVolume.h"
#include "base/Transformation3D.h"
#include "volumes/UnplacedExtruded.h"

#include "volumes/USolidsAdapter.h"

class UExtrudedSolid : public vecgeom::USolidsAdapter<vecgeom::UnplacedExtruded> {

  // just forwards UExtrudedSolid to vecgeom extruded solid
  using Shape_t = vecgeom::UnplacedExtruded;
  using Base_t  = vecgeom::USolidsAdapter<vecgeom::UnplacedExtruded>;

  // inherit all constructors
  using Base_t::Base_t;

public:
  struct ZSection {
    ZSection(double z, UVector2 offset, double scale) : fZ(z), fOffset(offset), fScale(scale) {}

    double fZ;
    UVector2 fOffset;
    double fScale;
  };

  UExtrudedSolid(const std::string & /*pName*/, std::vector<UVector2> polygon, std::vector<ZSection> zsections)
  {
    // General constructor
    size_t Nvert = polygon.size();
    size_t Nsect = zsections.size();

    vecgeom::XtruVertex2 *vertices = new vecgeom::XtruVertex2[Nvert];
    vecgeom::XtruSection *sections = new vecgeom::XtruSection[Nsect];

    for (size_t i = 0; i < Nvert; ++i) {
      vertices[i].x = polygon[i](0);
      vertices[i].y = polygon[i](1);
    }
    for (size_t i = 0; i < Nsect; ++i) {
      sections[i].fOrigin.Set(zsections[i].fOffset[0], zsections[i].fOffset[1], zsections[i].fZ);
      sections[i].fScale = zsections[i].fScale;
    }
    Initialize(Nvert, vertices, Nsect, sections);
    delete[] vertices;
    delete[] sections;
  }

  UExtrudedSolid(const std::string & /*pName*/, std::vector<UVector2> polygon, double halfZ, UVector2 off1,
                 double scale1, UVector2 off2, double scale2)
  {
    // Special constructor for 2 sections
    size_t Nvert = polygon.size();
    size_t Nsect = 2;

    vecgeom::XtruVertex2 *vertices = new vecgeom::XtruVertex2[Nvert];
    vecgeom::XtruSection *sections = new vecgeom::XtruSection[Nsect];

    for (size_t i = 0; i < Nvert; ++i) {
      vertices[i].x = polygon[i](0);
      vertices[i].y = polygon[i](1);
    }
    sections[0].fOrigin.Set(off1[0], off1[1], -halfZ);
    sections[0].fScale = scale1;
    sections[1].fOrigin.Set(off2[0], off1[2], halfZ);
    sections[1].fScale = scale2;
    Initialize(Nvert, vertices, Nsect, sections);
    delete[] vertices;
    delete[] sections;
  }

  virtual ~UExtrudedSolid() {}

  int GetNofVertices() const { return GetNVertices(); }

  UVector2 GetVertex(int index) const
  {
    UVector2 vert;
    UnplacedExtruded::GetVertex(index, vert[0], vert[1]);
    return vert;
  }

  std::vector<UVector2> GetPolygon() const
  {
    std::vector<UVector2> polygon;
    for (size_t i = 0; i < GetNVertices(); ++i)
      polygon.push_back(GetVertex(i));
    return polygon;
  }

  int GetNofZSections() const { return GetNSections(); }

  ZSection GetZSection(int index) const
  {
    vecgeom::XtruSection sect = GetSection(index);
    return ZSection(sect.fOrigin[2], UVector2(sect.fOrigin[0], sect.fOrigin[1]), sect.fScale);
  }

  std::vector<ZSection> GetZSections() const
  {
    std::vector<ZSection> sections;
    for (size_t i = 0; i < GetNSections(); ++i) {
      vecgeom::XtruSection sect = GetSection(i);
      sections.push_back(ZSection(sect.fOrigin[2], UVector2(sect.fOrigin[0], sect.fOrigin[1]), sect.fScale));
    }
    return sections;
  }

  UGeometryType GetEntityType() const override { return "UExtrudedSolid"; }

  VUSolid *Clone() const override { return new UExtrudedSolid(GetName().c_str()); }

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
};

#else // UExtrudedSolid USolids implementation

#include "UTessellatedSolid.hh"

class VUFacet;

class UExtrudedSolid : public UTessellatedSolid {

public:
  struct ZSection {
    ZSection(double z, UVector2 offset, double scale) : fZ(z), fOffset(offset), fScale(scale) {}

    double fZ;
    UVector2 fOffset;
    double fScale;
  };

public:
  UExtrudedSolid(const std::string &pName, std::vector<UVector2> polygon, std::vector<ZSection> zsections);
  // General constructor

  UExtrudedSolid(const std::string &pName, std::vector<UVector2> polygon, double halfZ, UVector2 off1, double scale1,
                 UVector2 off2, double scale2);
  // Special constructor for solid with 2 z-sections

  virtual ~UExtrudedSolid();
  // Destructor

  // Accessors

  inline int GetNofVertices() const;
  inline UVector2 GetVertex(int index) const;
  inline std::vector<UVector2> GetPolygon() const;

  inline int GetNofZSections() const;
  inline ZSection GetZSection(int index) const;
  inline std::vector<ZSection> GetZSections() const;

  // Solid methods

  EnumInside Inside(const UVector3 &aPoint) const;
  double DistanceToOut(const UVector3 &aPoint, const UVector3 &aDirection, UVector3 &aNormalVector, bool &aConvex,
                       double aPstep = UUtils::kInfinity) const;
  double SafetyFromInside(const UVector3 &aPoint, bool aAccurate = false) const;

  UGeometryType GetEntityType() const;
  VUSolid *Clone() const;

  std::ostream &StreamInfo(std::ostream &os) const;

public:
  UExtrudedSolid();
  // Fake default constructor for usage restricted to direct object
  // persistency for clients requiring preallocation of memory for
  // persistifiable objects.

  UExtrudedSolid(const UExtrudedSolid &rhs);
  UExtrudedSolid &operator=(const UExtrudedSolid &rhs);
  // Copy constructor and assignment operator.

  void Initialise(std::vector<UVector2> &polygon, std::vector<ZSection> &zsections);
  void Initialise(std::vector<UVector2> &polygon, double dz, UVector2 off1, double scale1, UVector2 off2,
                  double scale2);
  // Initialisation methods for constructors.

private:
  void ComputeProjectionParameters();

  UVector3 GetVertex(int iz, int ind) const;
  UVector2 ProjectPoint(const UVector3 &point) const;

  bool IsSameLine(UVector2 p, UVector2 l1, UVector2 l2) const;
  bool IsSameLineSegment(UVector2 p, UVector2 l1, UVector2 l2) const;
  bool IsSameSide(UVector2 p1, UVector2 p2, UVector2 l1, UVector2 l2) const;
  bool IsPointInside(UVector2 a, UVector2 b, UVector2 c, UVector2 p) const;
  double GetAngle(UVector2 p0, UVector2 pa, UVector2 pb) const;

  VUFacet *MakeDownFacet(int ind1, int ind2, int ind3) const;
  VUFacet *MakeUpFacet(int ind1, int ind2, int ind3) const;

  bool AddGeneralPolygonFacets();
  bool MakeFacets();
  bool IsConvex() const;

private:
  int fNv;
  int fNz;
  std::vector<UVector2> fPolygon;
  std::vector<ZSection> fZSections;
  std::vector<std::vector<int>> fTriangles;
  bool fIsConvex;
  UGeometryType fGeometryType;

  std::vector<double> fKScales;
  std::vector<double> fScale0s;
  std::vector<UVector2> fKOffsets;
  std::vector<UVector2> fOffset0s;
};

#include "UExtrudedSolid.icc"
#endif // UExtrudedSolid

#endif

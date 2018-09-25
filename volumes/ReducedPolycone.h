/*
 * ReducedPolycone.h
 *
 *  Created on: Sep 20, 2018
 *      Author: rasehgal
 */

#ifndef VOLUMES_REDUCEDPOLYCONE_H_
#define VOLUMES_REDUCEDPOLYCONE_H_

#include "base/Vector2D.h"
#include "base/Vector.h"
#include <iostream>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct Line2D;);
VECGEOM_DEVICE_DECLARE_CONV(struct, Line2D);
VECGEOM_DEVICE_FORWARD_DECLARE(struct Section;);
VECGEOM_DEVICE_DECLARE_CONV(struct, Section);
VECGEOM_DEVICE_FORWARD_DECLARE(class ReducedPolycone;);
VECGEOM_DEVICE_DECLARE_CONV(class, ReducedPolycone);

inline namespace VECGEOM_IMPL_NAMESPACE {

// Just a Container for 2D Line
struct Line2D {
  Vector2D<Precision> p1;
  Vector2D<Precision> p2;

  VECCORE_ATT_HOST_DEVICE
  Line2D() {}

  VECCORE_ATT_HOST_DEVICE
  ~Line2D() {}

  VECCORE_ATT_HOST_DEVICE
  Line2D(Vector2D<Precision> p1v, Vector2D<Precision> p2v)
  {
    p1 = p1v;
    p2 = p2v;
  }

  void Print()
  {
    std::cout << p1 << std::endl;
    std::cout << p2 << std::endl;
  }

  VECCORE_ATT_HOST_DEVICE
  Line2D(Precision rmax, Precision zval)
  {
    p1 = Vector2D<Precision>(0, zval);
    p2 = Vector2D<Precision>(rmax, zval);
  }
};

// Just a container for a Section
struct Section {
  Precision rMin1;
  Precision rMax1;
  Precision rMin2;
  Precision rMax2;
  Precision z1;
  Precision z2;

  VECCORE_ATT_HOST_DEVICE
  Section() {}

  VECCORE_ATT_HOST_DEVICE
  ~Section() {}

  VECCORE_ATT_HOST_DEVICE
  Section(Precision rmin1, Precision rmax1, Precision Z1, Precision rmin2, Precision rmax2, Precision Z2)
      : rMin1(rmin1), rMax1(rmax1), rMin2(rmin2), rMax2(rmax2), z1(Z1), z2(Z2)
  {
  }

  void Print()
  {

    std::cout << "Rmin1 : " << rMin1 << " :: Rmax1 : " << rMax1 << " :: Z1 : " << z1 << std::endl
              << "Rmin2 : " << rMin2 << " :: Rmax2 : " << rMax2 << " :: Z2 : " << z2 << std::endl;
  }
};

class ReducedPolycone {
  Vector<Vector2D<Precision>> fRZVect;
  Precision fRMax;
  Vector<Section> fSectionVect;

public:
  VECCORE_ATT_HOST_DEVICE
  ReducedPolycone(Vector<Vector2D<Precision>> rzVect);
  VECCORE_ATT_HOST_DEVICE
  ~ReducedPolycone();
  VECCORE_ATT_HOST_DEVICE
  void SetRZ(Vector<Vector2D<Precision>>);
  VECCORE_ATT_HOST_DEVICE
  void SetRMax();
  VECCORE_ATT_HOST_DEVICE
  void ConvertToUniqueVector(Vector<Precision> &vect);
  VECCORE_ATT_HOST_DEVICE
  Vector<Precision> GetUniqueZVector();
  VECCORE_ATT_HOST_DEVICE
  void PrintVector(Vector<Precision> vect);
  VECCORE_ATT_HOST_DEVICE
  Vector<Line2D> GetLineVector();
  VECCORE_ATT_HOST_DEVICE
  void CalcPoIVectorFor2DPolygon(Vector<Vector2D<Precision>> &poiVect, Vector<Precision> z);
  VECCORE_ATT_HOST_DEVICE
  bool ContourCheck(Vector<Precision> z);
  VECCORE_ATT_HOST_DEVICE
  bool Contour(Vector<Precision> z);
  VECCORE_ATT_HOST_DEVICE
  bool PointExist(Vector2D<Precision> pt);
  VECCORE_ATT_HOST_DEVICE
  void CreateNewContour();
  VECCORE_ATT_HOST_DEVICE
  void ProcessContour(Vector<Precision> z);
  VECCORE_ATT_HOST_DEVICE
  Vector<Vector<Precision>> GetRandZVectorAtDiffZ(Vector<Vector2D<Precision>> poiVect, Vector<Precision> &dz);
  VECCORE_ATT_HOST_DEVICE
  void CreateSectionFromTwoLines(Line2D l1, Line2D l2);
  VECCORE_ATT_HOST_DEVICE
  void Swap(Precision &a, Precision &b);
  VECCORE_ATT_HOST_DEVICE
  bool Check();
  VECCORE_ATT_HOST_DEVICE
  bool GetPolyconeParameters(Vector<Precision> &rmin, Vector<Precision> &rmax, Vector<Precision> &z);

  VECCORE_ATT_HOST_DEVICE
  bool GetLineIntersection(Precision p0_x, Precision p0_y, Precision p1_x, Precision p1_y, Precision p2_x,
                           Precision p2_y, Precision p3_x, Precision p3_y, Precision *i_x, Precision *i_y);

  VECCORE_ATT_HOST_DEVICE
  bool GetLineIntersection(Line2D l1, Line2D l2);

  VECCORE_ATT_HOST_DEVICE
  bool GetLineIntersection(Line2D l1, Line2D l2, Vector2D<Precision> &poi)
  {
    // Vector2D<Precision>  poi(0.,0.);
    if (l1.p2.x() == l2.p1.x() && l1.p2.y() == l2.p1.y()) {
      poi.x() = l1.p2.x();
      poi.y() = l1.p2.y();
      return true;
    }
    return GetLineIntersection(l1.p1.x(), l1.p1.y(), l1.p2.x(), l1.p2.y(), l2.p1.x(), l2.p1.y(), l2.p2.x(), l2.p2.y(),
                               &poi.x(), &poi.y());
  }
};

} // namespace VECGEOM_IMPL_NAMESPACE
} /* namespace vecgeom */

#endif /* VOLUMES_REDUCEDPOLYCONE_H_ */

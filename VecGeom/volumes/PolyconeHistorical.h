/*
 * PolyconeHistorical.h
 *
 *  Created on: Apr 27, 2017
 *      Author: rsehgal
 */

#ifndef VOLUMES_POLYCONEHISTORICAL_H_
#define VOLUMES_POLYCONEHISTORICAL_H_

/*
#include "VecGeom/base/Global.h"
#include "VecGeom/base/AlignedBase.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/UnplacedCone.h"
#include "VecGeom/base/Vector.h"
#include <vector>
#include "VecGeom/volumes/Wedge.h"
*/
namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

class PolyconeHistorical {
public:
  VECCORE_ATT_HOST_DEVICE
  PolyconeHistorical() : fHStart_angle(0.), fHOpening_angle(0.), fHNum_z_planes(0), fHZ_values(0), fHRmin(0), fHRmax(0)
  {
  }
  VECCORE_ATT_HOST_DEVICE
  PolyconeHistorical(int z_planes) : fHStart_angle(0.), fHOpening_angle(0.), fHNum_z_planes(z_planes)
  {
    fHZ_values = new double[z_planes];
    fHRmin     = new double[z_planes];
    fHRmax     = new double[z_planes];
    for (int i = 0; i < z_planes; i++) {
      fHZ_values[i] = 0.0;
      fHRmin[i]     = 0.0;
      fHRmax[i]     = 0.0;
    }
  }
  VECCORE_ATT_HOST_DEVICE
  ~PolyconeHistorical()
  {
    delete[] fHZ_values;
    delete[] fHRmin;
    delete[] fHRmax;
  }
  VECCORE_ATT_HOST_DEVICE
  PolyconeHistorical(const PolyconeHistorical &source)
  {
    fHStart_angle   = source.fHStart_angle;
    fHOpening_angle = source.fHOpening_angle;
    fHNum_z_planes  = source.fHNum_z_planes;

    fHZ_values = new double[fHNum_z_planes];
    fHRmin     = new double[fHNum_z_planes];
    fHRmax     = new double[fHNum_z_planes];

    for (int i = 0; i < fHNum_z_planes; i++) {
      fHZ_values[i] = source.fHZ_values[i];
      fHRmin[i]     = source.fHRmin[i];
      fHRmax[i]     = source.fHRmax[i];
    }
  }
  VECCORE_ATT_HOST_DEVICE
  PolyconeHistorical &operator=(const PolyconeHistorical &right)
  {
    if (&right == this) return *this;

    fHStart_angle   = right.fHStart_angle;
    fHOpening_angle = right.fHOpening_angle;
    fHNum_z_planes  = right.fHNum_z_planes;

    delete[] fHZ_values;
    delete[] fHRmin;
    delete[] fHRmax;
    fHZ_values = new double[fHNum_z_planes];
    fHRmin     = new double[fHNum_z_planes];
    fHRmax     = new double[fHNum_z_planes];

    for (int i = 0; i < fHNum_z_planes; i++) {
      fHZ_values[i] = right.fHZ_values[i];
      fHRmin[i]     = right.fHRmin[i];
      fHRmax[i]     = right.fHRmax[i];
    }

    return *this;
  }

  double fHStart_angle;
  double fHOpening_angle;
  int fHNum_z_planes;
  double *fHZ_values;
  double *fHRmin;
  double *fHRmax;
};
}
}

#endif /* VOLUMES_POLYCONEHISTORICAL_H_ */

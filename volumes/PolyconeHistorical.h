/*
 * PolyconeHistorical.h
 *
 *  Created on: Apr 27, 2017
 *      Author: rsehgal
 */

#ifndef VOLUMES_POLYCONEHISTORICAL_H_
#define VOLUMES_POLYCONEHISTORICAL_H_

/*
#include "base/Global.h"
#include "base/AlignedBase.h"
#include "base/Vector3D.h"
#include "volumes/UnplacedVolume.h"
#include "volumes/UnplacedCone.h"
#include "base/Vector.h"
#include <vector>
#include "volumes/Wedge.h"
*/
namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

class PolyconeHistorical {
public:
  PolyconeHistorical() : fHStart_angle(0.), fHOpening_angle(0.), fHNum_z_planes(0), fHZ_values(0), fHRmin(0), fHRmax(0)
  {
  }
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
  ~PolyconeHistorical()
  {
    delete[] fHZ_values;
    delete[] fHRmin;
    delete[] fHRmax;
  }
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

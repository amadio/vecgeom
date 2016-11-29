//
// Implementation of the batch solid  test
//

#include "base/RNG.h"

#include <iomanip>
#include <sstream>
#include <ctime>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "ShapeTester.h"
#include "volumes/PlacedVolume.h"
#include "VUSolid.hh"
#include "UTransform3D.hh"

#include "base/Vector3D.h"
#include "volumes/Box.h"

#ifdef VECGEOM_ROOT
#include "TGraph2D.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TGeoManager.h"
#include "TPolyMarker3D.h"
#include "TRandom3.h"
#include "TColor.h"
#include "TROOT.h"
#include "TAttMarker.h"
#include "TGraph.h"
#include "TLegend.h"
#include "TH1D.h"
#include "TH2F.h"
#include "TF1.h"
#include "TVirtualPad.h"
#include "TView3D.h"
#include "management/RootGeoManager.h"
#endif

using namespace std;
using namespace vecgeom;

/* The definitions of core ShapeTester functions are not modified,
 * Only duplicate tests which are now implemented in Convention checker
 * are removed. These are basically tests for wrong side points.
 *
 */

ShapeTester::ShapeTester()
{
  SetDefaults();
}
// ShapeTester::ShapeTester(bool val) { SetDefaults(); fVisualizer = new vecgeom::Visualizer(val); fVisualize = val; }

ShapeTester::~ShapeTester()
{
}

void ShapeTester::SetDefaults()
{
  fNumDisp   = 2;
  fMaxPoints = 10000;
  fVerbose   = 1;
  // fRepeat = 1000;
  fInsidePercent  = 100.0 / 3;
  fOutsidePercent = 100.0 / 3;
  fEdgePercent    = 0;

  fOutsideMaxRadiusMultiple      = 10;
  fOutsideRandomDirectionPercent = 50;
  fIfSaveAllData                 = true;
  fIfMoreTests                   = true;
  fIfDifUSolids                  = true;
  fMinDifference                 = VUSolid::Tolerance();

  fDefinedNormal = false;
  fIfException   = false;

  fMethod        = "all";
  fVolumeUSolids = NULL;

  fGCapacitySampled    = 0;
  fGCapacityError      = 0;
  fGCapacityAnalytical = 0;
  fGNumberOfScans      = 15;

  //
  // Zero error list
  //
  fErrorList = 0;

  fVisualize          = false;
  fSolidTolerance     = vecgeom::kTolerance;
  fStat               = false;
  fTestBoundaryErrors = true;
  fDebug              = false;
}

void ShapeTester::EnableDebugger(bool val)
{
  fVisualize = val;
}

UVector3 ShapeTester::GetRandomDirection()
{
  double phi   = 2. * UUtils::kPi * UUtils::Random();
  double theta = UUtils::ACos(1. - 2. * UUtils::Random());
  double vx    = std::sin(theta) * std::cos(phi);
  double vy    = std::sin(theta) * std::sin(phi);
  double vz    = std::cos(theta);
  UVector3 vec(vx, vy, vz);
  vec.Normalize();

  return vec;
}

UVector3 ShapeTester::GetPointOnOrb(double r)
{
  double phi   = 2. * UUtils::kPi * UUtils::Random();
  double theta = UUtils::ACos(1. - 2. * UUtils::Random());
  double vx    = std::sin(theta) * std::cos(phi);
  double vy    = std::sin(theta) * std::sin(phi);
  double vz    = std::cos(theta);
  UVector3 vec(vx, vy, vz);
  vec.Normalize();
  vec = vec * r;
  return vec;
}

// DONE: all set point Methods are performance equivalent

int ShapeTester::TestBoundaryPrecision(int mode)
{
  // Testing of boundary precision.
  // Supported modes:
  //   0 - default mode computing a boundary tolerance standard deviation
  //       averaged on random directions and distances

  int nsamples        = 1000;
  int errCode         = 0;
  int nError          = 0;
  constexpr int ndist = 8;
  double dtest;
  double maxerr;
  double ndotvmin = 0.2; // avoid directions parallel to surface
  std::cout << "# Testing boundary precision\n";
  double x[ndist];
#ifdef VECGEOM_ROOT
  double y[ndist];
  TCanvas *cerrors = new TCanvas("cerrors", "Boundary precision", 1200, 800);
  TLegend *legend  = new TLegend(0.12, 0.75, 0.32, 0.87);
  legend->SetLineColor(0);
#endif
  // Generate several "move away" distances
  dtest = 1.e-3;
  for (int idist = 0; idist < ndist; ++idist) {
    maxerr = 0.;
    dtest *= 10.;
    x[idist] = dtest;
    for (int i = 0; i < fMaxPointsSurface + fMaxPointsEdge; ++i) {
      // Initial point on surface.
      UVector3 point = fPoints[fOffsetSurface + i];
      // Make sure point is on surface
      if (fVolumeUSolids->Inside(point) != vecgeom::EInside::kSurface) {
        // Do not report the error here - it is tested in TestSurfacePoint
        continue;
      }
      // Compute normal to surface in this point
      UVector3 norm, v;
      bool valid = fVolumeUSolids->Normal(point, norm);
      if (!valid) continue;
      // Test boundary tolerance when coming from outside from distance = 1.
      for (int isample = 0; isample < nsamples; ++isample) {
        // Generate a random direction outwards the solid, then
        // move the point from boundary outwards with distance = 1, making sure
        // that the new point lies outside.
        UVector3 pout;
        int ntries = 0;
        while (1) {
          if (ntries == 1000) {
            errCode = 1; // do we have a rule coding the error number?
            ReportError(&nError, point, norm, 1., "TBE: Cannot reach outside from surface when "
                                                  "propagating with unit distance after 1000 tries.");
            return errCode;
          }
          ntries++;
          // Random direction outwards
          v = GetRandomDirection();
          if (norm.Dot(v) < ndotvmin) continue;
          // Move the point from boundary outwards with distance = dtest.
          pout = point + dtest * v;
          // Cross-check that the point is actually outside
          if (fVolumeUSolids->Inside(pout) == vecgeom::EInside::kOutside) break;
        }
        // Compute distance back to boundary.
        double dunit = fVolumeUSolids->DistanceToIn(pout, -v);
        // Compute rounded boundary error (along normal)
        double error = (dunit - dtest) * norm.Dot(v);
        // Ignore large errors which can be due to missing the shape or by
        // shooting from inner boundaries
        if (Abs(error) < 1.e-1 && Abs(error) > maxerr) maxerr = Abs(error);
      }
    }
#ifdef VECGEOM_ROOT
    y[idist] = maxerr;
#endif
    std::cout << "==    error[dist = " << x[idist] << "] = " << maxerr << std::endl;
  }
#ifdef VECGEOM_ROOT
  TGraph *grerrdist = new TGraph(ndist, x, y);
  grerrdist->SetTitle("DistanceToIn error on boundary propagation");
  grerrdist->GetXaxis()->SetTitle("distance (internal unit)");
  grerrdist->GetYaxis()->SetTitle("Max sampled propagation error");
  cerrors->SetGridy();
  cerrors->SetLogx();
  cerrors->SetLogy();
  grerrdist->Draw("AL*");
  grerrdist->SetMarkerColor(kRed);
  grerrdist->SetMarkerSize(2);
  grerrdist->SetMarkerStyle(20);
  grerrdist->SetLineColor(kRed);
  grerrdist->SetLineWidth(2);
  //  grerrdist->GetYaxis()->SetRangeUser(1.e-16,1.e-1);
  // legend->AddEntry(grerrdist, fVolumeUSolids->GetEntityType().c_str(), "lpe");
  legend->Draw();
  char name[100];
  // sprintf(name, "%s_errors.gif", fVolumeUSolids->GetEntityType().c_str());
  cerrors->SaveAs(name);
  // sprintf(name, "%s_errors.root", fVolumeUSolids->GetEntityType().c_str());
  cerrors->SaveAs(name);
#endif
  return errCode;
}

int ShapeTester::TestConsistencySolids()
{
  int errCode = 0;

  std::cout << "% Performing CONSISTENCY TESTS: ConsistencyTests for Inside, Outside and Surface fPoints " << std::endl;

  errCode += TestInsidePoint();
  errCode += TestOutsidePoint();
  errCode += TestSurfacePoint();

  if (fIfSaveAllData) {
    UVector3 point;
    for (int i = 0; i < fMaxPoints; i++) {
      GetVectorUSolids(point, fPoints, i);
      VUSolid::EnumInside inside = fVolumeUSolids->Inside(point);
      fResultDoubleUSolids[i]    = (double)inside;
    }
    SaveResultsToFile("Inside");
  }
  return errCode;
}

int ShapeTester::ShapeNormal()
{
  int errCode = 0;
  int nError  = 0;
  ClearErrors();
  int i;
  int numTrials = 1000;
#ifdef VECGEOM_ROOT
  // Visualisation
  TPolyMarker3D *pm2 = 0;
  pm2                = new TPolyMarker3D();
  pm2->SetMarkerSize(0.02);
  pm2->SetMarkerColor(kBlue);
#endif
  UVector3 minExtent, maxExtent;
  fVolumeUSolids->Extent(minExtent, maxExtent);
  double maxX   = std::max(std::fabs(maxExtent.x()), std::fabs(minExtent.x()));
  double maxY   = std::max(std::fabs(maxExtent.y()), std::fabs(minExtent.y()));
  double maxZ   = std::max(std::fabs(maxExtent.z()), std::fabs(minExtent.z()));
  double maxXYZ = 2 * std::sqrt(maxX * maxX + maxY * maxY + maxZ * maxZ);
  double step   = maxXYZ * fSolidTolerance;
  for (i = 0; i < fMaxPointsInside; i++) {
    // Initial point is inside
    UVector3 point = fPoints[i + fOffsetInside];
    UVector3 dir   = fDirections[i + fOffsetInside];
    UVector3 norm;
    bool convex;

    VUSolid::EnumInside inside;
    int count   = 0;
    double dist = fVolumeUSolids->DistanceToOut(point, dir, norm, convex);
    // Propagate on boundary
    point = point + dist * dir;
    for (int j = 0; j < numTrials; j++) {
      UVector3 dir_new;
      do {
        // Generate a random direction from the point on boundary
        dir_new = GetRandomDirection();
        // We expect that if we propagate with the shape tolerance
        // corrected by the shooting distance, at least on some directions
        // the new point will be inside
        inside = fVolumeUSolids->Inside(point + dir_new * step);
        count++;
      } while ((inside != vecgeom::EInside::kInside) && (count < 1000));

      if (count >= 1000) {
        ReportError(&nError, point, dir_new, 0,
                    "SN: Can not reach inside solid "
                    "from point on boudary after propagation with tolerance after 1000 trials");
        break;
      }
      count = 0;
      // Propagate the point to new location close to boundary, but inside
      point += dir_new * step;
      // Now shoot along the direction that just crossed the surface and expect
      // to find a distance bigger than the tolerance
      dist = fVolumeUSolids->DistanceToOut(point, dir_new, norm, convex);
      if (dist < VUSolid::Tolerance()) {
        ReportError(&nError, point, dir_new, dist,
                    "SN: DistanceToOut has to be  bigger than tolerance for point Inside");
      }
      // Distance to exit should not be infinity
      if (dist >= UUtils::kInfinity) {
        ReportError(&nError, point, dir_new, dist, "SN: DistanceToOut has to be finite number");
      }
      // The normal vector direction at the exit point has to point outwards
      double dot = norm.Dot(dir_new);
      if (dot < 0.) {
        ReportError(&nError, point, dir_new, dot, "SN: Wrong direction of Normal calculated by DistanceToOut");
      }
      // Propagate the point to the exiting surface and compute normal vector
      // using the Normal method
      point = point + dist * dir_new;
      if (fDefinedNormal) {
        UVector3 normal;
        bool valid = fVolumeUSolids->Normal(point, normal);
        if (!valid) ReportError(&nError, point, dir_new, 0, "SN: Normal has to be valid for point on the Surface");
        dot = normal.Dot(dir_new);
        // Normal has to point outwards
        if (dot < 0.) {
          ReportError(&nError, point, dir_new, dot, "SN: Wrong direction of Normal calculated by Normal");
        }
      }
#ifdef VECGEOM_ROOT
      // visualisation
      pm2->SetNextPoint(point.x(), point.y(), point.z());
#endif
      // Check if exiting point is actually on surface
      if (fVolumeUSolids->Inside(point) == vecgeom::EInside::kOutside) {
        ReportError(&nError, point, dir_new, 0, "SN: DistanceToOut is overshooting,  new point must be on the Surface");
        break;
      }
      if (fVolumeUSolids->Inside(point) == vecgeom::EInside::kInside) {
        ReportError(&nError, point, dir_new, 0,
                    "SN: DistanceToOut is undershooting,  new point must be on the Surface");
        break;
      }
      // Compute safety from point on boundary - they should be no more than
      //  the solid tolerance
      double safFromIn  = fVolumeUSolids->SafetyFromInside(point);
      double safFromOut = fVolumeUSolids->SafetyFromOutside(point);
      if (safFromIn > fSolidTolerance)
        ReportError(&nError, point, dir_new, safFromIn, "SN: SafetyFromInside must be less than tolerance on Surface ");
      if (safFromOut > fSolidTolerance)
        ReportError(&nError, point, dir_new, safFromOut,
                    "SN: SafetyFromOutside must be less than tolerance on Surface");
    }
  }

#ifdef VECGEOM_ROOT
  // visualisation
  if (fStat) {
    new TCanvas("shape03", "ShapeNormals", 1000, 800);
    pm2->Draw();
  }
#endif
  std::cout << "% " << std::endl;
  std::cout << "% TestShapeNormal reported = " << CountErrors() << " errors" << std::endl;
  std::cout << "% " << std::endl;

  if (CountErrors()) errCode = 256; // errCode: 0001 0000 0000
  return errCode;
}

int ShapeTester::ShapeDistances()
{
  int errCode = 0;
  int i;
  int nError = 0;
  ClearErrors();
  double maxDifOut = 0, maxDifIn = 0., delta = 0., tolerance = fSolidTolerance;
  bool convex, convex2;
  bool globalConvex = true;
  if (dynamic_cast<VPlacedVolume *>(fVolumeUSolids))
    globalConvex = dynamic_cast<VPlacedVolume *>(fVolumeUSolids)->GetUnplacedVolume()->IsConvex();
  std::cout << "globalConvex = " << globalConvex << std::endl;
  UVector3 norm;
  UVector3 minExtent, maxExtent;

  fVolumeUSolids->Extent(minExtent, maxExtent);
  double maxX   = std::max(std::fabs(maxExtent.x()), std::fabs(minExtent.x()));
  double maxY   = std::max(std::fabs(maxExtent.y()), std::fabs(minExtent.y()));
  double maxZ   = std::max(std::fabs(maxExtent.z()), std::fabs(minExtent.z()));
  double maxXYZ = 2 * std::sqrt(maxX * maxX + maxY * maxY + maxZ * maxZ);
  double dmove  = maxXYZ;

#ifdef VECGEOM_ROOT
  // Histograms
  TH1D *hist1 = new TH1D("Residual", "Residual DistancetoIn/Out", 200, -20, 0);
  hist1->GetXaxis()->SetTitle("delta[mm] - first bin=overflow");
  hist1->GetYaxis()->SetTitle("count");
  hist1->SetMarkerStyle(kFullCircle);
  TH1D *hist2 = new TH1D("AccuracyIn", "Accuracy distanceToIn for Points near Surface", 200, -20, 0);
  hist2->GetXaxis()->SetTitle("delta[mm] - first bin=overflow");
  hist2->GetYaxis()->SetTitle("count");
  hist2->SetMarkerStyle(kFullCircle);
  TH1D *hist3 = new TH1D("AccuracyOut", "Accuracy distanceToOut for Points near Surface", 200, -20, 0);
  hist3->GetXaxis()->SetTitle("delta[mm] - first bin=overflow");
  hist3->GetYaxis()->SetTitle("count");
  hist3->SetMarkerStyle(kFullCircle);
#endif

  for (i = 0; i < fMaxPointsInside; i++) {
    // Take initial point inside
    UVector3 point = fPoints[i + fOffsetInside];
    UVector3 dir   = fDirections[i + fOffsetInside];
    // Compute distance to outside
    double DistanceOut2 = fVolumeUSolids->DistanceToOut(point, dir, norm, convex2);
    // Compute a new point before boundary
    /*
    Instead of creating new point like
    point + dir * DistanceOut2 * (1. - 10 * tolerance);
    better way is to take the point to surface and then move it back by
    required distance, otherwise for some of the shapes like Hype and Cone
    It will give "DistanceToOut is not precise", or "DistanceToIn is not precise"
    error
    */
    UVector3 pointSurf = point + dir * DistanceOut2;
    UVector3 pointIn   = pointSurf - dir * 10 * tolerance;
    // Compute distance to outside from pointIn
    double DistanceOut = fVolumeUSolids->DistanceToOut(pointIn, dir, norm, convex);
    // Compute a new point just after the boundary outside
    UVector3 pointOut = pointSurf + dir * 10 * tolerance;
    // Now shoot in the opposite direction and compute distance to inside
    double DistanceIn = fVolumeUSolids->DistanceToIn(pointOut, -dir);
    // The distances to the boindary from points near boundary should be small
    if (DistanceOut > 1000. * tolerance)
      ReportError(&nError, pointIn, dir, DistanceOut, "SD: DistanceToOut is not precise");
    if (DistanceIn > 1000. * tolerance)
      ReportError(&nError, pointOut, dir, DistanceIn, "SD: DistanceToIn is not precise ");

    // Calculate distances for convex or non-convex cases, from the point
    // propagated on surface
    double DistanceToInSurf = fVolumeUSolids->DistanceToIn(point + dir * DistanceOut2, dir);
    if (DistanceToInSurf >= UUtils::kInfinity) {
      // The solid is not crossed again, so it may be convex on this surface
      // Aim to move the point outside, but not too far
      dmove = maxXYZ;
      if (globalConvex && !convex2) {
        bool matchConvexity = false;
        UVector3 pointSurf  = point + dir * DistanceOut2;
        // To cross-check convexity, shoot randomly in the attempt to cross
        // again the solid. Note that this check may fail even if the solid is
        // really non-convex on this surface (sampling to be increased)
        for (int k = 0; k < 100; k++) {
          UVector3 rndDir = GetRandomDirection();
          double distTest = fVolumeUSolids->DistanceToIn(pointSurf, rndDir);
          if ((distTest <= UUtils::kInfinity) && (distTest > fSolidTolerance)) {
            matchConvexity = true;
            break;
          }
        }
        // #### Check disabled until the check of convexity from DistanceToOut gets
        // activated ####
        if (!matchConvexity)
          ReportError(&nError, point, dir, DistanceToInSurf, "SD: Error in convexity, must be convex");
      }

    } else {
      // Re-entering solid, it is not convex
      if (globalConvex && convex2)
        ReportError(&nError, point, dir, DistanceToInSurf, "SD: Error in convexity, must be NOT convex");
      // Aim to move the point outside, but not re-enter
      dmove = DistanceOut2 + DistanceToInSurf * 0.5;
    }
    // Shoot back to the solid from point moved outside
    double DistanceToIn2 = fVolumeUSolids->DistanceToIn(point + dir * dmove, -dir);

    if (maxDifOut < DistanceOut) {
      maxDifOut = DistanceOut;
    }
    if ((fVolumeUSolids->Inside(pointOut - dir * DistanceIn) != vecgeom::EInside::kOutside) &&
        (maxDifIn < DistanceIn)) {
      maxDifIn = DistanceIn;
    }

    // dmove should be close to the sum between DistanceOut2 and DistanceIn2
    double difDelta = dmove - DistanceOut2 - DistanceToIn2;
    if (std::fabs(difDelta) > 10. * tolerance)
      ReportError(&nError, point, dir, difDelta, "SD: Distances calculation is not precise");
    if (difDelta > delta) delta = std::fabs(difDelta);

#ifdef VECGEOM_ROOT
    // Histograms
    if (std::fabs(difDelta) < 1E-20) difDelta    = 1E-30;
    if (std::fabs(DistanceIn) < 1E-20) difDelta  = 1E-30;
    if (std::fabs(DistanceOut) < 1E-20) difDelta = 1E-30;
    hist1->Fill(std::max(0.5 * std::log(std::fabs(difDelta)), -20.));
    hist2->Fill(std::max(0.5 * std::log(std::fabs(DistanceIn)), -20.));
    hist3->Fill(std::max(0.5 * std::log(std::fabs(DistanceOut)), -20.));
#endif
  }
  if (fVerbose) {
    std::cout << "% TestShapeDistances:: Accuracy max for DistanceToOut=" << maxDifOut
              << " from asked accuracy eps=" << 10 * tolerance << std::endl;
    std::cout << "% TestShapeDistances:: Accuracy max for DistanceToIn=" << maxDifIn
              << " from asked accuracy eps=" << 10 * tolerance << std::endl;
    std::cout << "% TestShapeDistances:: Accuracy max for Delta=" << delta << std::endl;
  }
  std::cout << "% " << std::endl;
  std::cout << "% TestShapeDistances reported = " << CountErrors() << " errors" << std::endl;
  std::cout << "% " << std::endl;

  if (CountErrors()) errCode = 32; // errCode: 0000 0010 0000

#ifdef VECGEOM_ROOT
  // Histograms
  if (fStat) {
    TCanvas *c4 = new TCanvas("c4", "Residuals DistancsToIn/Out", 800, 600);
    c4->Update();
    hist1->Draw();
    TCanvas *c5 = new TCanvas("c5", "Residuals DistancsToIn", 800, 600);
    c5->Update();
    hist2->Draw();
    TCanvas *c6 = new TCanvas("c6", "Residuals DistancsToOut", 800, 600);
    c6->Update();
    hist3->Draw();
  }
#endif

  return errCode;
}

int ShapeTester::TestNormalSolids()
{
  // This saves the result of Normal method to file
  int errCode = 0;
  UVector3 point, normal;

  for (int i = 0; i < fMaxPoints; i++) {
    GetVectorUSolids(point, fPoints, i);
    bool valid = fVolumeUSolids->Normal(point, normal);
    if (fIfSaveAllData) {
      fResultBoolUSolids[i] = valid;
      SetVectorUSolids(normal, fResultVectorUSolids, i);
    }
  }

  SaveResultsToFile("Normal");

  return errCode;
}

int ShapeTester::TestSafetyFromOutsideSolids()
{
  // This saves the result of SafetyFromOutside method to file
  int errCode = 0;
  std::cout << "% Performing SAFETYFromOUTSIDE TESTS: ShapeSafetyFromOutside " << std::endl;
  errCode += ShapeSafetyFromOutside(1000);

  if (fIfSaveAllData) {
    UVector3 point;
    for (int i = 0; i < fMaxPoints; i++) {
      GetVectorUSolids(point, fPoints, i);
      double res              = fVolumeUSolids->SafetyFromOutside(point, true);
      fResultDoubleUSolids[i] = res;
    }
    SaveResultsToFile("SafetyFromOutside");
  }

  return errCode;
}

int ShapeTester::TestSafetyFromInsideSolids()
{
  // This saves the result of SafetyFromInside method to file
  int errCode = 0;
  std::cout << "% Performing SAFETYFromINSIDE TESTS: ShapeSafetyFromInside " << std::endl;
  errCode += ShapeSafetyFromInside(1000);

  if (fIfSaveAllData) {
    UVector3 point;

    for (int i = 0; i < fMaxPoints; i++) {
      GetVectorUSolids(point, fPoints, i);
      double res              = fVolumeUSolids->SafetyFromInside(point);
      fResultDoubleUSolids[i] = res;
    }

    SaveResultsToFile("SafetyFromInside");
  }

  return errCode;
}

void ShapeTester::PropagatedNormalU(const UVector3 &point, const UVector3 &direction, double distance, UVector3 &normal)
{
  // Compute surface point and correspondinf surface normal after computing
  // the distance to the solid
  normal.Set(0);
  if (distance < UUtils::kInfinity) {
    UVector3 shift        = distance * direction;
    UVector3 surfacePoint = point + shift;
    fVolumeUSolids->Normal(surfacePoint, normal);
  }
}

int ShapeTester::TestDistanceToInSolids()
{
  // Combined test for DistanceToIn
  int errCode = 0;
  std::cout << "% Performing DISTANCEtoIn TESTS: ShapeDistances, TestsAccuracyDistanceToIn and TestFarAwayPoint "
            << std::endl;
  errCode += ShapeDistances();
  errCode += TestAccuracyDistanceToIn(1000.);
  errCode += TestFarAwayPoint();

  if (fIfSaveAllData) {
    UVector3 point, direction;
    for (int i = 0; i < fMaxPoints; i++) {
      GetVectorUSolids(point, fPoints, i);
      GetVectorUSolids(direction, fDirections, i);
      double res              = fVolumeUSolids->DistanceToIn(point, direction);
      fResultDoubleUSolids[i] = res;

      UVector3 normal;
      PropagatedNormalU(point, direction, res, normal);
      SetVectorUSolids(normal, fResultVectorUSolids, i);
    }
    SaveResultsToFile("DistanceToIn");
  }

  return errCode;
}

int ShapeTester::TestDistanceToOutSolids()
{
  // Combined test for DistanceToOut
  int errCode = 0;

  std::cout << "% Performing DISTANCEtoOUT TESTS: Shape Normals " << std::endl;
  errCode += ShapeNormal();

  if (fIfSaveAllData) {

    UVector3 point, normal, direction;
    bool convex;

    for (int i = 0; i < fMaxPoints; i++) {
      GetVectorUSolids(point, fPoints, i);
      GetVectorUSolids(direction, fDirections, i);
      normal.Set(0);
      double res = fVolumeUSolids->DistanceToOut(point, direction, normal, convex);

      fResultDoubleUSolids[i] = res;
      fResultBoolUSolids[i]   = convex;
      SetVectorUSolids(normal, fResultVectorUSolids, i);
    }
  }
  SaveResultsToFile("DistanceToOut");

  return errCode;
}

int ShapeTester::TestFarAwayPoint()
{
  int errCode = 0;
  UVector3 point, point1, vec, direction, normal, pointSurf;
  int icount = 0, icount1 = 0, nError = 0;
  double distIn, diff, difMax = 0., maxDistIn = 0.;
  double tolerance = fSolidTolerance;
  ClearErrors();

  // for ( int j=0; j<fMaxPointsSurface+fMaxPointsEdge; j++)
  for (int j = 0; j < fMaxPointsInside; j++) {
    // point = fPoints[j+fOffsetSurface];
    // Initial point inside
    point = fPoints[j + fOffsetInside];
    vec   = GetRandomDirection();
    // The test below makes no sense: DistanceToIn from inside point should be
    // negative, so the full test would be skipped
    // if (fVolumeUSolids->DistanceToIn(point, vec) < UUtils::kInfinity)
    //  continue;
    point1 = point;

    // Move point far away
    for (int i = 0; i < 10000; i++) {
      point1 = point1 + vec * 10000;
    }
    // Shoot back to solid, then compute point on surface
    distIn                                                              = fVolumeUSolids->DistanceToIn(point1, -vec);
    pointSurf                                                           = point1 - distIn * vec;
    if ((distIn < UUtils::kInfinity) && (distIn > maxDistIn)) maxDistIn = distIn;
    // Compute error and check against the solid tolerance
    diff = std::fabs((point1 - pointSurf).Mag() - distIn);
    if (diff > 100 * tolerance) // Note that moving to 10000 we have cut 4 digits, not just 2
      icount++;
    // If we do not hit back the solid report an error
    if (distIn >= UUtils::kInfinity) {
      icount1++;
      UVector3 temp = -vec;
      ReportError(&nError, point1, temp, diff, "TFA:  Point missed Solid (DistanceToIn = Infinity)");
    } else {
      if (diff > difMax) difMax = diff;
    }
  }
  if (fVerbose) {
    std::cout << "% TestFarAwayPoints:: number of Points with big difference (( DistanceToIn- Dist) ) >  tolerance ="
              << icount << std::endl;
    std::cout << "%  Maxdif = " << difMax << " from MaxDist=" << maxDistIn
              << " Number of fPoints missing Solid (DistanceToIn = Infinity) = " << icount1 << std::endl;
  }
  std::cout << "% " << std::endl;
  std::cout << "% TestFarAwayPoints reported = " << CountErrors() << " errors" << std::endl;
  std::cout << "% " << std::endl;

  if (CountErrors()) errCode = 128; // errCode: 0000 1000 0000

  return errCode;
}

int ShapeTester::TestSurfacePoint()
{
  // Combined tests for surface points
  int errCode = 0;
  UVector3 point, pointSurf, vec, direction, normal;
  bool convex;
  int icount = 0, icount1 = 0;
  double distIn, distOut;
  int iIn = 0, iInNoSurf = 0, iOut = 0, iOutNoSurf = 0;
  double tolerance = fSolidTolerance;
  int nError       = 0;
  ClearErrors();
#ifdef VECGEOM_ROOT
  // Visualisation
  TPolyMarker3D *pm5 = 0;
  pm5                = new TPolyMarker3D();
  pm5->SetMarkerStyle(20);
  pm5->SetMarkerSize(1);
  pm5->SetMarkerColor(kRed);

#endif

  for (int i = 0; i < fMaxPointsSurface + fMaxPointsEdge; i++) { // test GetPointOnSurface()
    // Initial point on surface
    point = fPoints[fOffsetSurface + i];
#ifdef VECGEOM_ROOT
    // visualisation
    pm5->SetNextPoint(point.x(), point.y(), point.z());
#endif
    if (fVolumeUSolids->Inside(point) != vecgeom::EInside::kSurface) {
      icount++;
      UVector3 v(0, 0, 0);
      ReportError(&nError, point, v, 0, "TS:  Point on not on the Surface");
    }
    // test if for point on Surface distIn and distOut are not 0 at the same time
    UVector3 v = GetRandomDirection();
    distIn     = fVolumeUSolids->DistanceToIn(point, v);
    distOut    = fVolumeUSolids->DistanceToOut(point, v, normal, convex);

    if (distIn == 0. && distOut == 0.) {
      icount1++;
      ReportError(&nError, point, v, 0, "TS: DistanceToIn=DistanceToOut=0 for point on Surface");
    }
    // test Accuracy distance for fPoints near Surface
    // The point may be slightly outside or inside
    pointSurf                  = point + v * 10 * tolerance;
    VUSolid::EnumInside inside = fVolumeUSolids->Inside(pointSurf);
    if (inside != vecgeom::EInside::kSurface) {
      if (inside == vecgeom::EInside::kOutside) {
        // Shoot randomly from point slightly outside
        for (int j = 0; j < 1000; j++) {
          vec    = GetRandomDirection();
          distIn = fVolumeUSolids->DistanceToIn(pointSurf, vec);
          if (distIn < UUtils::kInfinity) {
            iIn++;
            // If we hit, propagate on surface and check kSurface
            VUSolid::EnumInside surfaceP = fVolumeUSolids->Inside(pointSurf + distIn * vec);
            if (surfaceP != vecgeom::EInside::kSurface) {
              iInNoSurf++;
              ReportError(&nError, pointSurf, vec, distIn,
                          "TS: Wrong DistToIn for point near Surface (final point not reported on surface)");
            }
          }
        }
      } else {
        // Shoot randomly from point slightly inside
        for (int j = 0; j < 1000; j++) {
          iOut++;
          vec     = GetRandomDirection();
          distOut = fVolumeUSolids->DistanceToOut(pointSurf, vec, normal, convex);
          // If we hit, propagate on surface and check kSurface
          VUSolid::EnumInside surfaceP = fVolumeUSolids->Inside(pointSurf + distOut * vec);
          if (surfaceP != vecgeom::EInside::kSurface) {
            iOutNoSurf++;
            ReportError(&nError, pointSurf, vec, distOut,
                        "TS: Wrong DistToOut for point near Surface (final point not reported on surface)");
          }
        }
      }
    }
  }
  if (fVerbose) {
    std::cout << "% TestSurfacePoints GetPointOnSurface() for Solid  " << fVolumeUSolids->GetName() << " had " << icount
              << " errors" << std::endl;
    std::cout << "% TestSurfacePoints both  DistanceToIN and DistanceToOut ==0 for " << fVolumeUSolids->GetName()
              << " had " << icount1 << " errors" << std::endl;
    std::cout << "% TestSurfacePoints new moved point is not on Surface::iInNoSurf = " << iInNoSurf
              << ";    iOutNoSurf = " << iOutNoSurf << std::endl;
  }
#ifdef VECGEOM_ROOT
  // visualisation
  if (fStat) {
    new TCanvas("shape05", "GetPointOnSurface", 1000, 800);
    pm5->Draw();
  }
#endif
  std::cout << "% " << std::endl;
  std::cout << "% Test Surface Point reported = " << CountErrors() << " errors" << std::endl;
  std::cout << "% " << std::endl;

  if (CountErrors()) errCode = 4; // errCode: 0000 0000 0100

  return errCode;
}

int ShapeTester::TestInsidePoint()
{
  // Combined test for inside points
  int errCode = 0;
  int i, n = fMaxPointsOutside;
  int nError = 0;
  ClearErrors();

  UVector3 minExtent, maxExtent;
  fVolumeUSolids->Extent(minExtent, maxExtent);
  double maxX   = std::max(std::fabs(maxExtent.x()), std::fabs(minExtent.x()));
  double maxY   = std::max(std::fabs(maxExtent.y()), std::fabs(minExtent.y()));
  double maxZ   = std::max(std::fabs(maxExtent.z()), std::fabs(minExtent.z()));
  double maxXYZ = 2 * std::sqrt(maxX * maxX + maxY * maxY + maxZ * maxZ);

  for (int j = 0; j < fMaxPointsInside; j++) {
    // Check values of Safety
    // Initial point inside
    UVector3 point      = fPoints[j + fOffsetInside];
    double safeDistance = fVolumeUSolids->SafetyFromInside(point);
    // Safety from inside should be positive
    if (safeDistance <= 0.0) {
      UVector3 zero(0);
      ReportError(&nError, point, zero, safeDistance, "TI: SafetyFromInside(p) <= 0");

      if (CountErrors()) errCode = 1; // errCode: 0000 0000 0001

      return errCode;
    }
    // Safety from wrong side should be negative (VecGeom) or zero (USolids-compatible)
    double safeDistanceFromOut = fVolumeUSolids->SafetyFromOutside(point);
#ifdef VECGEOM_REPLACE_USOLIDS
    if (safeDistanceFromOut != 0.0) {
      std::string message("TI: SafetyFromOutside(p) should be Zero for Points Inside");
#else
    if (safeDistanceFromOut >= 0.0) {
      std::string message("TI: SafetyFromOutside(p) should be Negative value (-1.) for Points Inside");
#endif
      UVector3 zero(0);
      // disable this message as it is alreay part of ConventionChecker
      // ReportError(&nError, point, zero, safeDistanceFromOut, message.c_str());
      continue;
    }

    // Check values of Extent
    // Every point inside should be also within the extent
    if (point.x() < minExtent.x() || point.x() > maxExtent.x() || point.y() < minExtent.y() ||
        point.y() > maxExtent.y() || point.z() < minExtent.z() || point.z() > maxExtent.z()) {
      UVector3 zero(0);
      ReportError(&nError, point, zero, safeDistance, "TI: Point is outside Extent");
    }

    // Check values with fPoints and fDirections to outside fPoints
    for (i = 0; i < n; i++) {
      UVector3 vr = fPoints[i + fOffsetOutside] - point;
      UVector3 v  = vr.Unit();
      bool valid, convex;
      valid = false;
      UVector3 norm;
      // Shoot towards outside point and compute distance to out
      double dist = fVolumeUSolids->DistanceToOut(point, v, norm, convex);
      double NormalDist;

      NormalDist = fVolumeUSolids->SafetyFromInside(point);
      // Distance to out has to be always smaller than the extent diagonal
      if (dist > maxXYZ) {
        ReportError(&nError, point, v, dist, "TI: DistanceToOut(p,v) > Solid's Extent  dist = ");
        continue;
      }
      // Distance to out has to be positive
      if (dist <= 0) {
        ReportError(&nError, point, v, NormalDist, "TI: DistanceToOut(p,v) <= 0  Normal Dist = ");
        continue;
      }
      // Distance to out cannot be infinite
      if (dist >= UUtils::kInfinity) {
        ReportError(&nError, point, v, safeDistance, "TI: DistanceToOut(p,v) == kInfLength");
        continue;
      }
      // Distance to out from inside point should be bigger than the safety
      if (dist < safeDistance - 1E-10) {
        ReportError(&nError, point, v, safeDistance, "TI: DistanceToOut(p,v) < DistanceToIn(p)");
        continue;
      }

      if (valid) {
        // Check outwards condition
        if (norm.Dot(v) < 0) {
          ReportError(&nError, point, v, safeDistance, "TI: Outgoing normal incorrect");
          continue;
        }
      }
      // DistanceToIn from point on wrong side has to be negative (VecGeom) or zero (USolids-compatible)
      double distIn = fVolumeUSolids->DistanceToIn(point, v);
#ifdef VECGEOM_REPLACE_USOLIDS
      if (distIn != 0.) {
        std::string message("TI: DistanceToIn(p,v) has to be Zero for Inside points (USolids convention).");
#else
      if (distIn >= 0.) {
        std::string message(
            "TI: DistanceToIn(p,v) has to be Negative value (-1.) for Inside points (VecGeom convention).");
#endif
        ReportError(&nError, point, v, distIn, message.c_str());
        continue;
      }
      // Move to the boundary and check
      UVector3 p = point + v * dist;

      VUSolid::EnumInside insideOrNot = fVolumeUSolids->Inside(p);
      // Propagated point with DistanceToOut has to be on boundary
      if (insideOrNot == vecgeom::EInside::kInside) {
        ReportError(&nError, point, v, dist, "TI: DistanceToOut(p,v) undershoots");
        continue;
      }
      if (insideOrNot == vecgeom::EInside::kOutside) {
        ReportError(&nError, point, v, dist, "TI: DistanceToOut(p,v) overshoots");
        continue;
      }
      UVector3 norm1;
      valid = fVolumeUSolids->Normal(p, norm1);

      // Direction of motion should not be inward
      if (norm1.Dot(v) < 0) {
        if (fVolumeUSolids->DistanceToIn(p, v) != 0) {
          ReportError(&nError, p, v, safeDistance, "TI: SurfaceNormal is incorrect");
        }
      } // End Check fPoints and fDirections
    }
  }
  std::cout << "% " << std::endl;
  std::cout << "% TestInsidePoint reported = " << CountErrors() << " errors" << std::endl;
  std::cout << "% " << std::endl;

  if (CountErrors()) errCode = 1; // errCode: 0000 0000 0001

  return errCode;
}

int ShapeTester::TestOutsidePoint()
{
  // Combined test for outside points
  int errCode = 0;
  int i, n = fMaxPointsInside;
  int nError = 0;
  ClearErrors();

  for (int j = 0; j < fMaxPointsOutside; j++) {
    // std::cout<<"ConsistencyOutside check"<<j<<std::endl;
    // Initial point outside
    UVector3 point      = fPoints[j + fOffsetOutside];
    double safeDistance = fVolumeUSolids->SafetyFromOutside(point);
    // Safety has to be positive
    if (safeDistance <= 0.0) {
      UVector3 zero(0);
      ReportError(&nError, point, zero, safeDistance, "TO: SafetyFromOutside(p) <= 0");

      if (CountErrors()) errCode = 2; // errCode: 0000 0000 0010

      return errCode;
    }

    double safeDistanceFromInside = fVolumeUSolids->SafetyFromInside(point);
// Safety from wrong side point has to be negative (VecGeom) or zero (USolids-compatible)
#ifdef VECGEOM_REPLACE_USOLIDS
    if (safeDistanceFromInside != 0.0) {
      std::string msg("TO: SafetyFromInside(p) should be Zero for points Outside (USolids convention)");
#else
    if (safeDistanceFromInside >= 0.0) {
      std::string msg("TO: SafetyFromInside(p) should be Negative value (-1.) for points Outside (VecGeom conv)");
#endif
      UVector3 zero(0);
      // disable this message as it is part of ConventionChecker
      // ReportError(&nError, point, zero, safeDistanceFromInside, msg.c_str());
    }

    for (i = 0; i < n; i++) {
      // Connecting point inside
      UVector3 vr = fPoints[i + fOffsetInside] - point;
      UVector3 v  = vr.Unit();

      double dist = fVolumeUSolids->DistanceToIn(point, v);
      // Distance to inside has to be positive
      if (dist <= 0) {
        ReportError(&nError, point, v, safeDistance, "TO: DistanceToIn(p,v) <= 0");
        continue;
      }
      // Make sure we hit the solid
      if (dist >= UUtils::kInfinity) {
        ReportError(&nError, point, v, safeDistance, "TO: DistanceToIn(p,v) == kInfLength");
        continue;
      }
      // Make sure the distance is bigger than the safety
      if (dist < safeDistance - 1E-10) {
        ReportError(&nError, point, v, safeDistance, "TO: DistanceToIn(p,v) < DistanceToIn(p)");
        continue;
      }

      // Moving the point to the Surface
      UVector3 p                      = point + dist * v;
      VUSolid::EnumInside insideOrNot = fVolumeUSolids->Inside(p);
      // Propagated point has to be on surface
      if (insideOrNot == vecgeom::EInside::kOutside) {
        ReportError(&nError, point, v, dist, "TO: DistanceToIn(p,v) undershoots");
        continue;
      }
      if (insideOrNot == vecgeom::EInside::kInside) {
        ReportError(&nError, point, v, dist, "TO: DistanceToIn(p,v) overshoots");
        continue;
      }

      safeDistance = fVolumeUSolids->SafetyFromOutside(p);
      // The safety from a boundary should not be bigger than the tolerance
      if (safeDistance > fSolidTolerance) {
        ReportError(&nError, p, v, safeDistance, "TO2: SafetyToIn(p) should be zero");
        continue;
      }
      safeDistance = fVolumeUSolids->SafetyFromInside(p);
      if (safeDistance > fSolidTolerance) {
        ReportError(&nError, p, v, safeDistance, "TO2: SafetyToOut(p) should be zero");
        continue;
      }

      dist         = fVolumeUSolids->DistanceToIn(p, v);
      safeDistance = fVolumeUSolids->SafetyFromOutside(p);
      //
      // Beware! We might expect dist to be precisely zero, but this may not
      // be true at corners due to roundoff of the calculation of p = point + dist*v.
      // It should, however, *not* be infinity.
      //
      if (dist >= UUtils::kInfinity) {
        ReportError(&nError, p, v, dist, "TO2: DistanceToIn(p,v) == kInfLength");
        continue;
      }

      bool valid, convex, convex1;
      valid = false;
      UVector3 norm;

      dist = fVolumeUSolids->DistanceToOut(p, v, norm, convex);
      // But distance can be infinity if it is a corner point. Needs to handled carefully.
      // For the time being considering that those situation does not happens.
      if (dist >= UUtils::kInfinity) {
        ReportError(&nError, p, v, dist, "TO2: DistanceToOut(p,v) == kInfLength");
        continue;
      } else if (dist < 0) {
        ReportError(&nError, p, v, dist, "TO2: DistanceToOut(p,v) < 0");
        continue;
      }
      // Check the exiting normal when going outwards
      if (valid) {
        if (norm.Dot(v) < 0) {
          ReportError(&nError, p, v, dist, "TO2: Outgoing normal incorrect");
          continue;
        }
      }

      UVector3 norm1;
      valid = fVolumeUSolids->Normal(p, norm1);
      // Check the entering normal when going inwards
      if (norm1.Dot(v) > 0) {
        ReportError(&nError, p, v, dist, "TO2: Ingoing surfaceNormal is incorrect");
      }

      UVector3 p2 = p + v * dist;

      insideOrNot = fVolumeUSolids->Inside(p2);
      // Propagated point has to be on surface
      if (insideOrNot == vecgeom::EInside::kInside) {
        ReportError(&nError, p, v, dist, "TO2: DistanceToOut(p,v) undershoots");
        continue;
      }
      if (insideOrNot == vecgeom::EInside::kOutside) {
        ReportError(&nError, p, v, dist, "TO2: DistanceToOut(p,v) overshoots");
        continue;
      }

      UVector3 norm2, norm3;
      valid = fVolumeUSolids->Normal(p2, norm2);
      // Normal in exit point
      if (norm2.Dot(v) < 0) {
        if (fVolumeUSolids->DistanceToIn(p2, v) != 0)
          ReportError(&nError, p2, v, dist, "TO2: Outgoing surfaceNormal is incorrect");
      }
      // Check sign agreement on normals given by Normal and DistanceToOut
      if (convex) {
        if (norm.Dot(norm2) < 0.0) {
          ReportError(&nError, p2, v, dist, "TO2: SurfaceNormal and DistanceToOut disagree on normal");
        }
      }

      if (convex) {
        dist = fVolumeUSolids->DistanceToIn(p2, v);
        if (dist == 0) {
          //
          // We may have grazed a corner, which is a problem of design.
          // Check distance out
          //
          dist = fVolumeUSolids->DistanceToOut(p2, v, norm3, convex1);
          if (dist != 0) {
            ReportError(&nError, p, v, dist,
                        "TO2: DistanceToOut incorrectly returns validNorm==true (line of sight)(c)");
            if (nError <= 3) std::cout << "Point on opposite surface: p2=" << p2 << "\n";
            continue;
          }
        } else if (dist != UUtils::kInfinity) {
          // ReportError(  &nError, p, v, safeDistance, "TO2: DistanceToOut incorrectly returns validNorm==true (line of
          // sight)" );
          continue;
        }

        int k;
        for (k = 0; k < n; k++) {
          // for (k = 0; k < 10; k++) {
          UVector3 p2top = fPoints[k + fOffsetInside] - p2;

          if (p2top.Dot(norm) > 0) {
            ReportError(&nError, p, v, safeDistance,
                        "TO2: DistanceToOut incorrectly returns validNorm==true (horizon)");
            continue;
          }
        }
      } // if valid normal
    }   // Loop over inside fPoints

    n = fMaxPointsOutside;

    // ### The test below seems to be a duplicate - check this ####
    for (int l = 0; l < n; l++) {
      UVector3 vr = fPoints[l + fOffsetOutside] - point;
      if (vr.Mag2() < DBL_MIN) continue;

      UVector3 v = vr.Unit();

      double dist = fVolumeUSolids->DistanceToIn(point, v);

      if (dist <= 0) {
        ReportError(&nError, point, v, dist, "TO3: DistanceToIn(p,v) <= 0");
        continue;
      }
      if (dist >= UUtils::kInfinity) {
        // G4cout << "dist == kInfLength" << G4endl ;
        continue;
      }
      if (dist < safeDistance - 1E-10) {
        ReportError(&nError, point, v, safeDistance, "TO3: DistanceToIn(p,v) < DistanceToIn(p)");
        continue;
      }
      UVector3 p = point + dist * v;

      VUSolid::EnumInside insideOrNot = fVolumeUSolids->Inside(p);
      if (insideOrNot == vecgeom::EInside::kOutside) {
        ReportError(&nError, point, v, dist, "TO3: DistanceToIn(p,v) undershoots");
        continue;
      }
      if (insideOrNot == vecgeom::EInside::kInside) {
        ReportError(&nError, point, v, dist, "TO3: DistanceToIn(p,v) overshoots");
        continue;
      }
    } // Loop over outside fPoints
  }
  std::cout << "% " << std::endl;
  std::cout << "% TestOutsidePoint reported = " << CountErrors() << " errors" << std::endl;
  std::cout << "% " << std::endl;

  if (CountErrors()) errCode = 2; // errCode: 0000 0000 0010

  return errCode;
}
//
// Surface Checker
//
int ShapeTester::TestAccuracyDistanceToIn(double dist)
{
  // Test accuracy of DistanceToIn method against required one
  int errCode = 0;
  UVector3 point, pointSurf, pointIn, v, direction, normal;
  bool convex;
  double distIn, distOut;
  double maxDistIn = 0, diff = 0, difMax = 0;
  int nError = 0;
  ClearErrors();
  int iIn = 0, iInNoSurf = 0, iOut = 0, iOutNoSurf = 0;
  int iInInf = 0, iInZero = 0;
  double tolerance = VUSolid::Tolerance();

#ifdef VECGEOM_ROOT
  // Histograms
  TH1D *hist10 = new TH1D("AccuracySurf", "Accuracy DistancetoIn", 200, -20, 0);
  hist10->GetXaxis()->SetTitle("delta[mm] - first bin=overflow");
  hist10->GetYaxis()->SetTitle("count");
  hist10->SetMarkerStyle(kFullCircle);
#endif

  // test Accuracy distance
  for (int i = 0; i < fMaxPointsSurface + fMaxPointsEdge; i++) {

    // test GetPointOnSurface
    pointSurf    = fPoints[i + fOffsetSurface];
    UVector3 vec = GetRandomDirection();

    point = pointSurf + vec * dist;

    VUSolid::EnumInside inside = fVolumeUSolids->Inside(point);

    if (inside != vecgeom::EInside::kSurface) {
      if (inside == vecgeom::EInside::kOutside) {
        distIn = fVolumeUSolids->DistanceToIn(pointSurf, vec);
        if (distIn >= UUtils::kInfinity) {
          // Accuracy Test for convex part
          distIn                            = fVolumeUSolids->DistanceToIn(point, -vec);
          if (maxDistIn < distIn) maxDistIn = distIn;
          diff                              = ((pointSurf - point).Mag() - distIn);
          if (diff > difMax) difMax         = diff;
          if (std::fabs(diff) < 1E-20) diff = 1E-30;
#ifdef VECGEOM_ROOT
          hist10->Fill(std::max(0.5 * std::log(std::fabs(diff)), -20.));
#endif
        }

        // Test for consistency for fPoints situated Outside
        for (int j = 0; j < 1000; j++) {
          vec = GetRandomDirection();

          distIn  = fVolumeUSolids->DistanceToIn(point, vec);
          distOut = fVolumeUSolids->DistanceToOut(point, vec, normal, convex);

          // Test for consistency for fPoints situated Inside
          pointIn = pointSurf + vec * 1000. * VUSolid::Tolerance();
          if (fVolumeUSolids->Inside(pointIn) == vecgeom::EInside::kInside) {
            double distOut1              = fVolumeUSolids->DistanceToOut(pointIn, vec, normal, convex);
            VUSolid::EnumInside surfaceP = fVolumeUSolids->Inside(pointIn + distOut1 * vec);
            if (distOut1 >= UUtils::kInfinity) {
              iInInf++;
              ReportError(&nError, pointIn, vec, distOut1, "TAD1: Distance ToOut is Infinity  for point Inside");
            }
            if (std::fabs(distOut1) < tolerance) {
              iInZero++;
              ReportError(&nError, pointIn, vec, distOut1, "TAD1: Distance ToOut < tolerance  for point Inside");
            }
            iIn++;
            if (surfaceP != vecgeom::EInside::kSurface) {
              iOutNoSurf++;
              ReportError(&nError, pointIn, vec, distOut1, "TAD: Moved to Surface point is not on Surface");
            }
          }

          // Test for consistency for fPoints situated on Surface
          if (distIn < UUtils::kInfinity) {
            iIn++;

            // Surface Test
            VUSolid::EnumInside surfaceP = fVolumeUSolids->Inside(point + distIn * vec);
            if (surfaceP != vecgeom::EInside::kSurface) {
              iInNoSurf++;
              ReportError(&nError, point, vec, distIn, "TAD: Moved to Solid point is not on Surface");
            }
          }
        }
      } else // here for point Inside
      {
        for (int j = 0; j < 1000; j++) {
          iOut++;
          vec = GetRandomDirection();

          distOut                      = fVolumeUSolids->DistanceToOut(point, vec, normal, convex);
          VUSolid::EnumInside surfaceP = fVolumeUSolids->Inside(point + distOut * vec);
          distIn                       = fVolumeUSolids->DistanceToIn(point, vec);
          // iWrongSideIn++;
          if (distOut >= UUtils::kInfinity) {
            iInInf++;
            ReportError(&nError, point, vec, distOut, "TAD2: Distance ToOut is Infinity  for point Inside");
          }
          if (std::fabs(distOut) < tolerance) {
            iInZero++;
            ReportError(&nError, point, vec, distOut, "TAD2: Distance ToOut < tolerance  for point Inside");
          }

          if (surfaceP != vecgeom::EInside::kSurface) {
            iOutNoSurf++;
            ReportError(&nError, point, vec, distOut, "TAD2: Moved to Surface point is not on Surface");
          }
        }
      }
    }
  }
  if (fVerbose) {
    // Surface
    std::cout << "TestAccuracyDistanceToIn::Errors for moved point is not on Surface ::iInNoSurf = " << iInNoSurf
              << ";    iOutNoSurf = " << iOutNoSurf << std::endl;
    std::cout << "TestAccuracyDistanceToIn::Errors SolidUSolid ::From total number of Points  = " << iIn << std::endl;
  }
#ifdef VECGEOM_ROOT
  if (fStat) {
    TCanvas *c7 = new TCanvas("c7", "Accuracy DistancsToIn", 800, 600);
    c7->Update();
    hist10->Draw();
  }
#endif
  std::cout << "% " << std::endl;
  std::cout << "% TestAccuracyDistanceToIn reported = " << CountErrors() << " errors" << std::endl;
  std::cout << "% " << std::endl;

  if (CountErrors()) errCode = 64; // errCode: 0000 0100 0000

  return errCode;
}

int ShapeTester::ShapeSafetyFromInside(int max)
{
  int errCode = 0;
  UVector3 point, dir, pointSphere, norm;
  bool convex;
  int count = 0, count1 = 0;
  int nError = 0;
  ClearErrors();
#ifdef VECGEOM_ROOT
  // visualisation
  TPolyMarker3D *pm3 = 0;
  pm3                = new TPolyMarker3D();
  pm3->SetMarkerSize(0.2);
  pm3->SetMarkerColor(kBlue);
#endif

  if (max > fMaxPointsInside) max = fMaxPointsInside;
  for (int i = 0; i < max; i++) {
    GetVectorUSolids(point, fPoints, i);
    double res = fVolumeUSolids->SafetyFromInside(point);
    for (int j = 0; j < 1000; j++) {
      dir         = GetRandomDirection();
      pointSphere = point + res * dir;
#ifdef VECGEOM_ROOT
      // visualisation
      pm3->SetNextPoint(pointSphere.x(), pointSphere.y(), pointSphere.z());
#endif
      double distOut = fVolumeUSolids->DistanceToOut(point, dir, norm, convex);
      if (distOut < res) {
        count1++;
        ReportError(&nError, pointSphere, dir, distOut, "SSFI: DistanceToOut is underestimated,  less that Safety");
      }
      if (fVolumeUSolids->Inside(pointSphere) == vecgeom::EInside::kOutside) {
        ReportError(&nError, pointSphere, dir, res, "SSFI: Safety is not safe, point on the SafetySphere is Outside");
        double error = fVolumeUSolids->DistanceToIn(pointSphere, -dir);
        if (error > 100 * VUSolid::Tolerance()) {
          count++;
        }
      }
    }
  }
  if (fVerbose) {
    std::cout << "% " << std::endl;
    std::cout << "% ShapeSafetyFromInside ::  number of Points Outside Safety=" << count
              << " number of Points with  distance smaller that safety=" << count1 << std::endl;
    std::cout << "% " << std::endl;
  }
#ifdef VECGEOM_ROOT
  // visualisation
  if (fStat) {
    new TCanvas("shape", "ShapeSafetyFromInside", 1000, 800);
    pm3->Draw();
  }
#endif
  std::cout << "% " << std::endl;
  std::cout << "% TestShapeSafetyFromInside reported = " << CountErrors() << " errors" << std::endl;
  std::cout << "% " << std::endl;

  if (CountErrors()) errCode = 8; // errCode: 0000 0000 1000

  return errCode;
}

int ShapeTester::ShapeSafetyFromOutside(int max)
{
  int errCode = 0;
  UVector3 point, temp, dir, pointSphere, normal;
  double res, error;
  int count = 0, count1 = 0;
  int nError;
  ClearErrors();
#ifdef VECGEOM_ROOT
  // visualisation
  TPolyMarker3D *pm4 = 0;
  pm4                = new TPolyMarker3D();
  pm4->SetMarkerSize(0.2);
  pm4->SetMarkerColor(kBlue);
#endif

  UVector3 minExtent, maxExtent;
  fVolumeUSolids->Extent(minExtent, maxExtent);
  if (max > fMaxPointsOutside) max = fMaxPointsOutside;
  for (int i = 0; i < max; i++) {
    // GetVectorUSolids(point, fPoints, i);
    point = fPoints[i + fOffsetOutside];
    res   = fVolumeUSolids->SafetyFromOutside(point);
    if (res > 0) { // Safety Sphere test
      bool convex;
      int numTrials = 1000;

      for (int j = 0; j < numTrials; j++) {
        dir           = GetRandomDirection();
        double distIn = fVolumeUSolids->DistanceToIn(point, dir);
        if (distIn < res) {
          count1++;
          ReportError(&nError, point, dir, distIn, "SSFO: DistanceToIn is underestimated,  less that Safety");
        }
        pointSphere = point + res * dir;
// std::cout<<"SFO "<<pointSphere<<std::endl;
#ifdef VECGEOM_ROOT
        // visualisation
        pm4->SetNextPoint(pointSphere.x(), pointSphere.y(), pointSphere.z());
#endif
        if (fVolumeUSolids->Inside(pointSphere) == vecgeom::EInside::kInside) {
          ReportError(&nError, pointSphere, dir, res, "SSFO: Safety is not safe, point on the SafetySphere is Inside");
          error = fVolumeUSolids->DistanceToOut(pointSphere, -dir, normal, convex);
          if (error > 100 * VUSolid::Tolerance()) {
            count++;
          }
        }
      }
    }
  }
  if (fVerbose) {
    std::cout << "% " << std::endl;
    std::cout << "% TestShapeSafetyFromOutside::  number of Points Inside Safety Sphere =" << count
              << " number of fPoints with Distance smaller that Safety=" << count1 << std::endl;
    std::cout << "% " << std::endl;
  }
#ifdef VECGEOM_ROOT
  // visualisation
  if (fStat) {
    new TCanvas("shapeTest", "ShapeSafetyFromOutside", 1000, 800);
    pm4->Draw();
  }
#endif
  std::cout << "% " << std::endl;
  std::cout << "% TestShapeSafetyFromOutside reported = " << CountErrors() << " errors" << std::endl;
  std::cout << "% " << std::endl;

  if (CountErrors()) errCode = 16; // errCode: 0000 0001 0000

  return errCode;
}
/////////////////////////////////////////////////////////////////////////////
int ShapeTester::TestXRayProfile()
{
  int errCode = 0;

  std::cout << "% Performing XRayPROFILE number of scans =" << fGNumberOfScans << std::endl;
  std::cout << "% \n" << std::endl;
  if (fGNumberOfScans == 1) {
    errCode += Integration(0, 45, 200, true);
  } // 1-theta,2-phi
  else {
    errCode += XRayProfile(0, fGNumberOfScans, 1000);
  }

  return errCode;
}
/////////////////////////////////////////////////////////////////////////////
int ShapeTester::XRayProfile(double theta, int nphi, int ngrid, bool useeps)
{
  int errCode = 0;

#ifdef VECGEOM_ROOT
  int nError = 0;
  ClearErrors();

  TH1F *hxprofile = new TH1F("xprof", Form("X-ray capacity profile of shape %s for theta=%g degrees",
                                           fVolumeUSolids->GetName().c_str(), theta),
                             nphi, 0, 360);
  if (fStat) {
    new TCanvas("c8", "X-ray capacity profile");
  }
  double dphi   = 360. / nphi;
  double phi    = 0;
  double phi0   = 5;
  double maxerr = 0;

  for (int i = 0; i < nphi; i++) {
    phi = phi0 + (i + 0.5) * dphi;
    // graphic option
    if (nphi == 1) {
      Integration(theta, phi, ngrid, useeps);
    } else {
      Integration(theta, phi, ngrid, useeps, 1, false);
    }
    hxprofile->SetBinContent(i + 1, fGCapacitySampled);
    hxprofile->SetBinError(i + 1, fGCapacityError);
    if (fGCapacityError > maxerr) maxerr = fGCapacityError;
    if ((fGCapacitySampled - fGCapacityAnalytical) > 10 * fGCapacityError) {
      nError++;
      std::cout << "capacity analytical: " << fGCapacityAnalytical << "   sampled: " << fGCapacitySampled << "+/- "
                << fGCapacityError << std::endl;
    }
  }

  double minval = hxprofile->GetBinContent(hxprofile->GetMinimumBin()) - 2 * maxerr;
  double maxval = hxprofile->GetBinContent(hxprofile->GetMaximumBin()) + 2 * maxerr;
  hxprofile->GetXaxis()->SetTitle("phi [deg]");
  hxprofile->GetYaxis()->SetTitle("Sampled capacity");
  hxprofile->GetYaxis()->SetRangeUser(minval, maxval);
  hxprofile->SetMarkerStyle(4);
  hxprofile->SetStats(kFALSE);
  if (fStat) {
    hxprofile->Draw();
  }
  TF1 *lin = new TF1("linear", Form("%f", fGCapacityAnalytical), 0, 360);
  lin->SetLineColor(kRed);
  lin->SetLineStyle(kDotted);
  lin->Draw("SAME");

  std::cout << "% " << std::endl;
  std::cout << "% TestShapeRayProfile reported = " << nError << " errors" << std::endl;
  std::cout << "% " << std::endl;

  if (nError) errCode = 1024; // errCode: 0100 0000 0000
#endif

  return errCode;
}
/////////////////////////////////////////////////////////////////////////////
int ShapeTester::Integration(double theta, double phi, int ngrid, bool useeps, int npercell, bool graphics)
{
  // integrate shape capacity by sampling rays
  int errCode = 0;
  int nError  = 0;
  UVector3 minExtent, maxExtent;
  fVolumeUSolids->Extent(minExtent, maxExtent);
  double maxX   = 2 * std::max(std::fabs(maxExtent.x()), std::fabs(minExtent.x()));
  double maxY   = 2 * std::max(std::fabs(maxExtent.y()), std::fabs(minExtent.y()));
  double maxZ   = 2 * std::max(std::fabs(maxExtent.z()), std::fabs(minExtent.z()));
  double extent = std::sqrt(maxX * maxX + maxY * maxY + maxZ * maxZ);
  double cell   = 2 * extent / ngrid;

  std::vector<UVector3> grid_fPoints; // new double[3*ngrid*ngrid*npercell];
  grid_fPoints.resize(ngrid * ngrid * npercell);
  UVector3 point;
  UVector3 dir;
  double xmin, ymin;
  dir.x() = std::sin(theta * UUtils::kDegToRad) * std::cos(phi * UUtils::kDegToRad);
  dir.y() = std::sin(theta * UUtils::kDegToRad) * std::sin(phi * UUtils::kDegToRad);
  dir.z() = std::cos(theta * UUtils::kDegToRad);

#ifdef VECGEOM_ROOT
  int nfPoints       = ngrid * ngrid * npercell;
  TPolyMarker3D *pmx = 0;
  TH2F *xprof        = 0;
  if (graphics) {
    pmx = new TPolyMarker3D(nfPoints);
    pmx->SetMarkerColor(kRed);
    pmx->SetMarkerStyle(4);
    pmx->SetMarkerSize(0.2);
    xprof = new TH2F(
        "x-ray", Form("X-ray profile from theta=%g phi=%g of shape %s", theta, phi, fVolumeUSolids->GetName().c_str()),
        ngrid, -extent, extent, ngrid, -extent, extent);
  }
#endif

  UTransform3D *matrix = new UTransform3D(0, 0, 0, phi, theta, 0.);
  UVector3 origin      = UVector3(extent * dir.x(), extent * dir.y(), extent * dir.z());

  dir = -dir;

  if ((fVerbose) && (graphics)) printf("=> x-ray direction:( %f, %f, %f)\n", dir.x(), dir.y(), dir.z());
  // loop cells
  int ip = 0;
  for (int i = 0; i < ngrid; i++) {
    for (int j = 0; j < ngrid; j++) {
      xmin = -extent + i * cell;
      ymin = -extent + j * cell;
      if (npercell == 1) {
        point.x()        = xmin + 0.5 * cell;
        point.y()        = ymin + 0.5 * cell;
        point.z()        = 0;
        grid_fPoints[ip] = matrix->GlobalPoint(point) + origin;
#ifdef VECGEOM_ROOT
        if (graphics) pmx->SetNextPoint(grid_fPoints[ip].x(), grid_fPoints[ip].y(), grid_fPoints[ip].z());
#endif
        ip++;
      } else {
        for (int k = 0; k < npercell; k++) {
          point.x()        = xmin + cell * vecgeom::RNG::Instance().uniform();
          point.y()        = ymin + cell * vecgeom::RNG::Instance().uniform();
          point.z()        = 0;
          grid_fPoints[ip] = matrix->GlobalPoint(point) + origin;
#ifdef VECGEOM_ROOT
          if (graphics) pmx->SetNextPoint(grid_fPoints[ip].x(), grid_fPoints[ip].y(), grid_fPoints[ip].z());
#endif
          ip++;
        }
      }
    }
  }
  double sum    = 0;
  double sumerr = 0;
  double dist, lastdist;
  int nhit         = 0;
  int ntransitions = 0;
  bool last        = false;
  for (int i = 0; i < ip; i++) {
    dist = CrossedLength(grid_fPoints[i], dir, useeps);
    sum += dist;

    if (dist > 0) {
      lastdist = dist;
      nhit++;
      if (!last) {
        ntransitions++;
        sumerr += lastdist;
      }
      last  = true;
      point = matrix->LocalPoint(grid_fPoints[i]);
#ifdef VECGEOM_ROOT
      if (graphics) {
        xprof->Fill(point.x(), point.y(), dist);
      }
#endif
    } else {
      if (last) {
        ntransitions++;
        sumerr += lastdist;
      }
      last = false;
    }
  }
  fGCapacitySampled    = sum * cell * cell / npercell;
  fGCapacityError      = sumerr * cell * cell / npercell;
  fGCapacityAnalytical = fVolumeUSolids->Capacity();
  if ((fVerbose) && (graphics)) {
    printf("th=%g phi=%g: analytical: %f    --------   sampled: %f +/- %f\n", theta, phi, fGCapacityAnalytical,
           fGCapacitySampled, fGCapacityError);
    printf("Hit ratio: %f\n", double(nhit) / ip);
    if (nhit > 0) printf("Average crossed length: %f\n", sum / nhit);
  }
  if ((fGCapacitySampled - fGCapacityAnalytical) > 10 * fGCapacityError) nError++;

#ifdef VECGEOM_ROOT
  if (graphics) {
    if (fStat) {
      new TCanvas("c11", "X-ray scan");
      xprof->DrawCopy("LEGO1");
    }
  }
#endif

  if (nError) errCode = 512; // errCode: 0010 0000 0000

  return errCode;
}
//////////////////////////////////////////////////////////////////////////////
double ShapeTester::CrossedLength(const UVector3 &point, const UVector3 &dir, bool useeps)
{
  // Return crossed length of the shape for the given ray, taking into account possible multiple crossings
  double eps = 0;

  if (useeps) eps = 1.E-9;
  double len      = 0;
  double dist     = fVolumeUSolids->DistanceToIn(point, dir);
  if (dist > 1E10) return len;
  // Propagate from starting point with the found distance (on the numerical boundary)
  UVector3 pt(point), norm;
  bool convex;

  while (dist < 1E10) {
    pt = pt + (dist + eps) * dir; // ray entering
    // Compute distance from inside
    dist = fVolumeUSolids->DistanceToOut(pt, dir, norm, convex);
    len += dist;
    pt   = pt + (dist + eps) * dir; // ray exiting
    dist = fVolumeUSolids->DistanceToIn(pt, dir);
  }
  return len;
}
////////////////////////////////////////////////////////////////////////////
void ShapeTester::FlushSS(stringstream &ss)
{
  string s = ss.str();
  cout << s;
  *fLog << s;
  ss.str("");
}

void ShapeTester::Flush(const string &s)
{
  cout << s;
  *fLog << s;
}

void ShapeTester::CreatePointsAndDirectionsSurface()
{
  UVector3 norm, point;
  for (int i = 0; i < fMaxPointsSurface; i++) {

    UVector3 pointU;
#if 0
    int retry = 100;
    do
    { bool surfaceExist=true;
      if(surfaceExist) {pointU = fVolumeUSolids->GetPointOnSurface(); }
      else {
        UVector3 dir = GetRandomDirection(), norm;
        bool convex;
        double random=UUtils::Random();
        int index = (int)fMaxPointsInside*random;
        double dist = fVolumeUSolids->DistanceToOut(fPoints[index],dir,norm,convex);
        pointU = fPoints[index]+dir*dist ;

      }
      if (retry-- == 0) break;
    }
    while (fVolumeUSolids->Inside(pointU) != vecgeom::EInside::kSurface);
#endif
    int retry = 100;
    do {
      pointU                          = fVolumeUSolids->GetPointOnSurface();
      UVector3 vec                    = GetRandomDirection();
      fDirections[i + fOffsetSurface] = vec;
      point.Set(pointU.x(), pointU.y(), pointU.z());
      fPoints[i + fOffsetSurface] = point;
      if (retry-- == 0) {
        std::cout << "Couldn't find point on surface in 100 trials, so skipping this point." << std::endl;
        break;
      }
    } while (fVolumeUSolids->Inside(pointU) != vecgeom::EInside::kSurface);
  }
}
void ShapeTester::CreatePointsAndDirectionsEdge()
{
  UVector3 norm, point;

  for (int i = 0; i < fMaxPointsEdge; i++) {
    UVector3 pointU;
    int retry = 100;
    do {
      fVolumeUSolids->SamplePointsOnEdge(1, &pointU);
      if (retry-- == 0) break;
    } while (fVolumeUSolids->Inside(pointU) != vecgeom::EInside::kSurface);
    UVector3 vec   = GetRandomDirection();
    fDirections[i] = vec;

    point.Set(pointU.x(), pointU.y(), pointU.z());
    fPoints[i + fOffsetEdge] = point;
  }
}

void ShapeTester::CreatePointsAndDirectionsOutside()
{

  UVector3 minExtent, maxExtent;
  fVolumeUSolids->Extent(minExtent, maxExtent);
  double maxX = std::max(std::fabs(maxExtent.x()), std::fabs(minExtent.x()));
  double maxY = std::max(std::fabs(maxExtent.y()), std::fabs(minExtent.y()));
  double maxZ = std::max(std::fabs(maxExtent.z()), std::fabs(minExtent.z()));
  double rOut = std::sqrt(maxX * maxX + maxY * maxY + maxZ * maxZ);

  for (int i = 0; i < fMaxPointsOutside; i++) {

    UVector3 vec, point;
    do {
      point.x() = -1 + 2 * UUtils::Random();
      point.y() = -1 + 2 * UUtils::Random();
      point.z() = -1 + 2 * UUtils::Random();
      point *= rOut * fOutsideMaxRadiusMultiple;
    } while (fVolumeUSolids->Inside(point) != vecgeom::EInside::kOutside);

    double random = UUtils::Random();
    if (random <= fOutsideRandomDirectionPercent / 100.) {
      vec = GetRandomDirection();
    } else {
      UVector3 pointSurface = fVolumeUSolids->GetPointOnSurface();
      vec                   = pointSurface - point;
      vec.Normalize();
    }

    fPoints[i + fOffsetOutside]     = point;
    fDirections[i + fOffsetOutside] = vec;
  }
}

// DONE: inside fPoints generation uses random fPoints inside bounding box
void ShapeTester::CreatePointsAndDirectionsInside()
{
  UVector3 minExtent, maxExtent;
  fVolumeUSolids->Extent(minExtent, maxExtent);
  int i = 0;
  while (i < fMaxPointsInside) {
    double x                              = RandomRange(minExtent.x(), maxExtent.x());
    double y                              = RandomRange(minExtent.y(), maxExtent.y());
    if (minExtent.y() == maxExtent.y()) y = RandomRange(-1000, +1000);
    double z                              = RandomRange(minExtent.z(), maxExtent.z());
    UVector3 point0(x, y, z);
    if (fVolumeUSolids->Inside(point0) == vecgeom::EInside::kInside) {
      UVector3 point(x, y, z);
      UVector3 vec                   = GetRandomDirection();
      fPoints[i + fOffsetInside]     = point;
      fDirections[i + fOffsetInside] = vec;
      i++;
    }
  }
}

void ShapeTester::CreatePointsAndDirections()
{
  if (fMethod != "XRayProfile") {
    fMaxPointsInside  = (int)(fMaxPoints * (fInsidePercent / 100));
    fMaxPointsOutside = (int)(fMaxPoints * (fOutsidePercent / 100));
    fMaxPointsEdge    = (int)(fMaxPoints * (fEdgePercent / 100));
    fMaxPointsSurface = fMaxPoints - fMaxPointsInside - fMaxPointsOutside - fMaxPointsEdge;

    fOffsetInside  = 0;
    fOffsetSurface = fMaxPointsInside;
    fOffsetEdge    = fOffsetSurface + fMaxPointsSurface;
    fOffsetOutside = fOffsetEdge + fMaxPointsEdge;

    fPoints.resize(fMaxPoints);
    fDirections.resize(fMaxPoints);
    fResultDoubleDifference.resize(fMaxPoints);
    fResultBoolUSolids.resize(fMaxPoints);
    fResultDoubleUSolids.resize(fMaxPoints);

    fResultVectorDifference.resize(fMaxPoints);
    fResultVectorUSolids.resize(fMaxPoints);

    CreatePointsAndDirectionsOutside();
    CreatePointsAndDirectionsInside();
    CreatePointsAndDirectionsSurface();
  }
}

#include <sys/types.h> // For stat().
#include <sys/stat.h>  // For stat().

int directoryExists(string s)
{
  {
    struct stat status;
    stat(s.c_str(), &status);
    return (status.st_mode & S_IFDIR);
  }
  return false;
}

void ShapeTester::PrintCoordinates(stringstream &ss, const UVector3 &vec, const string &delimiter, int precision)
{
  ss.precision(precision);
  ss << vec.x() << delimiter << vec.y() << delimiter << vec.z();
}

string ShapeTester::PrintCoordinates(const UVector3 &vec, const string &delimiter, int precision)
{
  static stringstream ss;
  PrintCoordinates(ss, vec, delimiter, precision);
  string res(ss.str());
  ss.str("");
  return res;
}

string ShapeTester::PrintCoordinates(const UVector3 &vec, const char *delimiter, int precision)
{
  string d(delimiter);
  return PrintCoordinates(vec, d, precision);
}

void ShapeTester::PrintCoordinates(stringstream &ss, const UVector3 &vec, const char *delimiter, int precision)
{
  string d(delimiter);
  return PrintCoordinates(ss, vec, d, precision);
}

// NEW: output values precision setprecision (16)
// NEW: for each fMethod, one file

// NEW: print also different point coordinates

void ShapeTester::VectorToDouble(const vector<UVector3> &vectorUVector, vector<double> &vectorDouble)
{
  UVector3 vec;

  int size = vectorUVector.size();
  for (int i = 0; i < size; i++) {
    vec                = vectorUVector[i];
    double mag         = vec.Mag();
    if (mag > 1.1) mag = 1;
    vectorDouble[i]    = mag;
  }
}

void ShapeTester::BoolToDouble(const std::vector<bool> &vectorBool, std::vector<double> &vectorDouble)
{
  int size = vectorBool.size();
  for (int i        = 0; i < size; i++)
    vectorDouble[i] = (double)vectorBool[i];
}

int ShapeTester::SaveResultsToFile(const string &fMethod1)
{
  string name = fVolumeUSolids->GetName();
  string fFilename1(fFolder + name + "_" + fMethod1 + ".dat");
  std::cout << "Saving all results to " << fFilename1 << std::endl;
  ofstream file(fFilename1.c_str());
  bool saveVectors = (fMethod1 == "Normal");
  int prec         = 16;
  if (file.is_open()) {
    file.precision(prec);
    file << fVolumeString << "\n";
    string spacer("\t");
    for (int i = 0; i < fMaxPoints; i++) {

      file << PrintCoordinates(fPoints[i], spacer, prec) << spacer << PrintCoordinates(fDirections[i], spacer, prec)
           << spacer;
      if (saveVectors)
        file << PrintCoordinates(fResultVectorUSolids[i], spacer, prec) << "\n";
      else
        file << fResultDoubleUSolids[i] << "\n";
    }
    return 0;
  }
  std::cout << "Unable to create file " << fFilename1 << std::endl;
  return 1;
}

int ShapeTester::TestMethod(int (ShapeTester::*funcPtr)())
{
  int errCode = 0;

  std::cout << "========================================================= " << std::endl;

  if (fMethod != "XRayProfile") {
    std::cout << "% Creating " << fMaxPoints << " Points and Directions for Method =" << fMethod << std::endl;

    CreatePointsAndDirections();
    cout.precision(20);
    std::cout << "% Statistics: Points=" << fMaxPoints << ",\n";

    std::cout << "%             ";
    std::cout << "surface=" << fMaxPointsSurface << ", inside=" << fMaxPointsInside << ", outside=" << fMaxPointsOutside
              << "\n";
  }
  std::cout << "%     " << std::endl;

  errCode += (*this.*funcPtr)();
  std::cout << "========================================================= " << std::endl;

  return errCode;
}

// will run all tests. in this case, one file stream will be used
int ShapeTester::TestMethodAll()
{
  int errCode = 0;
  if (fTestBoundaryErrors) {
    fMethod = "BoundaryPrecision";
    TestBoundaryPrecision(0);
  }
  fMethod = "Consistency";
  errCode += TestMethod(&ShapeTester::TestConsistencySolids);
  if (fDefinedNormal) TestMethod(&ShapeTester::TestNormalSolids);
  fMethod = "SafetyFromInside";
  errCode += TestMethod(&ShapeTester::TestSafetyFromInsideSolids);
  fMethod = "SafetyFromOutside";
  errCode += TestMethod(&ShapeTester::TestSafetyFromOutsideSolids);
  fMethod = "DistanceToIn";
  errCode += TestMethod(&ShapeTester::TestDistanceToInSolids);
  fMethod = "DistanceToOut";
  errCode += TestMethod(&ShapeTester::TestDistanceToOutSolids);
  fMethod = "XRayProfile";
  errCode += TestMethod(&ShapeTester::TestXRayProfile);

  fMethod = "all";

  return errCode;
}

void ShapeTester::SetFolder(const string &newFolder)
{
  cout << "Checking for existance of " << newFolder << endl;

  if (!directoryExists(newFolder)) {
    string command;
#ifdef WIN32
    _mkdir(newFolder.c_str());
#else
    std::cout << "try to create dir for " << std::endl;
    mkdir(newFolder.c_str(), 0777);
#endif
    if (!directoryExists(newFolder)) {
      cout << "Directory " + newFolder + " does not exist, it must be created first\n";
      exit(1);
    }
  }
  fFolder = newFolder + "/";
}

void ShapeTester::Run(VUSolid *testVolume, const char *type)
{
  if (strcmp(type, "stat") == 0) {
    this->setStat(true);
  }
  if (strcmp(type, "debug") == 0) {
    this->setDebug(true);
  }

  this->Run(testVolume);
#ifdef VECGEOM_ROOT
  if (fStat) fVisualizer.GetTApp()->Run();
#endif
}

int ShapeTester::Run(VUSolid *testVolume)
{
  // debug mode doesn't work with USolids shape
  if (fDebug) {
    const vecgeom::VPlacedVolume *vgvol = dynamic_cast<vecgeom::VPlacedVolume *>(fVolumeUSolids);
    if (!vgvol) {
      std::cout << "\n\n==========================================================\n";
      std::cout << "***** ShapeTester WARNING: debug mode does not work with a USolids shape!!\n";
      std::cout << "      Try to use shapeDebug binary to visualize this shape.\n";
      std::cout << "*****  Resetting fDebug to false...\n";
      std::cout << "==========================================================\n\n\n";
      this->setDebug(false);
    }
  }

  // Running Convention first before running any ShapeTester tests
  RunConventionChecker(testVolume);
  fNumDisp    = 5;
  int errCode = 0;
  stringstream ss;

  int (ShapeTester::*funcPtr)() = NULL;

  fVolumeUSolids = testVolume;
  std::ofstream fLogger("/Log/box");
  fLog = &fLogger;

  SetFolder("Log");

#ifdef VECGEOM_ROOT
  // serialize the ROOT solid for later debugging
  // check if this is a VecGeom class
  if (VPlacedVolume *p = dynamic_cast<VPlacedVolume *>(fVolumeUSolids)) {
    // propagate label to logical volume (to be addressable in CompareDistance tool)
    const_cast<LogicalVolume *>(p->GetLogicalVolume())->SetLabel(p->GetName());
    RootGeoManager::Instance().ExportToROOTGeometry(p, "Log/ShapeTesterGeom.root");
  }
#endif

  if (fMethod == "") fMethod = "all";
  string name                = testVolume->GetName();
  std::cout << "\n\n";
  std::cout << "===============================================================================\n";
  std::cout << "Invoking test for Method " << fMethod << " on " << name << " ..."
            << "\nFolder is " << fFolder << std::endl;
  std::cout << "===============================================================================\n";
  std::cout << "\n";

  if (fMethod == "Consistency") funcPtr       = &ShapeTester::TestConsistencySolids;
  if (fMethod == "Normal") funcPtr            = &ShapeTester::TestNormalSolids;
  if (fMethod == "SafetyFromInside") funcPtr  = &ShapeTester::TestSafetyFromInsideSolids;
  if (fMethod == "SafetyFromOutside") funcPtr = &ShapeTester::TestSafetyFromOutsideSolids;
  if (fMethod == "DistanceToIn") funcPtr      = &ShapeTester::TestDistanceToInSolids;
  if (fMethod == "DistanceToOut") funcPtr     = &ShapeTester::TestDistanceToOutSolids;
  if (fMethod == "XRayProfile") funcPtr       = &ShapeTester::TestXRayProfile;

  if (fMethod == "all")
    errCode += TestMethodAll();
  else if (funcPtr)
    errCode += TestMethod(funcPtr);
  else
    std::cout << "Method " << fMethod << " is not supported" << std::endl;

  ClearErrors();
  fMethod = "all";

  errCode += fScore;
  if (errCode) {
    std::cout << "--------------------------------------------------------------------------------------" << std::endl;
    std::cout << "--- Either Shape Conventions not followed or some of the ShapeTester's test failed ---" << std::endl;
    std::cout << "--------------------------------------------------------------------------------------" << std::endl;
    std::cout << "----------------- Generated Overall Error Code : " << errCode << " -------------------" << std::endl;
    std::cout << "--------------------------------------------------------------------------------------" << std::endl;
  }
  return errCode;
}
int ShapeTester::RunMethod(VUSolid *testVolume, std::string fMethod1)
{
  int errCode = 0;
  stringstream ss;

  int (ShapeTester::*funcPtr)() = NULL;

  fVolumeUSolids = testVolume;
  std::ofstream fLogger("/Log/box");
  fLog = &fLogger;

  SetFolder("Log");

  fMethod = fMethod1;

  if (fMethod == "") fMethod = "all";
  string name                = testVolume->GetName();

  std::cout << "\n\n";
  std::cout << "===============================================================================\n";
  std::cout << "Invoking test for Method " << fMethod << " on " << name << " ..."
            << "\nFolder is " << fFolder << std::endl;
  std::cout << "===============================================================================\n";
  std::cout << "\n";

  if (fMethod == "Consistency") funcPtr       = &ShapeTester::TestConsistencySolids;
  if (fMethod == "Normal") funcPtr            = &ShapeTester::TestNormalSolids;
  if (fMethod == "SafetyFromInside") funcPtr  = &ShapeTester::TestSafetyFromInsideSolids;
  if (fMethod == "SafetyFromOutside") funcPtr = &ShapeTester::TestSafetyFromOutsideSolids;
  if (fMethod == "DistanceToIn") funcPtr      = &ShapeTester::TestDistanceToInSolids;
  if (fMethod == "DistanceToOut") funcPtr     = &ShapeTester::TestDistanceToOutSolids;

  if (fMethod == "XRayProfile") funcPtr = &ShapeTester::TestXRayProfile;
  if (fMethod == "all")
    errCode += TestMethodAll();
  else if (funcPtr)
    errCode += TestMethod(funcPtr);
  else
    std::cout << "Method " << fMethod << " is not supported" << std::endl;

  ClearErrors();
  fMethod = "all";

  return errCode;
}
//
// ReportError
//
// Report the specified error fMessage, but only if it has not been reported a zillion
// times already.
//
void ShapeTester::ReportError(int *nError, UVector3 &p, UVector3 &v, double distance,
                              std::string comment) //, std::ostream &fLogger )
{

  ShapeTesterErrorList *last = 0, *errors = fErrorList;
  while (errors) {

    if (errors->fMessage == comment) {
      if (++errors->fNUsed > fNumDisp) return;
      break;
    }
    last   = errors;
    errors = errors->fNext;
  }

  if (errors == 0) {
    //
    // New error: add it the end of our list
    //
    errors           = new ShapeTesterErrorList;
    errors->fMessage = comment;
    errors->fNUsed   = 1;
    errors->fNext    = 0;
    if (fErrorList)
      last->fNext = errors;
    else
      fErrorList = errors;
  }

  //
  // Output the fMessage
  //

  std::cout << "% " << comment;
  if (errors->fNUsed == fNumDisp) std::cout << " (any further such errors suppressed)";
  std::cout << " Distance = " << distance;
  std::cout << std::endl;

  std::cout << std::setprecision(25) << ++(*nError) << " : [point] : [direction] ::  " << p << " : " << v << std::endl;
#ifdef VECGEOM_ROOT
  if (fDebug) {
    fVisualizer.AddVolume(*dynamic_cast<vecgeom::VPlacedVolume *>(fVolumeUSolids));
    fVisualizer.AddPoint(p);
    fVisualizer.AddLine(p, (p + 10000. * v));
    fVisualizer.Show();
  }
#endif
  //
  // if debugging mode we have to exit now
  //
  if (fIfException) {
    std::ostringstream text;
    text << "Aborting due to Debugging mode in solid: " << fVolumeUSolids->GetName();
    UUtils::Exception("ShapeTester", "Debugging mode", UFatalErrorInArguments, 1, text.str().c_str());
  }
}
//
// ClearErrors
// Reset list of errors (and clear memory)
//
void ShapeTester::ClearErrors()
{
  ShapeTesterErrorList *here, *sNext;

  here = fErrorList;
  while (here) {
    sNext = here->fNext;
    delete here;
    here = sNext;
  }
  fErrorList = 0;
}
//
// CountErrors
//
int ShapeTester::CountErrors() const
{
  ShapeTesterErrorList *here;
  int answer = 0;

  here = fErrorList;
  while (here) {
    answer += here->fNUsed;
    here = here->fNext;
  }

  return answer;
}

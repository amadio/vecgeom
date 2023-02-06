#include "Visualizer.h"

#include "VecGeom/base/AOS3D.h"
#include "VecGeom/base/SOA3D.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/volumes/PlacedVolume.h"

#include "TApplication.h"
#include "TAxis3D.h"
#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoShape.h"
#include "TPolyLine3D.h"
#include "TPolyMarker3D.h"
#include "TVirtualPad.h"
#include "TView.h"

#include <iostream>

namespace vecgeom {

Visualizer::Visualizer() : fVerbosity(0), fVolumes(), fMarkers(), fLines(), fApp(0), fGeoManager(0)
{
  fApp = new TApplication("VecGeom Visualizer", NULL, NULL);
}

Visualizer::~Visualizer()
{
  delete fApp;
}

void Visualizer::AddVolume(VPlacedVolume const &volume)
{
  if (fGeoManager) delete fGeoManager;
  fGeoManager                = new TGeoManager("visualizer", "");
  TGeoShape const *rootShape = volume.ConvertToRoot();
  fVolumes.push_back(std::make_tuple(std::shared_ptr<const TGeoShape>(rootShape),
                                     std::unique_ptr<TGeoMatrix>(new TGeoIdentity()),
                                     std::unique_ptr<TGeoVolume>(new TGeoVolume("", rootShape, nullptr))));
  if (fVerbosity > 0) {
    std::cout << "Added volume " << volume << " to Visualizer.\n";
  }
}

void Visualizer::AddVolume(VPlacedVolume const &volume, Transformation3D const &transformation)
{
  // Cannot store const pointer because Draw() is not const
  if (fGeoManager) delete fGeoManager;
  fGeoManager                = new TGeoManager("visualizer", "");
  TGeoShape const *rootShape = volume.ConvertToRoot();
  fVolumes.push_back(std::make_tuple(std::shared_ptr<const TGeoShape>(rootShape),
                                     std::unique_ptr<TGeoMatrix>(Transformation3D::ConvertToTGeoMatrix(transformation)),
                                     std::unique_ptr<TGeoVolume>(new TGeoVolume("", rootShape, nullptr))));
  if (fVerbosity > 0) {
    std::cout << "Added volume " << volume << " to Visualizer.\n";
  }
}

void Visualizer::AddVolume(std::shared_ptr<const TGeoShape> volume)
{
  fVolumes.push_back(std::make_tuple(volume, std::unique_ptr<TGeoMatrix>(new TGeoIdentity),
                                     std::unique_ptr<TGeoVolume>(new TGeoVolume("", volume.get(), nullptr))));
  if (fVerbosity > 0) {
    std::cout << "Added ROOT volume to Visualizer.\n";
  }
}

void Visualizer::AddVolume(std::shared_ptr<const TGeoShape> volume, Transformation3D const &transformation)
{
  fVolumes.push_back(std::make_tuple(volume, std::unique_ptr<TGeoMatrix>(Transformation3D::ConvertToTGeoMatrix(transformation)),
                                     std::unique_ptr<TGeoVolume>(new TGeoVolume("", volume.get(), nullptr))));
  if (fVerbosity > 0) {
    std::cout << "Added ROOT volume to Visualizer.\n";
  }
}

void Visualizer::AddPoints(AOS3D<Precision> const &points, int color, int style, int size)
{
  AddPointsTemplate(points, color, style, size);
}

void Visualizer::AddPoints(SOA3D<Precision> const &points, int color, int style, int size)
{
  AddPointsTemplate(points, color, style, size);
}

void Visualizer::AddPoint(Vector3D<Precision> const &point, int color, int style, int size)
{
  SOA3D<Precision> c(0);
  c.push_back(point);
  AddPoints(c, color, style, size);
}

void Visualizer::AddPoints(TPolyMarker3D const &marker)
{
  fMarkers.emplace_back(new TPolyMarker3D(marker));
  if (fVerbosity > 0) {
    std::cout << "Added " << marker.GetN() << " points to Visualizer.\n";
  }
}

void Visualizer::AddLine(Vector3D<Precision> const &p0, Vector3D<Precision> const &p1)
{

  TPolyLine3D *line = new TPolyLine3D(2);
  line->SetPoint(0, p0[0], p0[1], p0[2]);
  line->SetPoint(1, p1[0], p1[1], p1[2]);
  line->SetLineColor(kBlue);
  fLines.emplace_back(line);
  if (fVerbosity > 0) {
    std::cout << "Added line " << p0 << "--" << p1 << " to Visualizer.\n";
  }
}

void Visualizer::AddLine(TPolyLine3D const &line)
{
  fLines.emplace_back(new TPolyLine3D(line));
  auto GetPoint = [&line](int index) {
    float *pointArray = line.GetP();
    int offset        = 3 * index;
    return Vector3D<Precision>(pointArray[offset], pointArray[offset + 1], pointArray[offset + 2]);
    ;
  };
  if (line.GetN() == 2) {
    if (fVerbosity > 0) {
      std::cout << "Added line " << GetPoint(0) << "--" << GetPoint(1) << " to Visualizer.\n";
    }
  } else {
    if (fVerbosity > 0) {
      std::cout << "Added line with " << line.GetN() << " points to Visualizer.\n";
    }
  }
}

void Visualizer::Show() const
{
  TAxis3D axes;
  TGeoVolume *top = fGeoManager->MakeBox("Top", NULL, kInfLength, kInfLength, kInfLength);
  fGeoManager->SetTopVolume(top);
  for (auto &volume : fVolumes) {
    top->AddNode(std::get<2>(volume).get(), top->GetNdaughters(), std::get<1>(volume).get());
  }
  top->Draw();
  for (auto &marker : fMarkers) {
    marker->Draw();
  }
  for (auto &line : fLines) {
    line->Draw();
  }
  gPad->GetView()->ShowAxis();
  axes.Draw();
  fApp->Run();
}

void Visualizer::Clear()
{
  fVolumes.clear();
  fMarkers.clear();
  fLines.clear();
  if (fVerbosity > 0) {
    std::cout << "Cleared Visualizer content.\n";
  }
}

void Visualizer::SetVerbosity(int level)
{
  fVerbosity = level;
}

template <class ContainerType>
void Visualizer::AddPointsTemplate(ContainerType const &points, int color, int style, int markersize)
{
  const int size        = points.size();
  TPolyMarker3D *marker = new TPolyMarker3D(size);
  marker->SetMarkerColor(color);
  marker->SetMarkerSize(markersize);
  marker->SetMarkerStyle(style);
  for (int i = 0; i < size; ++i) {
    marker->SetNextPoint(points.x(i), points.y(i), points.z(i));
  }
  fMarkers.emplace_back(marker);
  if (fVerbosity > 0) {
    std::cout << "Added " << size << " points to Visualizer.\n";
  }
}

} // End namespace vecgeom

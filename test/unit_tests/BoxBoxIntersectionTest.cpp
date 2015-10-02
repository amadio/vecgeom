#include "volumes/utilities/VolumeUtilities.h"
#include "base/Global.h"
#include "base/Vector3D.h"
#include "base/Transformation3D.h"
#include "volumes/Box.h"
#include "utilities/Visualizer.h"


using namespace vecgeom;

int main(){
Transformation3D t0(7.0,0,0,45,0,45);
std::cout<<t0<<std::endl;
std::cout<<t0.InverseTransformDirection(Vector3D<Precision>(1,0,0))<<std::endl;



SimpleBox box("VisualizerBox",2,3,4);
const Transformation3D *m = box.GetTransformation();
std::cout<<"Box Transformation : "<<m->InverseTransformDirection(Vector3D<Precision>(1,0,0))<<std::endl;

Precision dx=2., dy=5., dz=6.;
//Creating a box from unplaced then placed it at a known location
//UnplacedBox worldUnplaced = UnplacedBox(dx*4, dy*4, dz*4);
UnplacedBox boxUnplaced = UnplacedBox(dx, dy, dz);
SimpleBox boxSimple("SimpleBox",dx,dy,dz);
//LogicalVolume world = LogicalVolume("world", &worldUnplaced);
LogicalVolume box2 = LogicalVolume("box", &boxUnplaced);

//Transformation3D placement(0.1, 0, 0);
//world.PlaceDaughter("box", &box2, &placement);

//VPlacedVolume *worldPlaced = world.Place();
//const VPlacedVolume *boxPlaced = box2.Place(&t0);
VPlacedVolume *boxPlaced = box2.Place(&t0);
m = boxPlaced->GetTransformation();
//m = box2.GetTransformation();
std::cout<<"Placed Box Transformation : "<<m->InverseTransformDirection(Vector3D<Precision>(1,0,0))<<std::endl;

Vector3D<Precision> aMin(0,0,0),aMax(0,0,0);
boxPlaced->Extent(aMin,aMax);
//std::cout<<"LowerCorner : "<<m->InverseTransform(aMin)<<"  :: UpperCorner : "<<m->InverseTransform(aMax)<<std::endl;
Vector3D<Precision> centreTransformedBox = m->InverseTransform(Vector3D<Precision>(0,0,0));
Vector3D<Precision> newXAxis = m->InverseTransformDirection(Vector3D<Precision>(1,0,0));
Vector3D<Precision> newYAxis = m->InverseTransformDirection(Vector3D<Precision>(0,1,0));
Vector3D<Precision> newZAxis = m->InverseTransformDirection(Vector3D<Precision>(0,0,1));
Vector3D<Precision> pOnX = centreTransformedBox+(2.*newXAxis);
Vector3D<Precision> pOnY = centreTransformedBox+(2.*newYAxis);
Vector3D<Precision> pOnZ = centreTransformedBox+(2.*newZAxis);
std::cout<<"Centre of transformed box  : "<<centreTransformedBox<<std::endl;

//Box Intersection test with Real case
std::cout<<std::endl<<" ---- Trying intersection Detection with Real Boxes ----"<<std::endl;
Transformation3D transform1(0,0,0,0,0,0);
std::cout<<"Intersection Result from new function : "<<volumeUtilities::IntersectionExist(
		aMin,aMax,aMin,aMax,transform1,m,true)<<std::endl;
//-------------------------------------


//Visualization Stuff
Visualizer visualizer;
visualizer.AddVolume(boxSimple);
visualizer.AddVolume(boxSimple,t0);
visualizer.AddPoint(centreTransformedBox);
visualizer.AddPoint(m->InverseTransform(aMin));
visualizer.AddPoint(m->InverseTransform(aMax));
visualizer.AddPoint(aMin);
visualizer.AddPoint(aMax);
visualizer.AddLine(centreTransformedBox,pOnX);
visualizer.AddLine(centreTransformedBox,pOnY);
visualizer.AddLine(centreTransformedBox,pOnZ);

visualizer.Show();
return 0;


}

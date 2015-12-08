
#undef NDEBUG
#define GENTRABDEBUG
#include "base/Vector3D.h"
#include "volumes/LogicalVolume.h"
#include "volumes/UnplacedGenTrap.h"
#include "backend/Backend.h"
#include "volumes/kernel/GenTrapImplementation.h"
#include "base/Transformation3D.h"
#include <vector>
#include <iostream>
#include <Vc/Vc>

using namespace vecgeom;
// testing a couple of things in box
//#ifdef NDEBUG

//#endif


void my_assert( bool cond, std::string const & str )
{
    if( !cond )
    std::cerr << "assert failed " << str << "\n";
}

void my_assert( bool cond, double expected, double obtained )
{
    if( !cond )
    std::cerr << "assert failed (exp " << expected << " obtained " << obtained << " )\n";
}
// setting up a generic trap in the GeoManager
void createGenTrap()
{
  std::vector<Vector3D<Precision> > vertexlist;

  // no twist
  vertexlist.push_back( Vector3D<Precision>(-3,-3, 0 ) );
  vertexlist.push_back( Vector3D<Precision>(-3, 3, 0 ) );
  vertexlist.push_back( Vector3D<Precision>( 3, 3, 0 ) );
  vertexlist.push_back( Vector3D<Precision>( 3,-3, 0 ) );
  vertexlist.push_back( Vector3D<Precision>(-3,-3, 0 ) );
  vertexlist.push_back( Vector3D<Precision>(-3, 3, 0 ) );
  vertexlist.push_back( Vector3D<Precision>( 3, 3, 0 ) );
  vertexlist.push_back( Vector3D<Precision>( 3,-3, 0 ) );

  UnplacedGenTrap trapUnplaced1( &vertexlist[0], 10 );
  my_assert( trapUnplaced1.ComputeIsTwisted() == false, "twisted" );

  vertexlist.clear();
  vertexlist.push_back( Vector3D<Precision>(  -3, -3, 0 ) );
  vertexlist.push_back( Vector3D<Precision>(  -3,  3, 0 ) );
  vertexlist.push_back( Vector3D<Precision>(   3,  3, 0 ) );
  vertexlist.push_back( Vector3D<Precision>(   3, -3, 0 ) );
  vertexlist.push_back( Vector3D<Precision>(-0.5, -2, 0 ) );
  vertexlist.push_back( Vector3D<Precision>(  -2,  2, 0 ) );
  vertexlist.push_back( Vector3D<Precision>(   2,  2, 0 ) );
  vertexlist.push_back( Vector3D<Precision>(   2, -2, 0 ) );

  UnplacedGenTrap trapUnplaced2( &vertexlist[0], 10 );
  trapUnplaced2.Print(std::cout);
  my_assert( trapUnplaced2.ComputeIsTwisted() == true, "twisted" );

  // try to compile a kernel directly
  Vector3D<Precision> localPoint(0.,0.,0.);
  bool completelyInside;
  bool completelyOutside;
  vecgeom::GenTrapImplementation<0,0>::GenericKernelForContainsAndInside<kScalar, false>(
      trapUnplaced2,
      localPoint,
      completelyInside,
      completelyOutside
  );
  my_assert( completelyInside == true, "inside1" );
  my_assert( completelyOutside == false, "outside1" );

  Vector3D<Precision> localPoint2(-10.,0.,0.);
    vecgeom::GenTrapImplementation<0,0>::GenericKernelForContainsAndInside<kScalar, false>(
        trapUnplaced2,
        localPoint2,
        completelyInside,
        completelyOutside
    );
    my_assert( completelyInside == false, "inside1" );
    my_assert( completelyOutside == true, "outside1" );

    typedef Vector3D<Precision> Vec_t;
    typedef Vector3D<Vc::double_v> VecVc_t;

    Precision d;
    Precision step=100.;
//    vecgeom::GenTrapImplementation<vecgeom::translation::kIdentity,
//        vecgeom::rotation::kIdentity>::DistanceToIn<kScalar>(trapUnplaced2, Transformation3D(),
//            Vec_t(0,0,11), Vec_t(0,0,-1), step, d);
//    my_assert(d==1., 1, d);
//
//    typedef Vector3D<Precision> Vec_t;
//    vecgeom::GenTrapImplementation<vecgeom::translation::kIdentity,
//           vecgeom::rotation::kIdentity>::DistanceToIn<kScalar>(trapUnplaced2, Transformation3D(),
//               Vec_t(-2.9,2.9,11), Vec_t(0,0,-1), step, d);
//    assert(d>1);
//
//    typedef Vector3D<Precision> Vec_t;
//    vecgeom::GenTrapImplementation<vecgeom::translation::kIdentity,
//    vecgeom::rotation::kIdentity>::DistanceToIn<kScalar>(trapUnplaced2, Transformation3D(),
//                   Vec_t(-2.9,2.9,-11), Vec_t(0,0,1), step, d);
//    my_assert(d==1.,1.,d);

    // further complicated test for distancetoin
    vertexlist.clear();
    vertexlist.push_back( Vector3D<Precision>(-3,-2.5, 0 ) );
    vertexlist.push_back( Vector3D<Precision>(-3, 3, 0 ) );
    vertexlist.push_back( Vector3D<Precision>( 3, 3, 0 ) );
    vertexlist.push_back( Vector3D<Precision>( 3,-3, 0 ) );
    vertexlist.push_back( Vector3D<Precision>(-3,-3, 0 ) );
    vertexlist.push_back( Vector3D<Precision>(-3, 3, 0 ) );
    vertexlist.push_back( Vector3D<Precision>( 3, 2, 0 ) );
    vertexlist.push_back( Vector3D<Precision>( 3,-3, 0 ) );

    UnplacedGenTrap trapUnplaced3( &vertexlist[0], 10 );
    vecgeom::GenTrapImplementation<vecgeom::translation::kGeneric,
    vecgeom::rotation::kIdentity>::DistanceToIn<kScalar>(trapUnplaced3, Transformation3D(5.,5.,5.),
                           Vec_t(-9.441650,0.902802, 10.916641), Vec_t(0.954990,0.247627,-0.163324), step, d);
    assert(std::abs(d-11.9809)<1E-4);

    vecgeom::GenTrapImplementation<vecgeom::translation::kGeneric,
        vecgeom::rotation::kIdentity>::DistanceToIn<kScalar>(trapUnplaced3, Transformation3D(5.,5.,5.),
             Vec_t(4.546874, 8.865740, 8.098616), Vec_t(-0.382036, -0.561047, -0.734353), step, d);
//    std::cerr << "final " << d << "\n"; //  1.86284
    assert( std::abs(d-1.86284)<1.E-4);

    // should have a macro for this
    vecgeom::GenTrapImplementation<vecgeom::translation::kGeneric,
            vecgeom::rotation::kIdentity>::DistanceToIn<kScalar>(trapUnplaced3, Transformation3D(5.,5.,5.),
                 Vec_t(-3.642379, -7.785746, 18.814608), Vec_t(0.390819, 0.361792, -0.846385), step, d);
    //    std::cerr << "final " << d << "\n"; //  27.2668
        assert( std::abs(d-27.2668 )<1.E-4);

    vecgeom::GenTrapImplementation<vecgeom::translation::kGeneric,
            vecgeom::rotation::kIdentity>::DistanceToIn<kScalar>(trapUnplaced3, Transformation3D(5.,5.,5.),
                        Vec_t(-7.569044, -4.983397, 4.661511), Vec_t(0.838801, 0.531342, -0.118692), step, d);
    std::cerr << "final " << d << "\n"; //  13.5376
    assert( std::abs(d-13.5376 )<1.E-4);

    // see if root result is actually good
    Vec_t localp;
    vecgeom::GenTrapImplementation<vecgeom::translation::kGeneric,
             vecgeom::rotation::kIdentity>::Contains<kScalar>(trapUnplaced3, Transformation3D(5.,5.,5.),
             Vec_t(-7.569044, -4.983397, 4.661511)+(13.5376+1.E-4)*Vec_t(0.838801, 0.531342, -0.118692),
             localp,
             completelyInside);
    assert( completelyInside == true  );

    vecgeom::GenTrapImplementation<vecgeom::translation::kGeneric,
               vecgeom::rotation::kIdentity>::DistanceToIn<kScalar>(trapUnplaced3, Transformation3D(5.,5.,5.),
                           Vec_t(2.919067, 8.152522, 15.885466), Vec_t(-0.173286, -0.788227, -0.590482), step, d);
       std::cerr << "final " << d << "\n"; //  13.5376
       assert( std::abs(d-1.49956 )<1.E-4);

   // test with the vector interface
   typedef Vc::double_v Vc_t;
   Vc_t vcd;
   vecgeom::GenTrapImplementation<vecgeom::translation::kGeneric,
             vecgeom::rotation::kIdentity>::DistanceToIn<kVc>(trapUnplaced3, Transformation3D(5.,5.,5.),
             VecVc_t(Vc_t(2.919067),  Vc_t(8.152522),  Vc_t(15.885466)),
             VecVc_t(Vc_t(-0.173286), Vc_t(-0.788227), Vc_t(-0.590482)), Vc_t(100.), vcd);
   std::cerr << "final " << vcd << "\n"; //  13.5376
  // assert( std::abs(d-1.49956 )<1.E-4);


   vecgeom::GenTrapImplementation<vecgeom::translation::kGeneric,
                  vecgeom::rotation::kIdentity>::DistanceToIn<kScalar>(trapUnplaced3, Transformation3D(5.,5.,5.),
                              Vec_t(5.112479, 7.682771, 12.854574), Vec_t(-0.796774, -0.503014, -0.334855), step, d);
   std::cerr << "final " << d << "\n";
//    vecgeom::GenTrapImplementation<vecgeom::translation::kGeneric,
//               vecgeom::rotation::kIdentity>::DistanceToIn<kScalar>(trapUnplaced3, Transformation3D(5.,5.,5.),
//                       Vec_t(0.231799, -7.654186, 1.220068), Vec_t(0.298617, 0.862225, -0.409141), step, d);
//    std::cerr << "final " << d << "\n"; //  11.5825
//
//    assert( std::abs(d-11.5825 ) < 1.E-4 );

}

int main()
{
  createGenTrap();
}


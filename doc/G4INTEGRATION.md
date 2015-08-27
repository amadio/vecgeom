
       Instructions to run Geant4 applications using VecGeom shapes
       ============================================================


An important milestone of the GeantV project is being able to compare the performance of
VecGeom shapes with those from USolids and/or Geant4.  However, it is very difficult to
make such comparisons sensibly, unless one can ensure that the geometry libraries can be 
used interchangeably.

It is possible to run Geant4 jobs using either of Geant4, USolids or VecGeom solids.
Geant4 can be configured to run USolids shapes.  VecGeom has absorbed the USolids source
code, providing its own USolids library, which can be transparently called by a properly
configured Geant4 application.  No change is necessary at the detector construction stage
of the user code for a standard Geant4 application.


* Building VecGeom and USolids
  ============================

Both USolids and VecGeom libraries need to be available when building the Geant4 application.
There is a switch USOLIDS_VECGEOM in CMakeLists.txt, which controls whether the USolids shape
implementations will be used, or the VecGeom implementation will be used instead:

  + USOLIDS_VECGEOM=OFF (default) uses USolids implementations for all USolids shapes

  + USOLIDS_VECGEOM=ON  modifies USolids shapes to inherit from VecGeom shapes, so that the
       VecGeom geometrical algorithms are used instead

Other switches specially configured in our integration tests were BACKEND=Scalar, USOLIDS=ON 
and GEANT4=OFF.

Moreover, the USolids and VecGeom libraries must be properly installed, so that their compiler
configurations can be used from the Geant4 build.  For instance, assuming that the vecgeom
sources are located under $VGSOURCE:

----------
Note: the Geant4-integrated VecGeom code is available from the lima/jira-151 branch only,
      until it gets merged into the devel branch.
-----------

   cd somewhere
   export TOPDIR=`pwd`
   git clone ssh://git@gitlab.cern.ch:7999/VecGeom/VecGeom.git
   cd VecGeom
   export VGSOURCE=`pwd`
   git checkout lima/jira-151

   #.. configuration to use USolids algorithms
   mkdir -p ${TOPDIR}/usolids-build
   cd ${TOPDIR}/usolids-build
   cmake -DBACKEND=Scalar -DGEANT4=OFF -DUSOLIDS=ON \
       [...other vecgeom switches...] \
       -DUSOLIDS_VECGEOM=OFF -DCMAKE_INSTALL_PREFIX=${TOPDIR}/usolids \
       ${VGSOURCE}
   make -j8 install

and

   #.. configuration to use VecGeom algorithms
   mkdir ${TOPDIR}/vecgeom-build
   cd ${TOPDIR}/vecgeom-build
   cmake -DBACKEND=Scalar -DGEANT4=OFF -DUSOLIDS=ON \
       [...other vecgeom switches...] \
       -DUSOLIDS_VECGEOM=ON -DCMAKE_INSTALL_PREFIX=${TOPDIR}/vecgeom \
       ${VGSOURCE}
   make -j8 install

Note the different settings for CMAKE_INSTALL_PREFIX above.  Those will be needed
later, to properly direct Geant4 to use either USolids or VecGeom (environment
variable USolids_DIR).



* Building Geant4 to use VecGeom or USolids
  =========================================

Geant4 currently offers the option to use the USolids shape implementations to run Geant4 
applications.  This can be enabled by setting a couple Geant4 variables when building Geant4 
libraries:

   -DGEANT4_USE_USOLIDS=ON          (default is OFF)
   -DGEANT4_USE_SYSTEM_USOLIDS=ON   (default is OFF)

VecGeom was designed to be a drop-in replacement of USolids, by providing its own version of
USolids shapes, which can be re-configured to inherit from VecGeom shapes instead.  Since 
VecGeom requires C++11 compliance, we suggest requiring the same compliance during Geant4
build as well:

   -DGEANT4_BUILD_CXXSTD=c++11   (default is c++98)

For VecGeom integration tests, Geant4 release 10.2-beta or greater is required.
In this tutorial we assume that the Geant4 sources were unpacked at $SOURCE defined below:

   VERSION=10.02.b01
   G4SOURCE=${TOPDIR}/geant/geant4.${VERSION}

Before building Geant4, copy a few extra files, which have not yet been included yet into 
Geant4 releases:

   wget http://home.fnal.gov/~lima/download/G4UTorus.hh
   wget http://home.fnal.gov/~lima/download/G4UTorus.cc
   wget http://home.fnal.gov/~lima/download/G4Torus.hh
   wget http://home.fnal.gov/~lima/download/G4Torus.cc
   wget http://home.fnal.gov/~lima/download/G4UParaboloid.hh
   wget http://home.fnal.gov/~lima/download/G4UParaboloid.cc
   wget http://home.fnal.gov/~lima/download/G4Paraboloid.hh
   wget http://home.fnal.gov/~lima/download/G4Paraboloid.cc
   wget http://home.fnal.gov/~lima/download/csg-sources.cmake
   wget http://home.fnal.gov/~lima/download/specific-sources.cmake

   mv G4UTorus.hh       ${G4SOURCE}/source/geometry/solids/CSG/include/
   mv G4Torus.hh        ${G4SOURCE}/source/geometry/solids/CSG/include/
   mv G4UTorus.cc       ${G4SOURCE}/source/geometry/solids/CSG/src/
   mv G4Torus.cc        ${G4SOURCE}/source/geometry/solids/CSG/src/
   mv csg-sources.cmake ${G4SOURCE}/source/geometry/solids/CSG/sources.cmake
   mv G4UParaboloid.hh       ${G4SOURCE}/source/geometry/solids/specific/include/
   mv G4Paraboloid.hh        ${G4SOURCE}/source/geometry/solids/specific/include/
   mv G4UParaboloid.cc       ${G4SOURCE}/source/geometry/solids/specific/src/
   mv G4Paraboloid.cc        ${G4SOURCE}/source/geometry/solids/specific/src/
   mv specific-sources.cmake ${G4SOURCE}/source/geometry/solids/specific/sources.cmake

We suggest to build two versions of Geant4, one for each geometry version of USolids or
Vecgeom.  Here are the one-time configurations to build each one of the Geant4 libraries:

   #.. Configuring Geant4 to use USolids
   BUILD=${TOPDIR}/geant/build-g4-usolids
   INSTALL=${TOPDIR}/geant/install-${VERSION}-usolids
   export USolids_DIR=${TOPDIR}/usolids/lib/CMake/USolids/
   #.. then build Geant4, see below

   #.. Configuring Geant4 to use VecGeom
   BUILD=${TOPDIR}/geant/build-g4-vecgeom
   INSTALL=${TOPDIR}/geant/install-${VERSION}-vecgeom
   export USolids_DIR=${TOPDIR}/vecgeom/lib/CMake/USolids/
   #.. then build Geant4, see below


and here for the Geant4 cmake and build commands, which are common for both cases above.
First create directories for build and installation, and setup symlink to external data files:

   mkdir -p ${BUILD}
   mkdir -p ${INSTALL}/share/Geant4-10.2.0
   ln -sf ~/geant/data ${INSTALL}/share/Geant4-10.2.0/data

----------
Note: instructions are not given here on the external Geant4 data files.
      It is assumed that their 10.1.1 versions are installed on ~/geant/data.
     The commands above must be adjusted accordingly. 
----------

   #.. compile and build Geant4
   cd ${BUILD}
   cmake -DCMAKE_INSTALL_PREFIX=${INSTALL} \
      -DGEANT4_USE_USOLIDS=ON \
      -DGEANT4_USE_SYSTEM_USOLIDS=ON \
      -DGEANT4_BUILD_CXXSTD=c++11 \
      -DGEANT4_INSTALL_DATADIR=${INSTALL}/share/Geant4-10.2.0/data \
      -DGEANT4_USE_GDML=ON \
      -DGEANT4_BUILD_MULTITHREADED=OFF \
      ${G4SOURCE}

   make -j8 install

Note the ${INSTALL} directories for the two versions (USolids and VecGeom), to be 
used later when building the Geant4 application.



* Building a Geant4 example for testing
  =====================================

This tutorial uses Geant4 example under ${G4SOURCE}/examples/basic/B1:

  #.. copy the basic/B1 example source tree from G4 release area
  cd $TOPDIR
  mkdir b1
  cp -r ${G4SOURCE}/examples/basic/B1/*  b1/


This example has been extended, to include a larger set of shapes to be exercised.
The modified files are needed:

  #.. copy modified files which instantiate more shapes for testing
  cd ${TOPDIR}/b1/src
  wget http://home.fnal.gov/~lima/download/B1DetectorConstruction.cc
  wget http://home.fnal.gov/~lima/download/B1PrimaryGeneratorAction.cc

  #.. build B1 example against USolids version of Geant4
  MODE=usolids
  source ${TOPDIR}/geant/install-${VERSION}-${MODE}/bin/geant4.sh
  mkdir ${TOPDIR}/b1/build-${MODE}
  cd ${TOPDIR}/b1/build-${MODE}
  cmake -DGeant4_DIR=${TOPDIR}/geant/install-${VERSION}-${MODE}/lib/Geant4-10.2.0  ..
  make -j8

  #.. build B1 example against VecGeom version of Geant4
  Geant4_DIR=${TOPDIR}/geant/install-${VERSION}-vecgeom/lib/Geant4-10.2.0
  source ${Geant4_DIR}/bin/geant4.sh
  mkdir ${TOPDIR}/b1/build-vecgeom
  cd ${TOPDIR}/b1/build-vecgeom
  cmake -DGeant4_DIR=${TOPDIR}/geant/install-${VERSION}-vecgeom/lib/Geant4-10.2.0  ..
  make -j8

Then repeat the last two block above for MODE=vecgeom.

At this point both USolids-based and VecGeom-based Geant4 jobs are ready to be run.


* Running and testing

The runtime setup is much simpler, and most of the steps above are not needed.
A setup script is available to help with this:

  cd ${TOPDIR}/b1
  wget http://home.fnal.gov/~lima/download/setmeup.sh

The file setmeup.sh must be adapted according to the local environment, specially
the TOPDIR environment variable.

A typical session would then look like this:

  #.. set up to run for either mode (e.g. vecgeom)
  cd ${TOPDIR}/b1
  source setmeup.sh vecgeom

  #.. run the Geant4 job 
  exampleB1 run1.mac

A callgrind session can be used to show that vecgeom shapes are actually used:

  valgrind --tool=callgrind exampleB1 run1.mac

  kcachegrind callgrind.out.<PID>


Note: The torus and paraboloid shapes cannot be used in USolids mode.
      Therefore, shapes 7 and 8 must be commented out in b1/src/B1DetectorConstruction.cc
      before trying to run it.  Otherwise the job will probably abort at geometry validation
      stage.

Please send any comments to lima@fnalSPAMNOT.gov.


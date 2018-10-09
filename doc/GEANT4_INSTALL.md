
       Instructions to run Geant4 applications using VecGeom shapes
       ============================================================


It is possible to run Geant4 jobs replacing the original Geant4 solids with
VecGeom implementations.
Geant4 itself can be configured to build with VecGeom shapes.
VecGeom implements the evolution (currently under active develpment) of the
USolids source code, and can be transparently adopted, providing a properly
configured Geant4 installation. No code changes are necessary for the user
code in a standard Geant4 application.


* Building VecGeom
  ================

The VecGeom library needs to be built and installed before configuring Geant4.
These are the CMake switches required for building VecGeom library:

  -DBACKEND=Scalar -DGEANT4=OFF

Assuming that the VecGeom sources are located under $VGSOURCE:

   cd somewhere
   export TOPDIR=`pwd`
   git clone https://gitlab.cern.ch/VecGeom/VecGeom.git
   cd VecGeom
   export VGSOURCE=`pwd`

   #.. configuration to use VecGeom algorithms
   mkdir ${TOPDIR}/vecgeom-build
   cd ${TOPDIR}/vecgeom-build
   cmake -DBACKEND=Scalar -DGEANT4=OFF \
       [...other optional vecgeom switches...] \
       -DCMAKE_INSTALL_PREFIX=${TOPDIR}/vecgeom \
       ${VGSOURCE}
   make -j8 install


* Building Geant4 to use VecGeom
  ==============================

Geant4 currently offers the option to adopt VecGeom shape implementations to run
Geant4 applications. This can be enabled by setting a configuration variable
when building Geant4:

   -DGEANT4_USE_USOLIDS=ON          (default is OFF)

Geant4 release 10.5 or greater is required.
Assuming the Geant4 sources were unpacked at $SOURCE defined below:

   VERSION=10.05
   G4SOURCE=${TOPDIR}/geant4/geant4.${VERSION}

Here are the one-time configurations to build the Geant4 libraries with VecGeom:

   #.. Configuring Geant4 to use VecGeom
   G4BUILD=${TOPDIR}/geant4/build-g4-vecgeom
   G4INSTALL=${TOPDIR}/geant4/install-${VERSION}-vecgeom
   export VecGeom_DIR=${TOPDIR}/vecgeom/lib/cmake/VecGeom/

   ###.. then build Geant4, see options below:

   #.. Option 1: Standard compilation and installation of Geant4,
   #   replacing all available solids
   cd ${G4BUILD}
   cmake -DCMAKE_INSTALL_PREFIX=${G4INSTALL} \
      -DGEANT4_USE_USOLIDS="all" \
      -DGEANT4_INSTALL_DATADIR=${G4INSTALL}/share/Geant4-10.5.0/data \
      -DGEANT4_USE_GDML=ON \
      #.. any other configuration switch
      ${G4SOURCE}

   #.. Or Option 2: Standard compilation and installation of Geant4,
   #   replacing only a limited set of solids
   cd ${G4BUILD}
   cmake -DCMAKE_INSTALL_PREFIX=${G4INSTALL} \
      -DGEANT4_USE_USOLIDS="box;trap" \
      -DGEANT4_INSTALL_DATADIR=${G4INSTALL}/share/Geant4-10.5.0/data \
      -DGEANT4_USE_GDML=ON \
      #.. any other configuration switch
      ${G4SOURCE}

   make -j8 install

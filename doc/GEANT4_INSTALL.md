
       Instructions to run Geant4 applications using USolids/VecGeom shapes
       ====================================================================


It is possible to run Geant4 jobs replacing the original Geant4 solids with
either USolids or VecGeom implementations.
Geant4 itself can be configured to build with USolids/VecGeom shapes.
VecGeom implements the evolution (currently under active develpment) of the
USolids source code, and can be transparently adopted, providing a properly
configured Geant4 installation. No code changes are necessary for the user
code in a standard Geant4 application.


* Building USolids and VecGeom
  ============================

Both USolids and VecGeom libraries need to be installed before configuring
Geant4.
A switch USOLIDS_VECGEOM controls whether to use the USolids implementations
or the VecGeom implementation:

  + USOLIDS_VECGEOM=OFF (default) uses USolids implementations for all USolids
    shapes

  + USOLIDS_VECGEOM=ON  enables VecGeom implementation of the shapes, so that
    the VecGeom algorithms are used instead

Other switches required for the installation are: BACKEND=Scalar, USOLIDS=ON 
and GEANT4=OFF.

Assuming that the VecGeom sources are located under $VGSOURCE:

   cd somewhere
   export TOPDIR=`pwd`
   cd VecGeom
   export VGSOURCE=`pwd`

   #.. configuration to use VecGeom algorithms
   mkdir ${TOPDIR}/vecgeom-build
   cd ${TOPDIR}/vecgeom-build
   cmake -DBACKEND=Scalar -DGEANT4=OFF -DUSOLIDS=ON \
       [...other vecgeom switches...] \
       -DUSOLIDS_VECGEOM=ON -DCMAKE_INSTALL_PREFIX=${TOPDIR}/vecgeom \
       ${VGSOURCE}
   make -j8 install

or (obsolete):

   #.. configuration to use USolids algorithms
   mkdir -p ${TOPDIR}/usolids-build
   cd ${TOPDIR}/usolids-build

   cmake -DBACKEND=Scalar -DGEANT4=OFF -DUSOLIDS=ON \
       [...other vecgeom switches...] \
       -DUSOLIDS_VECGEOM=OFF -DCMAKE_INSTALL_PREFIX=${TOPDIR}/usolids \
       ${VGSOURCE}
   make -j8 install


* Building Geant4 to use VecGeom
  ==============================

Geant4 currently offers the option to adopt VecGeom shape implementations to run
Geant4 applications. This can be enabled by setting a configuration variable
when building Geant4:

   -DGEANT4_USE_USOLIDS=ON          (default is OFF)

Geant4 release 10.3 or greater is required.
Assuming the Geant4 sources were unpacked at $SOURCE defined below:

   VERSION=10.03
   G4SOURCE=${TOPDIR}/geant/geant4.${VERSION}

Here are the one-time configurations to build the Geant4 libraries with either
USolids or VecGeom:

   #.. Configuring Geant4 to use VecGeom
   G4BUILD=${TOPDIR}/geant/build-g4-vecgeom
   G4INSTALL=${TOPDIR}/geant/install-${VERSION}-vecgeom
   export USolids_DIR=${TOPDIR}/vecgeom/lib/CMake/USolids/
   #.. then build Geant4, see below

   #.. Or.. Configuring Geant4 to use USolids (obsolete)
   G4BUILD=${TOPDIR}/geant/build-g4-usolids
   G4INSTALL=${TOPDIR}/geant/install-${VERSION}-usolids
   export USolids_DIR=${TOPDIR}/usolids/lib/CMake/USolids/
   #.. then build Geant4, see below

   #.. Standard compilation and installation of Geant4,
   #   replacing all available solids
   cd ${G4BUILD}
   cmake -DCMAKE_INSTALL_PREFIX=${G4INSTALL} \
      -DGEANT4_USE_USOLIDS="all" \
      -DGEANT4_INSTALL_DATADIR=${G4INSTALL}/share/Geant4-10.3.0/data \
      -DGEANT4_USE_GDML=ON \
      #.. any other configuration switch
      ${G4SOURCE}

   #.. Or .. Standard compilation and installation of Geant4,
   #   replacing only a limited set of solids
   cd ${G4BUILD}
   cmake -DCMAKE_INSTALL_PREFIX=${G4INSTALL} \
      -DGEANT4_USE_USOLIDS="box;trap" \
      -DGEANT4_INSTALL_DATADIR=${G4INSTALL}/share/Geant4-10.3.0/data \
      -DGEANT4_USE_GDML=ON \
      #.. any other configuration switch
      ${G4SOURCE}

   make -j8 install

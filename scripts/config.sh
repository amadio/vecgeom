#!/bin/bash

# defaults

BACKEND=${BACKEND:-"-DVc=ON  -DVC_ACCELERATION=ON -DBACKEND=Vc"}
BUILD_TYPE=${BUILD_TYPE:-"Release"}
SRCDIR=${SRCDIR:-${HOME}/src/vecgeom}
DESTDIR=${DESTDIR:-${PWD}}

BENCHMARK="ON"
CTEST="ON"
GEANT4="ON"
ROOT="ON"
USOLIDS="ON"

# process options

for option in $@; do
case ${option} in
	# compilers
	icc|ICC|intel)
	export CC=icc CXX=icpc
	;;

	gcc|GCC|GNU)
	export CC=gcc CXX=g++
	;;

	clang|Clang)
	export CC=gcc CXX=g++
	;;

	# backends
	scalar|Scalar)
	BACKEND="-DVc=OFF -DVC_ACCELERATION=OFF -DBACKEND=Scalar"
	;;

	vc|Vc|VC)
	BACKEND="-DVc=ON  -DVC_ACCELERATION=ON -DBACKEND=Vc"
	;;

	# other options
	cuda|CUDA)
	CUDA="-DCUDA=ON -DNO_SPECIALIZATION=ON -DCUDA_VOLUME_SPECIALIZATION=OFF"
	;;

	test|ctest)     CTEST="ON"  ;;
	notest|noctest) CTEST="OFF" ;;

	bench|benchmark)     BENCHMARK="ON"  ;;
	nobench|nobenchmark) BENCHMARK="OFF" ;;

	usolids)   USOLIDS="ON"  ;;
	nousolids) USOLIDS="OFF" ;;

	geant4)    GEANT4="ON"  ;;
	nogeant4)  GEANT4="OFF" ;;

	root)      ROOT="ON"  ;;
	noroot)    ROOT="OFF" ;;
esac
done

cmake ${SRCDIR} -DCMAKE_INSTALL_PREFIX=${DESTDIR}          \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ${BACKEND} ${CUDA}    \
    -DUSOLIDS=${USOLIDS} -DROOT=${ROOT} -DGEANT4=${GEANT4} \
    -DBENCHMARK=${BENCHMARK} -DCTEST=${CTEST} ${EXTRA_CONF}


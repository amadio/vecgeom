#!/bin/bash -x

export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8

THIS=$(dirname ${BASH_SOURCE[0]})

# first arguments is the source directory
if [ $# -ge 5 ]; then
  LABEL=$1 ; shift
  COMPILER=$1 ; shift
  BUILDTYPE=$1 ; shift
  EXTERNALS=$1 ; shift
  BACKEND=$1 ; shift
else
  echo "$0: expecting 5 arguments [LABEL] [COMPILER] [BUILDTYPE] [EXTERNALS] [BACKEND]"
  return
fi

export BUILDTYPE
export COMPILER

if [ "$WORKSPACE" == "" ]; then WORKSPACE=$PWD; fi
PLATFORM=`$THIS/getPlatform.py`
COMPATIBLE=`$THIS/getCompatiblePlatform.py $PLATFORM`
ARCH=$(uname -m)


# Set up the externals against devgeantv in CVMFS
if [ -a /cvmfs/sft.cern.ch/lcg/views/devgeantv/$EXTERNALS/$PLATFORM ]; then
  source /cvmfs/sft.cern.ch/lcg/views/devgeantv/$EXTERNALS/$PLATFORM/setup.sh
elif [ -a /cvmfs/sft.cern.ch/lcg/views/devgeantv/$EXTERNALS/$COMPATIBLE ]; then
  source /cvmfs/sft.cern.ch/lcg/views/devgeantv/$EXTERNALS/$COMPATIBLE/setup.sh
elif [[ $PLATFORM == *slc6* ]] || [[ $PLATFORM == *centos7* ]]; then
  export PATH=/cvmfs/sft.cern.ch/lcg/contrib/CMake/3.7.0/Linux-$ARCH/bin:${PATH}
else
  echo "No externals for $PLATFORM in /cvmfs/sft.cern.ch/lcg/views/devgeantv/$EXTERNALS"
fi

if [ $LABEL == slc6 ] || [ $LABEL == gvslc6 ] || [ $LABEL == cc7 ] || [ $LABEL == cuda7 ] || [ $LABEL == slc6-physical ] || [  $LABEL == continuous-sl6 ] || [  $LABEL == continuous-cuda7 ] || [ $LABEL == continuous-xeonphi ] || [ $LABEL == c7-checker ]
then
  kinit sftnight@CERN.CH -5 -V -k -t /ec/conf/sftnight.keytab
elif [ $LABEL == xeonphi ]
then
  kinit sftnight@CERN.CH -5 -V -k -t /data/sftnight/ec/conf/sftnight.keytab
fi

if [[ $COMPILER == *gcc* ]]; then
  echo "The correct compiler should be setup by the externals ..."
elif [[ $COMPILER == *native* && $PLATFORM == *mac* ]]; then
  export LD_LIBRARY_PATH=/usr/local/gfortran/lib
  export PATH=/usr/bin:/usr/local/bin:/opt/X11/bin
  export CC=`which clang`
  export CXX=`which clang++`
  export FC=`which gfortran`
elif [[ $PLATFORM == *native* ]]; then
  export CC=`which gcc`
  export CXX=`which g++`
  export FC=`which gfortran`
elif [[ $COMPILER == *icc* ]]; then
  iccyear=2013
  icc14year=2013
  icc15year=2015
  icc16year=2016
  COMPILERyear=${COMPILER}year
  iccgcc=4.9
  icc14gcc=4.9
  icc15gcc=4.9
  icc16gcc=4.9
  GCCversion=${COMPILER}gcc
  ARCH=$(uname -m)
  . /afs/cern.ch/sw/lcg/contrib/gcc/${!GCCversion}/${ARCH}-slc6/setup.sh
  . /afs/cern.ch/sw/IntelSoftware/linux/setup.sh
  . /afs/cern.ch/sw/IntelSoftware/linux/${ARCH}/xe${!COMPILERyear}/bin/ifortvars.sh intel64
  . /afs/cern.ch/sw/IntelSoftware/linux/${ARCH}/xe${!COMPILERyear}/bin/iccvars.sh intel64
  export CC=icc
  export CXX=icc
  export FC=ifort
elif [[ $COMPILER == *clang* ]]; then
  clang34version=3.4
  clang35version=3.5
  clang36version=3.6
  clang37version=3.7
  clang38version=3.8
  COMPILERversion=${COMPILER}version
  clang34gcc=48
  clang35gcc=49
  clang37gcc=49
  clang38gcc=49
  GCCversion=${COMPILER}gcc
  . /afs/cern.ch/sw/lcg/external/llvm/${!COMPILERversion}/${ARCH}-${LABEL_COMPILER}/setup.sh
  export CC=`which clang`
  export CXX=`which clang++`
  export FC=`which gfortran`
fi

# Setup ccache
dir=$WORKSPACE
while [ $(basename $dir) != workspace ]; do dir=$(dirname $dir); done
export CCACHE_DIR=$dir/ccache
export CCACHE_MAXSIZE=10G

export CMAKE_SOURCE_DIR=$WORKSPACE/VecGeom
export CMAKE_BINARY_DIR=$WORKSPACE/build
export CMAKE_INSTALL_PREFIX=$WORKSPACE/install
export CMAKE_BUILD_TYPE=$BUILDTYPE

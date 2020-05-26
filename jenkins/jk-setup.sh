#!/bin/bash

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
export BACKEND

if [ "$WORKSPACE" == "" ]; then WORKSPACE=$PWD; fi
PLATFORM=`$THIS/getPlatform.py`
COMPATIBLE=`$THIS/getCompatiblePlatform.py $PLATFORM`
ARCH=$(uname -m)

# Set up the externals against devgeantv in CVMFS
if [ -a /cvmfs/sft.cern.ch/lcg/views/devgeantv/$EXTERNALS/$PLATFORM ]; then
  echo source /cvmfs/sft.cern.ch/lcg/views/devgeantv/$EXTERNALS/$PLATFORM/setup.sh
  source /cvmfs/sft.cern.ch/lcg/views/devgeantv/$EXTERNALS/$PLATFORM/setup.sh
elif [ -a /cvmfs/sft.cern.ch/lcg/views/devgeantv/$EXTERNALS/$COMPATIBLE ]; then
  echo source /cvmfs/sft.cern.ch/lcg/views/devgeantv/$EXTERNALS/$COMPATIBLE/setup.sh
  source /cvmfs/sft.cern.ch/lcg/views/devgeantv/$EXTERNALS/$COMPATIBLE/setup.sh
else
  echo "No externals for $PLATFORM in /cvmfs/sft.cern.ch/lcg/views/devgeantv/$EXTERNALS"
  exit 1
fi

# Setup kerberos for some of the labels
if [ $LABEL == slc6 ] || [ $LABEL == centos7 ] || [ $LABEL == cuda ] || [ $LABEL == physical ]; then
  kinit sftnight@CERN.CH -5 -V -k -t /ec/conf/sftnight.keytab
fi

if [ $LABEL == cuda10 ]; then
  source /cvmfs/sft.cern.ch/lcg/contrib/cuda/10.2/$(echo $PLATFORM| cut -d'-' -f 1,2)/setup.sh
fi

if [[ $COMPILER == *gcc* ]]; then
  echo "The correct compiler should be setup by the externals ..."
elif [[ $COMPILER == *native* && $PLATFORM == *mac* ]]; then
  export LD_LIBRARY_PATH=/usr/local/gfortran/lib
  export PATH=/usr/bin:/usr/local/bin:/opt/X11/bin
  export CC=`which clang`
  export CXX=`which clang++`
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
  clang10version=10
  clang9version=9
  clang8version=8
  COMPILERversion=${COMPILER}version
  source /cvmfs/sft.cern.ch/lcg/contrib/clang/${!COMPILERversion}/$(echo $PLATFORM| cut -d'-' -f 1,2)/setup.sh
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

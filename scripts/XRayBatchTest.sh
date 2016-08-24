# A script to batch-test a list of volumes using the XRayBenchmarker
# test acceptance currently based on strict equality cmp to ROOT
# (which we might have to be relaxed/modified)

detector=$1
filewithvolumelist=$2

NC='\033[0m' # no color
GREEN='\033[1;32m'
RED='\033[0;31m'

for v in `cat ${filewithvolumelist}`; do
  for d in "x" "z" "y"; do
  ./XRayBenchmarkFromRootFile ${detector} ${v} ${d} 250 2> errlog${v}${d} > log${v}${d}
  diff volumeImage_${v}${d}_VOXELIZED__ROOT.bmp volumeImage_${v}${d}_VOXELIZED__VecGeomNEW.bmp
  rc=$?
  if [[ $rc == 0 ]]; then 
   printf "${v} direction ${d} ${GREEN}passes${NC}\n"
  else
   printf "${v} direction ${d} ${RED}fails${NC}\n"
  fi
  done
done
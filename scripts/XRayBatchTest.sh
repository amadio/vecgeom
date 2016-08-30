# A script to batch-test a list of volumes using the XRayBenchmarker
# uses GNU parallel in order to run multiple XRayTasks concurrently

detector=$1
filewithvolumelist=$2

function XRayTask(){
  NC='\033[0m' # no color
  GREEN='\033[1;32m'
  RED='\033[0;31m'
  errlog="XRayBatchErrlog"
  log="XRayBatchLog"
  detector=$1
  v=$2
  d=$3
  timeout 30m ./XRayBenchmarkFromROOTFile ${detector} ${v} ${d} 250 2> ${errlog}${v}${d} > ${log}${v}${d}
  # diff volumeImage_${v}${d}_VOXELIZED__ROOT.bmp volumeImage_${v}${d}_VOXELIZED__VecGeomNEW.bmp
  rc=$?
  if [[ $rc == 0 ]]; then 
   printf "${v} direction ${d} ${GREEN}passes${NC}\n"
   # if pass cleanup file
   rm volumeImage_${v}${d}*.bmp
   rm ${errlog}${v}${d}
  else
   printf "${v} direction ${d} ${RED}fails${NC}\n"
  fi
}
export -f XRayTask

for d in "x" "z" "y"; do
for v in `cat ${filewithvolumelist}`; do
    sem -j+0 XRayTask $detector ${v} ${d}
  done
done
sem --wait

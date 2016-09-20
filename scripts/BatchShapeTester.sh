# A script to batch-test a list of volumes using the shapetester tool 
# "shape_testfromROOTFile"
# uses GNU parallel in order to run multiple tasks concurrently

detector=$1
filewithvolumelist=$2

function ShapeTestTask(){
  NC='\033[0m' # no color
  GREEN='\033[1;32m'
  RED='\033[0;31m'
  errlog="ShapeTestBatchErrlog"
  log="ShapeTestBatchLog"
  detector=$1
  v=$2
  ./shape_testFromROOTFile ${detector} ${v} 2> ${errlog}${v} > ${log}${v}
  rc=$?
  if [[ $rc == 0 ]]; then
   printf "${v} ${GREEN}passes${NC}\n"
   # if pass cleanup file
   rm ${errlog}${v}$
  else
   printf "${v} ${RED}fails${NC}\n"
  fi
}
export -f ShapeTestTask

for v in `cat ${filewithvolumelist}`; do
    sem -j+0 ShapeTestTask $detector ${v}
done
sem --wait

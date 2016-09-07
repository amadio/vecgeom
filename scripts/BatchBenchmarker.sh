# A script to batch-test a list of volumes using the Benchmarker tool "BenchmarkShapeFromROOTFile"
# uses GNU parallel in order to run multiple tasks concurrently

detector=$1
filewithvolumelist=$2

function BenchTask(){
  NC='\033[0m' # no color
  GREEN='\033[1;32m'
  RED='\033[0;31m'
  errlog="BenchBatchErrlog"
  log="BenchBatchLog"
  detector=$1
  v=$2
  timeout 30m ./BenchmarkShapeFromROOTFile ${detector} ${v} 2> ${errlog}${v} > ${log}${v}
  rc=$?
  if [[ $rc == 0 ]]; then
   printf "${v} ${GREEN}passes${NC}\n"
   # if pass cleanup file
   rm ${errlog}${v}$
  else
   printf "${v} ${RED}fails${NC}\n"
  fi
}
export -f BenchTask

for v in `cat ${filewithvolumelist}`; do
    sem -j+0 BenchTask $detector ${v}
done
sem --wait

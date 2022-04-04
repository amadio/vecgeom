#!/usr/bin/env bash
# running the static code analysis as a unit test

#FIXME:
  # restrict test to actual new lines of code

echo "Avoid ctest truncation of output: CTEST_FULL_OUTPUT";


# get the clang-tidy plugin
plugin=$1
# get clang-tidy-binary
clangtidybinary=$2
# the clang-tidy (parallel) run-script
script=$3
# the build dir (where to find the compilations database)
builddir=$4

# obtain the source files changed in this request
# an error reported outside of files changed might be discarded
fileschanged=`git diff origin/master... --name-only`

LOGFILE=static_analysis_log
# this is the way to hook our static analysis to clang-tidy
eval ${script} -clang-tidy-binary ${clangtidybinary} -load=$plugin -checks=-*,vecgeom* -p ${builddir} | tee ${LOGFILE}
grep "error:" ${LOGFILE}
if [ "$?" == "0" ]; then
  # this means that the word "error:" was found in the log

  # check if the erroneous files are part of this commit
  # first of all get the set of files with problems
  fileswithproblems=`grep "error:" ${LOGFILE} | sed 's/:.*//g' | sort | uniq | sed 's/.*VecGeom\///g'`

  # check for intersection with files that where changed:
  # FIXME: get rid of this unelegant O(N^2) solution
  for f in ${fileschanged}; do
    for f2 in ${fileswithproblems}; do
      if [ "${f}" == "${f2}" ]; then
        echo "FILE ${f} IS PROBLEMATIC AND PART OF THIS COMMIT";
        exit 1
      fi
    done
  done
fi
exit 0


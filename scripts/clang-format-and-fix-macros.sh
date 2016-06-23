#!/bin/bash

macros=(
	__host__
	__device__
	VECCORE_CUDA_HOST
	VECCORE_CUDA_DEVICE
	VECCORE_CUDA_HOST_DEVICE
	VECCORE_FORCE_INLINE
	VECCORE_FORCE_NOINLINE
	VECGEOM_CUDA_HEADER_HOST
	VECGEOM_CUDA_HEADER_DEVICE
	VECGEOM_CUDA_HEADER_BOTH
	VECGEOM_INLINE
)

macroShorts=(
    "VECGEOM_INLINE[[:space:]]*VECGEOM_CUDA_HEADER_BOTH @GIGB \1VECGEOM_INLINE\1VECGEOM_CUDA_HEADER_BOTH\1"
    "VECGEOM_CUDA_HEADER_BOTH[[:space:]]*VECGEOM_INLINE @GIGB \1VECGEOM_INLINE\1VECGEOM_CUDA_HEADER_BOTH\1"
    "VECGEOM_CUDA_HEADER_BOTH @GB \1VECGEOM_CUDA_HEADER_BOTH\1"
    "VECGEOM_INLINE[[:space:]]*VECGEOM_CUDA_HEADER_DEVICE @GIGD \1VECGEOM_INLINE\1VECGEOM_CUDA_HEADER_DEVICE\1"
    "VECGEOM_CUDA_HEADER_DEVICE[[:space:]]*VECGEOM_INLINE @GIGD \1VECGEOM_INLINE\1VECGEOM_CUDA_HEADER_DEVICE\1"
    "VECGEOM_CUDA_HEADER_DEVICE @GD \1VECGEOM_CUDA_HEADER_DEVICE\1"
    "VECGEOM_INLINE[[:space:]]*VECGEOM_CUDA_HEADER_HOST @GIGH \1VECGEOM_INLINE VECGEOM_CUDA_HEADER_HOST\1"
    "VECGEOM_CUDA_HEADER_HOST[[:space:]]*VECGEOM_INLINE @GIGH \1VECGEOM_INLINE VECGEOM_CUDA_HEADER_HOST\1"
    "VECGEOM_CUDA_HEADER_HOST @GH \1VECGEOM_CUDA_HEADER_HOST\1"
    "VECGEOM_INLINE @GI \1VECGEOM_INLINE\1"
    "VECCORE_FORCE_INLINE[[:space:]]*VECCORE_CUDA_HOST_DEVICE @CICB \1VECCORE_FORCE_INLINE\1VECCORE_CUDA_HOST_DEVICE\1"
    "VECCORE_CUDA_HOST_DEVICE[[:space:]]*VECCORE_FORCE_INLINE @CICB \1VECCORE_FORCE_INLINE\1VECCORE_CUDA_HOST_DEVICE\1"
    "VECCORE_CUDA_HOST_DEVICE @CB \1VECCORE_CUDA_HOST_DEVICE\1"
    "VECCORE_FORCE_INLINE[[:space:]]*VECCORE_CUDA_DEVICE @CICD \1VECCORE_FORCE_INLINE\1VECCORE_CUDA_DEVICE\1"
    "VECCORE_CUDA_DEVICE[[:space:]]*VECCORE_FORCE_INLINE @CICD \1VECCORE_FORCE_INLINE\1VECCORE_CUDA_DEVICE\1"
    "VECCORE_CUDA_DEVICE @CD \1VECCORE_CUDA_DEVICE\1"
    "VECCORE_FORCE_INLINE[[:space:]]*VECCORE_CUDA_HOST @CICH \1VECCORE_FORCE_INLINE\1VECCORE_CUDA_HOST\1"
    "VECCORE_CUDA_HOST[[:space:]]*VECCORE_FORCE_INLINE @CICH \1VECCORE_FORCE_INLINE\1VECCORE_CUDA_HOST\1"
    "VECCORE_CUDA_HOST @CH \1VECCORE_CUDA_HOST\1"
    "VECCORE_FORCE_INLINE @CI \1VECCORE_FORCE_INLINE\1"
    "VECCORE_FORCE_NOINLINE @CNI \1VECCORE_FORCE_NOINLINE\1"
    "__host__ @H \1__host__\1"
    "__device__ @D \1__device__\1"
    "__host__ __device__ @B \1__host__\1__device__\1"
    "__device__ __host__ @B \1__host__\1__device__\1"
)

patternFrom=""
patternTo=""

for macroInfo in "${macroShorts[@]}" ; do
    macro=${macroInfo%% *}
    values=${macroInfo#* }
    short=${values%% *}
    newPattern=${values#* }
#    printf "%s switch to %s then %s\n" "$values" "$short" "$newPattern"

    patternFrom="${patternFrom}
s/\(\n[[:blank:]]*template[^;{(]*\)>[[:space:]]*${macro}/\1${short}>/g"

    patternTo="${patternTo}
s/ *${short}>\([[:space:]\n]*\)/>${newPattern}/g"
done

# Shorten the macro and move them out of the way so that they have
# no effect on line length calculation.
for file in $@; do
  sed  -i "" -n "
    # if the first line copy the pattern to the hold buffer
    1h
    # if not the first line then append the pattern to the hold buffer
    1!H
    # if the last line then ...
    $ {
      # copy from the hold to the pattern buffer
      g
      # do the search and replace
      ${patternFrom}
      # print
      p
    }" "${file}"
done

clang-format -i "$@"
# Run clang-format a 2nd time, this stability some of the comment positioning.
clang-format -i "$@"

# Put back the macros.
for file in $@; do
  sed  -i "" -n "
    # if the first line copy the pattern to the hold buffer
    1h
    # if not the first line then append the pattern to the hold buffer
    1!H
    # if the last line then ...
    $ {
      # copy from the hold to the pattern buffer
      g
      # do the search and replace
      ${patternTo}
      # print
      p
    }" "${file}"
done


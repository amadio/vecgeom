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


clang-format -i "$@"
for macro in ${macros[@]}; do
   sed -i "" -E -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/^([[:space:]]*)(.*)('${macro}')[[:space:]]+/\1\2\3\
\1/g;' "$@"
done

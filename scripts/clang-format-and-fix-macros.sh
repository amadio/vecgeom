#!/bin/bash

macros=(
	__host__
	__device__
	VECCORE_CUDA_HOST
	VECCORE_CUDA_DEVICE
	VECCORE_CUDA_HOST_DEVICE
	VECCORE_FORCE_INLINE
	VECCORE_FORCE_NOINLINE
	VECGEOM_CUDA_HEADER_BOTH
	VECGEOM_INLINE
)

for file in $@; do
	clang-format -i "${file}"
for macro in ${macros[@]}; do
	sed -i -e "/${macro}/ s/^\(\s*\)\(.*\)\(${macro}\)\s\+/\1\2\3\n\1/g;" "${file}"
done
done

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


# Shorten the macro we will move so that there effect on line length calculation is reduced.
sed -i "" -E \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/VECGEOM_INLINE VECGEOM_CUDA_HEADER_BOTH/@GIGB@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/VECGEOM_CUDA_HEADER_BOTH VECGEOM_INLINE/@GIGB@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/VECGEOM_CUDA_HEADER_BOTH/@GB@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/VECGEOM_INLINE VECGEOM_CUDA_HEADER_DEVICE/@GIGD@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/VECGEOM_CUDA_HEADER_DEVICE VECGEOM_INLINE/@GIGD@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/VECGEOM_CUDA_HEADER_DEVICE/@GD@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/VECGEOM_INLINE VECGEOM_CUDA_HEADER_HOST/@GIGH@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/VECGEOM_CUDA_HEADER_HOST VECGEOM_INLINE/@GIGH@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/VECGEOM_CUDA_HEADER_HOST/@GH@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/VECGEOM_INLINE/@GI@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/VECCORE_FORCE_INLINE VECCORE_CUDA_HOST_DEVICE/@CICB@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/VECCORE_CUDA_HOST_DEVICE VECCORE_FORCE_INLINE/@CICB@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/VECCORE_CUDA_HOST_DEVICE/@CB@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/VECCORE_FORCE_INLINE VECCORE_CUDA_DEVICE/@CICD@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/VECCORE_CUDA_DEVICE VECCORE_FORCE_INLINE/@CICD@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/VECCORE_CUDA_DEVICE/@CD@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/VECCORE_FORCE_INLINE VECCORE_CUDA_HOST/@CICH@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/VECCORE_CUDA_HOST VECCORE_FORCE_INLINE/@CICH@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/VECCORE_CUDA_HOST/@CH@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/VECCORE_FORCE_INLINE/@CI@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/VECCORE_FORCE_NOINLINE/@CNI@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/__host__/@H@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/__device__/@D@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/__host__ __device__/@B@/g' \
    -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/__device__ __host__/@B@/g' \
    "$@"

clang-format -i "$@"

# Put back the macros.

sed -i "" -E \
    -e 's/@GIGB@/VECGEOM_INLINE VECGEOM_CUDA_HEADER_BOTH/g' \
    -e 's/@GIGD@/VECGEOM_INLINE VECGEOM_CUDA_HEADER_DEVICE/g' \
    -e 's/@GIGH@/VECGEOM_INLINE VECGEOM_CUDA_HEADER_HOST/g' \
    -e 's/@GB@/VECGEOM_CUDA_HEADER_BOTH/g' \
    -e 's/@GD@/VECGEOM_CUDA_HEADER_DEVICE/g' \
    -e 's/@GH@/VECGEOM_CUDA_HEADER_HOST/g' \
    -e 's/@GI@/VECGEOM_INLINE/g' \
    -e 's/@CICB@/VECCORE_FORCE_INLINE VECCORE_CUDA_HOST_DEVICE/g' \
    -e 's/@CICD@/VECCORE_FORCE_INLINE VECCORE_CUDA_DEVICE/g' \
    -e 's/@CICH@/VECCORE_FORCE_INLINE VECCORE_CUDA_HOST/g' \
    -e 's/@CB@/VECCORE_CUDA_HOST_DEVICE/g' \
    -e 's/@CD@/VECCORE_CUDA_DEVICE/g' \
    -e 's/@CH@/VECCORE_CUDA_HOST/g' \
    -e 's/@CI@/VECCORE_FORCE_INLINE/g' \
    -e 's/@CNI@/VECCORE_FORCE_NOINLINE/g' \
    -e 's/@B@/__host__ __device__/g' \
    -e 's/@D@/__device__/g' \
    -e 's/@H@/__host__/g' \
    "$@"

# Put the macro on their own line ...

for macro in ${macros[@]}; do
   sed -i "" -E -e '/^([[:space:]]*)[^/#[:space:]].*[^\]$/ s/^([[:space:]]*)(.*)('${macro}')[[:space:]]+/\1\2\3\
\1/g;' "$@"
done

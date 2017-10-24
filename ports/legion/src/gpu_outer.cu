/* Copyright 2017 NVIDIA Corporation
 *
 * The U.S. Department of Energy funded the development of this software 
 * under subcontract B609478 with Lawrence Livermore National Security, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "snap_types.h"
#include "accessor.h"
#include "snap_cuda_help.h"

#include <vector>

using namespace LegionRuntime::Arrays;
using namespace LegionRuntime::Accessor;

template<int GROUPS, int STRIP_SIZE>
__global__
void gpu_flux0_outer_source(const PointerBuffer<GROUPS,double> qi0_ptrs,
                            const PointerBuffer<GROUPS,double> flux0_ptrs,
                            const PointerBuffer<GROUPS,MomentQuad> slgg_ptrs,
                            const int *mat_ptr,
                                  PointerBuffer<GROUPS,double> qo0_ptrs,
                            const ByteOffsetArray<3> qi0_offsets,
                            const ByteOffsetArray<3> flux0_offsets,
                            const ByteOffsetArray<2> slgg_offsets,
                            const ByteOffsetArray<3> mat_offsets,
                            const ByteOffsetArray<3> qo0_offsets)
{
  __shared__ double flux_buffer[GROUPS][STRIP_SIZE];
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z;
  const int group = threadIdx.z;
  const int strip_offset = threadIdx.y * blockDim.x + threadIdx.x;
  // First, update our pointers
  const double *qi0_ptr = qi0_ptrs[group] + x * qi0_offsets[0] +
    y * qi0_offsets[1] + z * qi0_offsets[2];
  const double *flux0_ptr = flux0_ptrs[group] + x * flux0_offsets[0] +
    y * flux0_offsets[1] + z * flux0_offsets[2];
  mat_ptr += x * mat_offsets[0] + y * mat_offsets[1] + z * mat_offsets[2];
  double *qo0_ptr = qo0_ptrs[group] + x * qo0_offsets[0] + 
    y *qo0_offsets[1] + z * qo0_offsets[2];
  // Do a little prefetching of other values we need too
  // Be intelligent about loads, we're trying to keep the slgg
  // matrix in L2 cache so make sure all other loads and stores 
  // are cached with a streaming prefix
  double flux0;
  asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(flux0) : "l"(flux0_ptr) : "memory");
  // Other threads will use the material so cache at all levels
  int mat;
  asm volatile("ld.global.ca.s32 %0, [%1];" : "=r"(mat) : "l"(mat_ptr) : "memory");
  double qo0;
  asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(qo0) : "l"(qi0_ptr) : "memory");
  // Write the value into shared
  flux_buffer[group][strip_offset] = flux0;
  // Can compute our slgg_ptr with the matrix result
#ifdef LEGION_ISSUE_214_FIX
  const MomentQuad *slgg_ptr = slgg_ptrs[group] + mat * slgg_offsets[0];
#else
  const MomentQuad *slgg_ptr = slgg_ptrs[group] + (mat-1) * slgg_offsets[0];
#endif
  // Synchronize when all the writes into shared memory are done
  __syncthreads();
  // Do the math
  #pragma unroll
  for (int g = 0; g < GROUPS; g++) {
    if (g == group)
      continue;
    const MomentQuad *local_slgg = slgg_ptr + g * slgg_offsets[1];
    double cs;
    asm volatile("ld.global.ca.f64 %0, [%1];" : "=d"(cs) : "l"(local_slgg) : "memory");
    qo0 += cs * flux_buffer[g][strip_offset];
  }
  // Write out our result
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(qo0_ptr), "d"(qo0) : "memory");
}

template<int GROUPS, int MAX_X, int MAX_Y>
__host__
void flux0_launch_helper(Rect<3> subgrid_bounds,
                         const std::vector<double*> &qi0_ptrs,
                         const std::vector<double*> &flux0_ptrs,
                         const std::vector<MomentQuad*> &slgg_ptrs,
                         const std::vector<double*> &qo0_ptrs, 
                         const int *mat_ptr, 
                         const ByteOffset qi0_offsets[3], 
                         const ByteOffset flux0_offsets[3],
                         const ByteOffset slgg_offsets[2], 
                         const ByteOffset qo0_offsets[3],
                         const ByteOffset mat_offsets[3])
{
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1;
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;
  dim3 block(gcd(x_range,MAX_X), gcd(y_range,MAX_Y), GROUPS);
  dim3 grid(x_range/block.x, y_range/block.y, z_range);
  gpu_flux0_outer_source<GROUPS,MAX_X*MAX_Y><<<grid,block>>>(
                              PointerBuffer<GROUPS,double>(qi0_ptrs),
                              PointerBuffer<GROUPS,double>(flux0_ptrs),
                              PointerBuffer<GROUPS,MomentQuad>(slgg_ptrs), mat_ptr,
                              PointerBuffer<GROUPS,double>(qo0_ptrs),
                              ByteOffsetArray<3>(qi0_offsets),
                              ByteOffsetArray<3>(flux0_offsets),
                              ByteOffsetArray<2>(slgg_offsets),
                              ByteOffsetArray<3>(mat_offsets),
                              ByteOffsetArray<3>(qo0_offsets));
}

__host__
void run_flux0_outer_source(Rect<3> subgrid_bounds,
                            const std::vector<double*> &qi0_ptrs,
                            const std::vector<double*> &flux0_ptrs,
                            const std::vector<MomentQuad*> &slgg_ptrs,
                            const std::vector<double*> &qo0_ptrs, 
                            const int *mat_ptr, 
                            const ByteOffset qi0_offsets[3], 
                            const ByteOffset flux0_offsets[3],
                            const ByteOffset slgg_offsets[2], 
                            const ByteOffset qo0_offsets[3],
                            const ByteOffset mat_offsets[3], const int num_groups)
{
  // TODO: replace this template madness with Terra
#define GROUP_CASE(g,x,y)                                                           \
  case g:                                                                           \
    {                                                                               \
      flux0_launch_helper<g,x,y>(subgrid_bounds, qi0_ptrs, flux0_ptrs, slgg_ptrs,   \
                               qo0_ptrs, mat_ptr, qi0_offsets, flux0_offsets,       \
                               slgg_offsets, qo0_offsets, mat_offsets);             \
      break;                                                                        \
    }
  switch (num_groups)
  {
    GROUP_CASE(1,32,32)
    GROUP_CASE(2,32,16)
    GROUP_CASE(3,32,8)
    GROUP_CASE(4,32,8)
    GROUP_CASE(5,32,4)
    GROUP_CASE(6,32,4)
    GROUP_CASE(7,32,4)
    GROUP_CASE(8,32,4)
    GROUP_CASE(9,32,2)
    GROUP_CASE(10,32,2)
    GROUP_CASE(11,32,2)
    GROUP_CASE(12,32,2)
    GROUP_CASE(13,32,2)
    GROUP_CASE(14,32,2)
    GROUP_CASE(15,32,2)
    GROUP_CASE(16,32,2)
    GROUP_CASE(17,16,2)
    GROUP_CASE(18,16,2)
    GROUP_CASE(19,16,2)
    GROUP_CASE(20,16,2)
    GROUP_CASE(21,16,2)
    GROUP_CASE(22,16,2)
    GROUP_CASE(23,16,2)
    GROUP_CASE(24,16,2)
    GROUP_CASE(25,16,2)
    GROUP_CASE(26,16,2)
    GROUP_CASE(27,16,2)
    GROUP_CASE(28,16,2)
    GROUP_CASE(29,16,2)
    GROUP_CASE(30,16,2)
    GROUP_CASE(31,16,2)
    GROUP_CASE(32,16,2)
    GROUP_CASE(33,16,1)
    GROUP_CASE(34,16,1)
    GROUP_CASE(35,16,1)
    GROUP_CASE(36,16,1)
    GROUP_CASE(37,16,1)
    GROUP_CASE(38,16,1)
    GROUP_CASE(39,16,1)
    GROUP_CASE(40,16,1)
    GROUP_CASE(41,16,1)
    GROUP_CASE(42,16,1)
    GROUP_CASE(43,16,1)
    GROUP_CASE(44,16,1)
    GROUP_CASE(45,16,1)
    GROUP_CASE(46,16,1)
    GROUP_CASE(47,16,1)
    GROUP_CASE(48,16,1)
    GROUP_CASE(49,16,1)
    GROUP_CASE(50,16,1)
    GROUP_CASE(51,16,1)
    GROUP_CASE(52,16,1)
    GROUP_CASE(53,16,1)
    GROUP_CASE(54,16,1)
    GROUP_CASE(55,16,1)
    GROUP_CASE(56,16,1)
    GROUP_CASE(57,16,1)
    GROUP_CASE(58,16,1)
    GROUP_CASE(59,16,1)
    GROUP_CASE(60,16,1)
    GROUP_CASE(61,16,1)
    GROUP_CASE(62,16,1)
    GROUP_CASE(63,16,1)
    GROUP_CASE(64,16,1)
    GROUP_CASE(65,8,1)
    GROUP_CASE(66,8,1)
    GROUP_CASE(67,8,1)
    GROUP_CASE(68,8,1)
    GROUP_CASE(69,8,1)
    GROUP_CASE(70,8,1)
    GROUP_CASE(71,8,1)
    GROUP_CASE(72,8,1)
    GROUP_CASE(73,8,1)
    GROUP_CASE(74,8,1)
    GROUP_CASE(75,8,1)
    GROUP_CASE(76,8,1)
    GROUP_CASE(77,8,1)
    GROUP_CASE(78,8,1)
    GROUP_CASE(79,8,1)
    GROUP_CASE(80,8,1)
    GROUP_CASE(81,8,1)
    GROUP_CASE(82,8,1)
    GROUP_CASE(83,8,1)
    GROUP_CASE(84,8,1)
    GROUP_CASE(85,8,1)
    GROUP_CASE(86,8,1)
    GROUP_CASE(87,8,1)
    GROUP_CASE(88,8,1)
    GROUP_CASE(89,8,1)
    GROUP_CASE(90,8,1)
    GROUP_CASE(91,8,1)
    GROUP_CASE(92,8,1)
    GROUP_CASE(93,8,1)
    GROUP_CASE(94,8,1)
    GROUP_CASE(95,8,1)
    GROUP_CASE(96,8,1)
    // About to drop down to 1 CTA per SM due to shared memory
    default:
      printf("Adding group case to outer flux0 computation!\n");
      assert(false);
  }
#undef GROUP_CASE
}

template<int GROUPS, int STRIP_SIZE>
__global__
void gpu_fluxm_outer_source(const PointerBuffer<GROUPS,MomentTriple> fluxm_ptrs,
                            const PointerBuffer<GROUPS,MomentQuad> slgg_ptrs,
                            const int           *mat_ptr,
                                  PointerBuffer<GROUPS,MomentTriple> qom_ptrs,
                            ByteOffsetArray<3> fluxm_offsets,
                            ByteOffsetArray<2> slgg_offsets,
                            ByteOffsetArray<3> mat_offsets,
                            ByteOffsetArray<3> qom_offsets,
                            const int           num_moments,
                            const ConstBuffer<4,int> lma)
{
  __shared__ double fluxm_buffer_0[GROUPS][STRIP_SIZE];
  __shared__ double fluxm_buffer_1[GROUPS][STRIP_SIZE];
  __shared__ double fluxm_buffer_2[GROUPS][STRIP_SIZE];
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y + blockDim.y + threadIdx.y;
  const int z = blockIdx.z;
  const int group = threadIdx.z;
  const int strip_offset = threadIdx.y * blockDim.x + threadIdx.x;
  const MomentTriple *fluxm_ptr = fluxm_ptrs[group] + x * fluxm_offsets[0] +
    y * fluxm_offsets[1] + z * fluxm_offsets[2];
  mat_ptr += x * mat_offsets[0] + y * mat_offsets[1] + z * mat_offsets[2];
  MomentTriple *qom_ptr = qom_ptrs[group] + x * qom_offsets[0] + 
    y *qom_offsets[1] + z * qom_offsets[2];
  MomentTriple fluxm;
  asm volatile("ld.global.cs.v2.f64 {%0,%1}, [%2];" : "=d"(fluxm[0]), "=d"(fluxm[1]) 
                : "l"(fluxm_ptr) : "memory");
  asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(fluxm[2]) 
                : "l"(((char*)fluxm_ptr)+16) : "memory");
  int mat;
  asm volatile("ld.global.ca.s32 %0, [%1];" : "=r"(mat) : "l"(mat_ptr) : "memory");
  // Write the fluxm into shared memory
  fluxm_buffer_0[group][strip_offset] = fluxm[0];
  fluxm_buffer_1[group][strip_offset] = fluxm[1];
  fluxm_buffer_2[group][strip_offset] = fluxm[2];
  // Can compute our slgg_ptr with the matrix result
  const MomentQuad *slgg_ptr = slgg_ptrs[group] + mat * slgg_offsets[0];
  // Synchronize to make sure all the writes to shared are done 
  __syncthreads();
  // Do the math
  MomentTriple qom;
  #pragma unroll
  for (int g = 0; g < GROUPS; g++) {
    if (g == group)
      continue;
    int moment = 0;
    const MomentQuad *local_slgg = slgg_ptr + g * slgg_offsets[1];
    MomentQuad scat;
    asm volatile("ld.global.ca.v2.f64 {%0,%1}, [%2];" : "=d"(scat[0]), "=d"(scat[1])
                  : "l"(local_slgg) : "memory");
    asm volatile("ld.global.ca.v2.f64 {%0,%1}, [%2];" : "=d"(scat[2]), "=d"(scat[3])
                  : "l"(((char*)local_slgg)+16) : "memory");
    MomentTriple csm;
    for (int l = 1; l < num_moments; l++) {
      for (int j = 0; j < lma[l]; j++)
        csm[moment+j] = scat[l];
      moment += lma[l];
    }
    fluxm[0] = fluxm_buffer_0[g][strip_offset];
    fluxm[1] = fluxm_buffer_1[g][strip_offset];
    fluxm[2] = fluxm_buffer_2[g][strip_offset];
    for (int l = 0; l < (num_moments-1); l++)
      qom[l] += csm[l] * fluxm[l];
  }
  // Now we can write out the result
  asm volatile("st.global.cs.v2.f64 [%0], {%1,%2};" : : "l"(qom_ptr), 
                "d"(qom[0]), "d"(qom[1]) : "memory");
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(((char*)qom_ptr)+16),
                "d"(qom[2]) : "memory");
}

template<int GROUPS, int MAX_X, int MAX_Y>
__host__
void fluxm_launch_helper(Rect<3> subgrid_bounds,
                         const std::vector<MomentTriple*> &fluxm_ptrs,
                         const std::vector<MomentQuad*> &slgg_ptrs,
                         const std::vector<MomentTriple*> &qom_ptrs, 
                         const int *mat_ptr, 
                         const ByteOffset fluxm_offsets[3], 
                         const ByteOffset slgg_offsets[2],
                         const ByteOffset mat_offsets[3], 
                         const ByteOffset qom_offsets[3],
                         const int num_groups, const int num_moments, const int lma[4])
{
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1;
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;
  dim3 block(gcd(x_range,MAX_X), gcd(y_range,MAX_Y), GROUPS);
  dim3 grid(x_range/block.x, y_range/block.y, z_range);
  gpu_fluxm_outer_source<GROUPS,MAX_X*MAX_Y><<<grid,block>>>(
                            PointerBuffer<GROUPS,MomentTriple>(fluxm_ptrs), 
                            PointerBuffer<GROUPS,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<GROUPS,MomentTriple>(qom_ptrs),
                            ByteOffsetArray<3>(fluxm_offsets),
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(qom_offsets),
                            num_moments, ConstBuffer<4,int>(lma));
}

__host__
void run_fluxm_outer_source(Rect<3> subgrid_bounds,
                            const std::vector<MomentTriple*> &fluxm_ptrs,
                            const std::vector<MomentQuad*> &slgg_ptrs,
                            const std::vector<MomentTriple*> &qom_ptrs, 
                            const int *mat_ptr, 
                            const ByteOffset fluxm_offsets[3], 
                            const ByteOffset slgg_offsets[2],
                            const ByteOffset mat_offsets[3], 
                            const ByteOffset qom_offsets[3],
                            const int num_groups, const int num_moments, const int lma[4])
{
  // TODO: replace this template madness with Terra
#define GROUP_CASE(g,x,y)                                                         \
  case g:                                                                         \
    {                                                                             \
      fluxm_launch_helper<g,x,y>(subgrid_bounds, fluxm_ptrs, slgg_ptrs, qom_ptrs, \
                             mat_ptr, fluxm_offsets, slgg_offsets, mat_offsets,   \
                             qom_offsets, num_groups, num_moments, lma);          \
      break;                                                                      \
    }
  switch (num_groups)
  {
    GROUP_CASE(1,32,32)
    GROUP_CASE(2,32,16)
    GROUP_CASE(3,32,8)
    GROUP_CASE(4,32,8)
    GROUP_CASE(5,32,4)
    GROUP_CASE(6,32,4)
    GROUP_CASE(7,32,4)
    GROUP_CASE(8,32,4)
    GROUP_CASE(9,32,2)
    GROUP_CASE(10,32,2)
    GROUP_CASE(11,32,2)
    GROUP_CASE(12,32,2)
    GROUP_CASE(13,32,2)
    GROUP_CASE(14,32,2)
    GROUP_CASE(15,32,2)
    GROUP_CASE(16,32,2)
    GROUP_CASE(17,16,2)
    GROUP_CASE(18,16,2)
    GROUP_CASE(19,16,2)
    GROUP_CASE(20,16,2)
    GROUP_CASE(21,16,2)
    GROUP_CASE(22,16,2)
    GROUP_CASE(23,16,2)
    GROUP_CASE(24,16,2)
    GROUP_CASE(25,16,2)
    GROUP_CASE(26,16,2)
    GROUP_CASE(27,16,2)
    GROUP_CASE(28,16,2)
    GROUP_CASE(29,16,2)
    GROUP_CASE(30,16,2)
    GROUP_CASE(31,16,2)
    GROUP_CASE(32,16,2)
    GROUP_CASE(33,16,1)
    GROUP_CASE(34,16,1)
    GROUP_CASE(35,16,1)
    GROUP_CASE(36,16,1)
    GROUP_CASE(37,16,1)
    GROUP_CASE(38,16,1)
    GROUP_CASE(39,16,1)
    GROUP_CASE(40,16,1)
    GROUP_CASE(41,16,1)
    GROUP_CASE(42,16,1)
    GROUP_CASE(43,16,1)
    GROUP_CASE(44,16,1)
    GROUP_CASE(45,16,1)
    GROUP_CASE(46,16,1)
    GROUP_CASE(47,16,1)
    GROUP_CASE(48,16,1)
    GROUP_CASE(49,16,1)
    GROUP_CASE(50,16,1)
    GROUP_CASE(51,16,1)
    GROUP_CASE(52,16,1)
    GROUP_CASE(53,16,1)
    GROUP_CASE(54,16,1)
    GROUP_CASE(55,16,1)
    GROUP_CASE(56,16,1)
    GROUP_CASE(57,16,1)
    GROUP_CASE(58,16,1)
    GROUP_CASE(59,16,1)
    GROUP_CASE(60,16,1)
    GROUP_CASE(61,16,1)
    GROUP_CASE(62,16,1)
    GROUP_CASE(63,16,1)
    GROUP_CASE(64,16,1)
    GROUP_CASE(65,8,1)
    GROUP_CASE(66,8,1)
    GROUP_CASE(67,8,1)
    GROUP_CASE(68,8,1)
    GROUP_CASE(69,8,1)
    GROUP_CASE(70,8,1)
    GROUP_CASE(71,8,1)
    GROUP_CASE(72,8,1)
    GROUP_CASE(73,8,1)
    GROUP_CASE(74,8,1)
    GROUP_CASE(75,8,1)
    GROUP_CASE(76,8,1)
    GROUP_CASE(77,8,1)
    GROUP_CASE(78,8,1)
    GROUP_CASE(79,8,1)
    GROUP_CASE(80,8,1)
    GROUP_CASE(81,8,1)
    GROUP_CASE(82,8,1)
    GROUP_CASE(83,8,1)
    GROUP_CASE(84,8,1)
    GROUP_CASE(85,8,1)
    GROUP_CASE(86,8,1)
    GROUP_CASE(87,8,1)
    GROUP_CASE(88,8,1)
    GROUP_CASE(89,8,1)
    GROUP_CASE(90,8,1)
    GROUP_CASE(91,8,1)
    GROUP_CASE(92,8,1)
    GROUP_CASE(93,8,1)
    GROUP_CASE(94,8,1)
    GROUP_CASE(95,8,1)
    GROUP_CASE(96,8,1)
    default:
      printf("Adding group case to outer fluxm computation!\n");
      assert(false);
  }
#undef GROUP_CASE
}

__global__
void gpu_outer_convergence(const double *flux0_ptr, const double *flux0po_ptr,
                           ByteOffsetArray<3> flux0_offsets,
                           ByteOffsetArray<3> flux0po_offsets,
                           const double epsi, int *total_converged)
{
  // We know there is never more than 32 warps in a CTA
  __shared__ int trampoline[32];

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  flux0_ptr += x * flux0_offsets[0] + y * flux0_offsets[1] + z * flux0_offsets[2];
  flux0po_ptr += x * flux0po_offsets[0] + y * flux0po_offsets[1] + z * flux0po_offsets[2];

  const double tolr = 1.0e-12;
  
  double flux0po = *flux0po_ptr;
  double df = 1.0;
  if (fabs(flux0po) < tolr) {
    flux0po = 1.0;
    df = 0.0;
  }
  double flux0 = *flux0_ptr;
  df = fabs( (flux0 / flux0po) - df );
  int local_converged = 1;
  if ((df >= -INFINITY) && (df > epsi))
    local_converged = 0;
  // Perform a local reduction inside the CTA
  // Butterfly reduction across all threads in all warps
  unsigned laneid;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(laneid) : );
  const unsigned warpid = 
    ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) >> 5;
  for (int i = 16; i >= 1; i/=2)
    local_converged += __shfl_xor(local_converged, i, 32);
  // Initialize the trampoline
  if (warpid == 0)
    trampoline[laneid] = 0;
  __syncthreads();
  // First thread in each warp writes out all values
  if (laneid == 0)
    trampoline[warpid] = local_converged;
  __syncthreads();
  // Butterfly reduction across all thread in the first warp
  if (warpid == 0) {
    local_converged = trampoline[laneid];
    for (int i = 16; i >= 1; i/=2)
      local_converged += __shfl_xor(local_converged, i, 32);
    // First thread does the atomic
    if (laneid == 0)
      atomicAdd(total_converged, local_converged);
  }
}

__host__
bool run_outer_convergence(Rect<3> subgrid_bounds,
                           const std::vector<double*> flux0_ptrs,
                           const std::vector<double*> flux0po_ptrs,
                           const ByteOffset flux0_offsets[3], 
                           const ByteOffset flux0po_offsets[3],
                           const double epsi)
{
  int *converged_d;
  cudaMalloc((void**)&converged_d, sizeof(int));
  // Initialize the result
  cudaMemset(converged_d, 0/*value*/, 1/*count*/); 
  // Launch the kernels
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1;
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;
  dim3 block(gcd(x_range,32),gcd(y_range,4),gcd(z_range,4));
  dim3 grid(x_range/block.x, y_range/block.y, z_range/block.z);

  assert(flux0_ptrs.size() == flux0po_ptrs.size());
  for (unsigned idx = 0; idx < flux0_ptrs.size(); idx++) {
    gpu_outer_convergence<<<grid,block>>>(flux0_ptrs[idx], flux0po_ptrs[idx],
                                          ByteOffsetArray<3>(flux0_offsets),
                                          ByteOffsetArray<3>(flux0po_offsets),
                                          epsi, converged_d); 
  }
  // Copy back: CUDA hijack synchronizes for us
  int converged_h;
  cudaMemcpy(&converged_h, converged_d, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(converged_d);
  // We've converged if the total converged points are the number of tests
  return (converged_h == int(x_range * y_range * z_range * flux0_ptrs.size()));
}


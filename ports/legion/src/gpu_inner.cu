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

using namespace LegionRuntime::Arrays;
using namespace LegionRuntime::Accessor;

__global__
void gpu_inner_source_single_moment(const MomentQuad  *sxs_ptr,
                                    const double      *flux0_ptr,
                                    const double      *q2grp0_ptr,
                                          MomentQuad  *qtot_ptr,
                                    ByteOffsetArray<3> sxs_offsets,
                                    ByteOffsetArray<3> flux0_offsets,
                                    ByteOffsetArray<3> q2grp0_offsets,
                                    ByteOffsetArray<3> qtot_offsets)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  // Straight up data parallel so nothing interesting to do
  sxs_ptr += x * sxs_offsets[0] + y * sxs_offsets[1] + z * sxs_offsets[2];
  MomentQuad sxs_quad = *sxs_ptr;

  flux0_ptr += x * flux0_offsets[0] + y * flux0_offsets[1] + z * flux0_offsets[2];
  double flux0 = *flux0_ptr;

  q2grp0_ptr += x * q2grp0_offsets[0] + y * q2grp0_offsets[1] + z * q2grp0_offsets[2];
  double q0 = *q2grp0_ptr;

  MomentQuad quad;
  quad[0] = q0 + flux0 * sxs_quad[0]; 

  qtot_ptr += x * qtot_offsets[0] + y * qtot_offsets[1] + z * qtot_offsets[2];
  *qtot_ptr = quad;
}

__host__
void run_inner_source_single_moment(Rect<3>           subgrid_bounds,
                                    const MomentQuad  *sxs_ptr,
                                    const double      *flux0_ptr,
                                    const double      *q2grp0_ptr,
                                          MomentQuad  *qtot_ptr,
                                    const ByteOffset  sxs_offsets[3],
                                    const ByteOffset  flux0_offsets[3],
                                    const ByteOffset  q2grp0_offsets[3],
                                    const ByteOffset  qtot_offsets[3])
{
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1;
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;

  dim3 block(gcd(x_range,32),gcd(y_range,4),gcd(z_range,4));
  dim3 grid(x_range/block.x, y_range/block.y, z_range/block.z);
  gpu_inner_source_single_moment<<<grid,block>>>(sxs_ptr, flux0_ptr,
                                                 q2grp0_ptr, qtot_ptr, 
                                                 ByteOffsetArray<3>(sxs_offsets),
                                                 ByteOffsetArray<3>(flux0_offsets),
                                                 ByteOffsetArray<3>(q2grp0_offsets),
                                                 ByteOffsetArray<3>(qtot_offsets));
}

__global__
void gpu_inner_source_multi_moment(const MomentQuad   *sxs_ptr,
                                   const double       *flux0_ptr,
                                   const double       *q2grp0_ptr,
                                   const MomentTriple *fluxm_ptr,
                                   const MomentTriple *q2grpm_ptr,
                                         MomentQuad   *qtot_ptr,
                                   ByteOffsetArray<3> sxs_offsets,
                                   ByteOffsetArray<3> flux0_offsets,
                                   ByteOffsetArray<3> q2grp0_offsets,
                                   ByteOffsetArray<3> fluxm_offsets,
                                   ByteOffsetArray<3> q2grpm_offsets,
                                   ByteOffsetArray<3> qtot_offsets,
                                   const int num_moments, 
                                   const ConstBuffer<4,int> lma)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  // Straight up data parallel so nothing interesting to do
  sxs_ptr += x * sxs_offsets[0] + y * sxs_offsets[1] + z * sxs_offsets[2];
  MomentQuad sxs_quad = *sxs_ptr;

  flux0_ptr += x * flux0_offsets[0] + y * flux0_offsets[1] + z * flux0_offsets[2];
  double flux0 = *flux0_ptr;

  q2grp0_ptr += x * q2grp0_offsets[0] + y * q2grp0_offsets[1] + z * q2grp0_offsets[2];
  double q0 = *q2grp0_ptr;

  fluxm_ptr += x * fluxm_offsets[0] + y * fluxm_offsets[1] + z * fluxm_offsets[2];
  MomentTriple fluxm = *fluxm_ptr;

  q2grpm_ptr += x * q2grpm_offsets[0] + y * q2grpm_offsets[1] + z * q2grpm_offsets[2];
  MomentTriple qom = *q2grpm_ptr;

  MomentQuad quad;
  quad[0] = q0 + flux0 * sxs_quad[0]; 
  
  int moment = 0;
  for (int l = 1; l < num_moments; l++) {
    for (int i = 0; i < lma[l]; i++)
      quad[moment+i+1] = qom[moment+i] + fluxm[moment+i] * sxs_quad[l];
    moment += lma[l];
  }

  qtot_ptr += x * qtot_offsets[0] + y * qtot_offsets[1] + z * qtot_offsets[2];
  *qtot_ptr = quad;
}

__host__
void run_inner_source_multi_moment(Rect<3> subgrid_bounds,
                                   const MomentQuad   *sxs_ptr,
                                   const double       *flux0_ptr,
                                   const double       *q2grp0_ptr,
                                   const MomentTriple *fluxm_ptr,
                                   const MomentTriple *q2grpm_ptr,
                                         MomentQuad   *qtot_ptr,
                                   const ByteOffset   sxs_offsets[3],
                                   const ByteOffset   flux0_offsets[3],
                                   const ByteOffset   q2grp0_offsets[3],
                                   const ByteOffset   fluxm_offsets[3],
                                   const ByteOffset   q2grpm_offsets[3],
                                   const ByteOffset   qtot_offsets[3],
                                   const int num_moments, const int lma[4])
{
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1;
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;

  dim3 block(gcd(x_range,32),gcd(y_range,4),gcd(z_range,4));
  dim3 grid(x_range/block.x, y_range/block.y, z_range/block.z);
  gpu_inner_source_multi_moment<<<grid,block>>>(sxs_ptr, flux0_ptr, q2grp0_ptr, 
                                                fluxm_ptr, q2grpm_ptr, qtot_ptr, 
                                                ByteOffsetArray<3>(sxs_offsets),
                                                ByteOffsetArray<3>(flux0_offsets),
                                                ByteOffsetArray<3>(q2grp0_offsets),
                                                ByteOffsetArray<3>(fluxm_offsets),
                                                ByteOffsetArray<3>(q2grpm_offsets),
                                                ByteOffsetArray<3>(qtot_offsets),
                                                num_moments, ConstBuffer<4,int>(lma));
}

__global__
void gpu_inner_convergence(const double *flux0_ptr, const double *flux0pi_ptr,
                           ByteOffsetArray<3> flux0_offsets,
                           ByteOffsetArray<3> flux0pi_offsets,
                           const double epsi, int *total_converged)
{
  // We know there is never more than 32 warps in a CTA
  __shared__ int trampoline[32];

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  flux0_ptr += x * flux0_offsets[0] + y * flux0_offsets[1] + z * flux0_offsets[2];
  flux0pi_ptr += x * flux0pi_offsets[0] + y * flux0pi_offsets[1] + z * flux0pi_offsets[2];

  const double tolr = 1.0e-12;

  double flux0pi = *flux0pi_ptr;
  double df = 1.0;
  if (fabs(flux0pi) < tolr) {
    flux0pi = 1.0;
    df = 0.0;
  }
  double flux0 = *flux0_ptr;
  df = fabs( (flux0 / flux0pi) - df );
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
bool run_inner_convergence(Rect<3> subgrid_bounds,
                           const std::vector<double*> flux0_ptrs,
                           const std::vector<double*> flux0pi_ptrs,
                           const ByteOffset flux0_offsets[3], 
                           const ByteOffset flux0pi_offsets[3],
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

  assert(flux0_ptrs.size() == flux0pi_ptrs.size());
  for (unsigned idx = 0; idx < flux0_ptrs.size(); idx++) {
    gpu_inner_convergence<<<grid,block>>>(flux0_ptrs[idx], flux0pi_ptrs[idx],
                                          ByteOffsetArray<3>(flux0_offsets),
                                          ByteOffsetArray<3>(flux0pi_offsets),
                                          epsi, converged_d); 
  }
  // Copy back: CUDA hijack synchronizes for us
  int converged_h;
  cudaMemcpy(&converged_h, converged_d, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(converged_d);
  // We've converged if the total converged points are the number of tests
  return (converged_h == int(x_range * y_range * z_range * flux0_ptrs.size()));
}


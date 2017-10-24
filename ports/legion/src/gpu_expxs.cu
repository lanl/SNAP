/* Copyright 2017 NVIDIA Corporation
 *
 * The U.S. Department of Energy funded the development of this software 
 * under subcontract B609478 with Lawrence Livermore National Security, LLC
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

template<int GROUPS>
__global__
void gpu_expand_cross_section(const PointerBuffer<GROUPS,double> sig_ptrs,
                              const int    *mat_ptr,
                                    PointerBuffer<GROUPS,double> xs_ptrs,
                              const ByteOffsetArray<1> sig_offsets,
                              const ByteOffsetArray<3> mat_offsets,
                              const ByteOffsetArray<3> xs_offsets)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  const int mat = *(mat_ptr + x * mat_offsets[0] + 
                              y * mat_offsets[1] + 
                              z * mat_offsets[2]) - 1/*IS starts at 1*/;
  #pragma unroll
  for (int g = 0; g < GROUPS; g++) {
    const double *sig_ptr = sig_ptrs[g] + mat * sig_offsets[0];
    double val;
    asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(val) : "l"(sig_ptr) : "memory");
    double *xs_ptr = xs_ptrs[g] + x * xs_offsets[0] +
                                  y * xs_offsets[1] + z * xs_offsets[2];
    asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(xs_ptr), "d"(val) : "memory");
  }
}

__host__
void run_expand_cross_section(const std::vector<double*> &sig_ptrs,
                              const int *mat_ptr,
                              const std::vector<double*> &xs_ptrs,
                              const ByteOffset sig_offsets[1],
                              const ByteOffset mat_offsets[3],
                              const ByteOffset xs_offsets[3],
                              const Rect<3> &subgrid_bounds)
{
  // Figure out the dimensions to launch
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1; 
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;

  dim3 block(gcd(x_range,32), gcd(y_range,4), gcd(z_range,4));
  dim3 grid(x_range/block.x, y_range/block.y, z_range/block.z);

  // Switch on the number of groups
  assert(sig_ptrs.size() == xs_ptrs.size());
  // TODO: replace this template foolishness with Terra
  switch (sig_ptrs.size())
  {
    case 1:
      {
        gpu_expand_cross_section<1><<<grid, block>>>(
                                       PointerBuffer<1,double>(sig_ptrs), mat_ptr,
                                       PointerBuffer<1,double>(xs_ptrs), 
                                       ByteOffsetArray<1>(sig_offsets),
                                       ByteOffsetArray<3>(mat_offsets),
                                       ByteOffsetArray<3>(xs_offsets));
        break;
      }
    case 2:
      {
        gpu_expand_cross_section<2><<<grid, block>>>(
                                       PointerBuffer<2,double>(sig_ptrs), mat_ptr,
                                       PointerBuffer<2,double>(xs_ptrs), 
                                       ByteOffsetArray<1>(sig_offsets),
                                       ByteOffsetArray<3>(mat_offsets),
                                       ByteOffsetArray<3>(xs_offsets));
        break;
      }
    case 3:
      {
        gpu_expand_cross_section<3><<<grid, block>>>(
                                       PointerBuffer<3,double>(sig_ptrs), mat_ptr,
                                       PointerBuffer<3,double>(xs_ptrs), 
                                       ByteOffsetArray<1>(sig_offsets),
                                       ByteOffsetArray<3>(mat_offsets),
                                       ByteOffsetArray<3>(xs_offsets));
        break;
      }
    case 4:
      {
        gpu_expand_cross_section<4><<<grid, block>>>(
                                       PointerBuffer<4,double>(sig_ptrs), mat_ptr,
                                       PointerBuffer<4,double>(xs_ptrs), 
                                       ByteOffsetArray<1>(sig_offsets),
                                       ByteOffsetArray<3>(mat_offsets),
                                       ByteOffsetArray<3>(xs_offsets));
        break;
      }
    case 5:
      {
        gpu_expand_cross_section<5><<<grid, block>>>(
                                       PointerBuffer<5,double>(sig_ptrs), mat_ptr,
                                       PointerBuffer<5,double>(xs_ptrs), 
                                       ByteOffsetArray<1>(sig_offsets),
                                       ByteOffsetArray<3>(mat_offsets),
                                       ByteOffsetArray<3>(xs_offsets));
        break;
      }
    case 6:
      {
        gpu_expand_cross_section<6><<<grid, block>>>(
                                       PointerBuffer<6,double>(sig_ptrs), mat_ptr,
                                       PointerBuffer<6,double>(xs_ptrs), 
                                       ByteOffsetArray<1>(sig_offsets),
                                       ByteOffsetArray<3>(mat_offsets),
                                       ByteOffsetArray<3>(xs_offsets));
        break;
      }
    case 7:
      {
        gpu_expand_cross_section<7><<<grid, block>>>(
                                       PointerBuffer<7,double>(sig_ptrs), mat_ptr,
                                       PointerBuffer<7,double>(xs_ptrs), 
                                       ByteOffsetArray<1>(sig_offsets),
                                       ByteOffsetArray<3>(mat_offsets),
                                       ByteOffsetArray<3>(xs_offsets));
        break;
      }
    case 8:
      {
        gpu_expand_cross_section<8><<<grid, block>>>(
                                       PointerBuffer<8,double>(sig_ptrs), mat_ptr,
                                       PointerBuffer<8,double>(xs_ptrs), 
                                       ByteOffsetArray<1>(sig_offsets),
                                       ByteOffsetArray<3>(mat_offsets),
                                       ByteOffsetArray<3>(xs_offsets));
        break;
      }
    case 9:
      {
        gpu_expand_cross_section<9><<<grid, block>>>(
                                       PointerBuffer<9,double>(sig_ptrs), mat_ptr,
                                       PointerBuffer<9,double>(xs_ptrs), 
                                       ByteOffsetArray<1>(sig_offsets),
                                       ByteOffsetArray<3>(mat_offsets),
                                       ByteOffsetArray<3>(xs_offsets));
        break;
      }
    case 10:
      {
        gpu_expand_cross_section<10><<<grid, block>>>(
                                       PointerBuffer<10,double>(sig_ptrs), mat_ptr,
                                       PointerBuffer<10,double>(xs_ptrs), 
                                       ByteOffsetArray<1>(sig_offsets),
                                       ByteOffsetArray<3>(mat_offsets),
                                       ByteOffsetArray<3>(xs_offsets));
        break;
      }
    case 11:
      {
        gpu_expand_cross_section<11><<<grid, block>>>(
                                       PointerBuffer<11,double>(sig_ptrs), mat_ptr,
                                       PointerBuffer<11,double>(xs_ptrs), 
                                       ByteOffsetArray<1>(sig_offsets),
                                       ByteOffsetArray<3>(mat_offsets),
                                       ByteOffsetArray<3>(xs_offsets));
        break;
      }
    case 12:
      {
        gpu_expand_cross_section<12><<<grid, block>>>(
                                       PointerBuffer<12,double>(sig_ptrs), mat_ptr,
                                       PointerBuffer<12,double>(xs_ptrs), 
                                       ByteOffsetArray<1>(sig_offsets),
                                       ByteOffsetArray<3>(mat_offsets),
                                       ByteOffsetArray<3>(xs_offsets));
        break;
      }
    case 13:
      {
        gpu_expand_cross_section<13><<<grid, block>>>(
                                       PointerBuffer<13,double>(sig_ptrs), mat_ptr,
                                       PointerBuffer<13,double>(xs_ptrs), 
                                       ByteOffsetArray<1>(sig_offsets),
                                       ByteOffsetArray<3>(mat_offsets),
                                       ByteOffsetArray<3>(xs_offsets));
        break;
      }
    case 14:
      {
        gpu_expand_cross_section<14><<<grid, block>>>(
                                       PointerBuffer<14,double>(sig_ptrs), mat_ptr,
                                       PointerBuffer<14,double>(xs_ptrs), 
                                       ByteOffsetArray<1>(sig_offsets),
                                       ByteOffsetArray<3>(mat_offsets),
                                       ByteOffsetArray<3>(xs_offsets));
        break;
      }
    case 15:
      {
        gpu_expand_cross_section<15><<<grid, block>>>(
                                       PointerBuffer<15,double>(sig_ptrs), mat_ptr,
                                       PointerBuffer<15,double>(xs_ptrs), 
                                       ByteOffsetArray<1>(sig_offsets),
                                       ByteOffsetArray<3>(mat_offsets),
                                       ByteOffsetArray<3>(xs_offsets));
        break;
      }
    case 16:
      {
        gpu_expand_cross_section<16><<<grid, block>>>(
                                       PointerBuffer<16,double>(sig_ptrs), mat_ptr,
                                       PointerBuffer<16,double>(xs_ptrs), 
                                       ByteOffsetArray<1>(sig_offsets),
                                       ByteOffsetArray<3>(mat_offsets),
                                       ByteOffsetArray<3>(xs_offsets));
        break;
      }
    case 24:
      {
        gpu_expand_cross_section<24><<<grid, block>>>(
                                       PointerBuffer<24,double>(sig_ptrs), mat_ptr,
                                       PointerBuffer<24,double>(xs_ptrs), 
                                       ByteOffsetArray<1>(sig_offsets),
                                       ByteOffsetArray<3>(mat_offsets),
                                       ByteOffsetArray<3>(xs_offsets));
        break;
      }
    case 32:
      {
        gpu_expand_cross_section<32><<<grid, block>>>(
                                       PointerBuffer<32,double>(sig_ptrs), mat_ptr,
                                       PointerBuffer<32,double>(xs_ptrs), 
                                       ByteOffsetArray<1>(sig_offsets),
                                       ByteOffsetArray<3>(mat_offsets),
                                       ByteOffsetArray<3>(xs_offsets));
        break;
      }
    case 40:
      {
        gpu_expand_cross_section<40><<<grid, block>>>(
                                       PointerBuffer<40,double>(sig_ptrs), mat_ptr,
                                       PointerBuffer<40,double>(xs_ptrs), 
                                       ByteOffsetArray<1>(sig_offsets),
                                       ByteOffsetArray<3>(mat_offsets),
                                       ByteOffsetArray<3>(xs_offsets));
        break;
      }
    case 48:
      {
        gpu_expand_cross_section<48><<<grid, block>>>(
                                       PointerBuffer<48,double>(sig_ptrs), mat_ptr,
                                       PointerBuffer<48,double>(xs_ptrs), 
                                       ByteOffsetArray<1>(sig_offsets),
                                       ByteOffsetArray<3>(mat_offsets),
                                       ByteOffsetArray<3>(xs_offsets));
        break;
      }
    case 56:
      {
        gpu_expand_cross_section<56><<<grid, block>>>(
                                       PointerBuffer<56,double>(sig_ptrs), mat_ptr,
                                       PointerBuffer<56,double>(xs_ptrs), 
                                       ByteOffsetArray<1>(sig_offsets),
                                       ByteOffsetArray<3>(mat_offsets),
                                       ByteOffsetArray<3>(xs_offsets));
        break;
      }
    case 64:
      {
        gpu_expand_cross_section<64><<<grid, block>>>(
                                       PointerBuffer<64,double>(sig_ptrs), mat_ptr,
                                       PointerBuffer<64,double>(xs_ptrs), 
                                       ByteOffsetArray<1>(sig_offsets),
                                       ByteOffsetArray<3>(mat_offsets),
                                       ByteOffsetArray<3>(xs_offsets));
        break;
      }
    default:
      assert(false); // add more cases
  }
}

template<int GROUPS>
__global__
void gpu_expand_scattering_cross_section(const PointerBuffer<GROUPS,MomentQuad> slgg_ptrs,
                                         const int        *mat_ptr,
                                               PointerBuffer<GROUPS,MomentQuad> xs_ptrs,
                                         const ByteOffsetArray<2> slgg_offsets,
                                         const ByteOffsetArray<3> mat_offsets,
                                         const ByteOffsetArray<3> xs_offsets,
                                         const int group_start)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  const int mat = *(mat_ptr + x * mat_offsets[0] + 
                              y * mat_offsets[1] + 
                              z * mat_offsets[2]) - 1/*IS starts at 1*/;
  #pragma unroll
  for (int g = 0; g < GROUPS; g++) {
    MomentQuad quad = *(slgg_ptrs[g] + mat * slgg_offsets[0] +
                        (group_start + g) * slgg_offsets[1]);
    *(xs_ptrs[g] + x * xs_offsets[0] + y * xs_offsets[1] +
        z * xs_offsets[2]) = quad;
  }
}

__host__
void run_expand_scattering_cross_section(
                                      const std::vector<MomentQuad*> &slgg_ptrs,
                                      const int *mat_ptr,
                                      const std::vector<MomentQuad*> &xs_ptrs,
                                      const ByteOffset slgg_offsets[2],
                                      const ByteOffset mat_offsets[3],
                                      const ByteOffset xs_offsets[3],
                                      const Rect<3> &subgrid_bounds,
                                      const int group_start)
{
  // Figure out the dimensions to launch
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1; 
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;

  dim3 block(gcd(x_range,32),gcd(y_range,4),gcd(z_range,4));
  dim3 grid(x_range/block.x, y_range/block.y, z_range/block.z);

  // Switch on the number of groups
  assert(slgg_ptrs.size() == xs_ptrs.size());
  // TODO: replace this template foolishness with Terra
  switch (slgg_ptrs.size())
  {
    case 1:
      {
        gpu_expand_scattering_cross_section<1><<<grid,block>>>(
                            PointerBuffer<1,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<1,MomentQuad>(xs_ptrs), 
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(xs_offsets),
                            group_start);
        break;
      }
    case 2:
      {
        gpu_expand_scattering_cross_section<2><<<grid,block>>>(
                            PointerBuffer<2,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<2,MomentQuad>(xs_ptrs), 
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(xs_offsets),
                            group_start);
        break;
      }
    case 3:
      {
        gpu_expand_scattering_cross_section<3><<<grid,block>>>(
                            PointerBuffer<3,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<3,MomentQuad>(xs_ptrs), 
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(xs_offsets),
                            group_start);
        break;
      }
    case 4:
      {
        gpu_expand_scattering_cross_section<4><<<grid,block>>>(
                            PointerBuffer<4,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<4,MomentQuad>(xs_ptrs), 
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(xs_offsets),
                            group_start);
        break;
      }
    case 5:
      {
        gpu_expand_scattering_cross_section<5><<<grid,block>>>(
                            PointerBuffer<5,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<5,MomentQuad>(xs_ptrs), 
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(xs_offsets),
                            group_start);
        break;
      }
    case 6:
      {
        gpu_expand_scattering_cross_section<6><<<grid,block>>>(
                            PointerBuffer<6,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<6,MomentQuad>(xs_ptrs), 
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(xs_offsets),
                            group_start);
        break;
      }
    case 7:
      {
        gpu_expand_scattering_cross_section<7><<<grid,block>>>(
                            PointerBuffer<7,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<7,MomentQuad>(xs_ptrs), 
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(xs_offsets),
                            group_start);
        break;
      }
    case 8:
      {
        gpu_expand_scattering_cross_section<8><<<grid,block>>>(
                            PointerBuffer<8,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<8,MomentQuad>(xs_ptrs), 
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(xs_offsets),
                            group_start);
        break;
      }
    case 9:
      {
        gpu_expand_scattering_cross_section<9><<<grid,block>>>(
                            PointerBuffer<9,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<9,MomentQuad>(xs_ptrs), 
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(xs_offsets),
                            group_start);
        break;
      }
    case 10:
      {
        gpu_expand_scattering_cross_section<10><<<grid,block>>>(
                            PointerBuffer<10,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<10,MomentQuad>(xs_ptrs), 
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(xs_offsets),
                            group_start);
        break;
      }
    case 11:
      {
        gpu_expand_scattering_cross_section<11><<<grid,block>>>(
                            PointerBuffer<11,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<11,MomentQuad>(xs_ptrs), 
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(xs_offsets),
                            group_start);
        break;
      }
    case 12:
      {
        gpu_expand_scattering_cross_section<12><<<grid,block>>>(
                            PointerBuffer<12,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<12,MomentQuad>(xs_ptrs), 
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(xs_offsets),
                            group_start);
        break;
      }
    case 13:
      {
        gpu_expand_scattering_cross_section<13><<<grid,block>>>(
                            PointerBuffer<13,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<13,MomentQuad>(xs_ptrs), 
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(xs_offsets),
                            group_start);
        break;
      }
    case 14:
      {
        gpu_expand_scattering_cross_section<14><<<grid,block>>>(
                            PointerBuffer<14,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<14,MomentQuad>(xs_ptrs), 
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(xs_offsets),
                            group_start);
        break;
      }
    case 15:
      {
        gpu_expand_scattering_cross_section<15><<<grid,block>>>(
                            PointerBuffer<15,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<15,MomentQuad>(xs_ptrs), 
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(xs_offsets),
                            group_start);
        break;
      }
    case 16:
      {
        gpu_expand_scattering_cross_section<16><<<grid,block>>>(
                            PointerBuffer<16,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<16,MomentQuad>(xs_ptrs), 
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(xs_offsets),
                            group_start);
        break;
      }
    case 24:
      {
        gpu_expand_scattering_cross_section<24><<<grid,block>>>(
                            PointerBuffer<24,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<24,MomentQuad>(xs_ptrs), 
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(xs_offsets),
                            group_start);
        break;
      }
    case 32:
      {
        gpu_expand_scattering_cross_section<32><<<grid,block>>>(
                            PointerBuffer<32,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<32,MomentQuad>(xs_ptrs), 
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(xs_offsets),
                            group_start);
        break;
      }
    case 40:
      {
        gpu_expand_scattering_cross_section<40><<<grid,block>>>(
                            PointerBuffer<40,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<40,MomentQuad>(xs_ptrs), 
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(xs_offsets),
                            group_start);
        break;
      }
    case 48:
      {
        gpu_expand_scattering_cross_section<48><<<grid,block>>>(
                            PointerBuffer<48,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<48,MomentQuad>(xs_ptrs), 
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(xs_offsets),
                            group_start);
        break;
      }
    case 56:
      {
        gpu_expand_scattering_cross_section<56><<<grid,block>>>(
                            PointerBuffer<56,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<56,MomentQuad>(xs_ptrs), 
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(xs_offsets),
                            group_start);
        break;
      }
    case 64:
      {
        gpu_expand_scattering_cross_section<64><<<grid,block>>>(
                            PointerBuffer<64,MomentQuad>(slgg_ptrs), mat_ptr,
                            PointerBuffer<64,MomentQuad>(xs_ptrs), 
                            ByteOffsetArray<2>(slgg_offsets),
                            ByteOffsetArray<3>(mat_offsets),
                            ByteOffsetArray<3>(xs_offsets),
                            group_start);
        break;
      }
    default:
      assert(false); // add more cases
  }
}


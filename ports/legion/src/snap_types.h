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

#ifndef __SNAP_TYPES_H__
#define __SNAP_TYPES_H__

#ifdef __CUDACC__
#define CUDAPREFIX __host__ __device__
#else
#define CUDAPREFIX
#endif

// Put these here since we'll need them separately in CUDA code

struct MomentTriple {
public:
  CUDAPREFIX MomentTriple(double x = 0.0, double y = 0.0, double z = 0.0)
    { vals[0] = x; vals[1] = y; vals[2] = z; }
public:
  CUDAPREFIX double& operator[](const int index) { return vals[index]; }
  CUDAPREFIX const double& operator[](const int index) const { return vals[index]; }
public:
  double vals[3];
};

struct MomentQuad {
public:
  CUDAPREFIX MomentQuad(double x = 0.0, double y = 0.0, double z = 0.0, double w = 0.0)
    { vals[0] = x; vals[1] = y; vals[2] = z; vals[3] = w; }
public:
  CUDAPREFIX double& operator[](const int index) { return vals[index]; }
  CUDAPREFIX const double& operator[](const int index) const { return vals[index]; }
public:
  double vals[4];
};

#endif


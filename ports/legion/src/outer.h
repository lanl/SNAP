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

#ifndef __OUTER_H__
#define __OUTER_H__

#include "snap.h"
#include "legion.h"

using namespace Legion;

class CalcOuterSource : public SnapTask<CalcOuterSource, 
                                        Snap::CALC_OUTER_SOURCE_TASK_ID> {
public:
  CalcOuterSource(const Snap &snap, const Predicate &pred,
                  const SnapArray &qi, const SnapArray &slgg,
                  const SnapArray &mat, const SnapArray &q2rgp0, 
                  const SnapArray &q2grpm, const SnapArray &flux0,
                  const SnapArray &fluxm);
public:
  static void preregister_cpu_variants(void);
  static void preregister_gpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void fast_implementation(const Task *task, Context ctx, Runtime *runtime,
     const std::vector<double*> &qi0_ptrs, const ByteOffset qi0_offsets[3],
     const std::vector<double*> &flux0_ptrs, const ByteOffset flux0_offsets[3],
     const std::vector<MomentQuad*> &slgg_ptrs, const ByteOffset slgg_offsets[2],
     const std::vector<int*> &mat_ptrs, const ByteOffset mat_offsets[3],
     const std::vector<double*> &qo0_ptrs, const ByteOffset qo0_offsets[3],
     const std::vector<MomentTriple*> &fluxm_ptrs, const ByteOffset fluxm_offsets[3],
     const std::vector<MomentTriple*> &qom_ptrs, const ByteOffset qom_offsets[3]);
  static void gpu_implementation(const Task *task, Context ctx, Runtime *runtime,
     const std::vector<double*> &qi0_ptrs, const ByteOffset qi0_offsets[3],
     const std::vector<double*> &flux0_ptrs, const ByteOffset flux0_offsets[3],
     const std::vector<MomentQuad*> &slgg_ptrs, const ByteOffset slgg_offsets[2],
     const std::vector<int*> &mat_ptrs, const ByteOffset mat_offsets[3],
     const std::vector<double*> &qo0_ptrs, const ByteOffset qo0_offsets[3],
     const std::vector<MomentTriple*> &fluxm_ptrs, const ByteOffset fluxm_offsets[3],
     const std::vector<MomentTriple*> &qom_ptrs, const ByteOffset qom_offsets[3]);
};

class TestOuterConvergence : public SnapTask<TestOuterConvergence,
                                             Snap::TEST_OUTER_CONVERGENCE_TASK_ID> {
public:
  TestOuterConvergence(const Snap &snap, const Predicate &pred,
                       const SnapArray &flux0, const SnapArray &flux0po,
                       const Future &inner_converged, const Future &true_future,
                       int group_start, int group_stop);
public:
  static void preregister_cpu_variants(void);
  static void preregister_gpu_variants(void);
public:
  static bool cpu_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static bool fast_implementation(const Task *task, Context ctx, Runtime *runtime,
     const std::vector<double*> &flux0_ptrs, const ByteOffset flux0_offsets[3],
     const std::vector<double*> &flux0po_ptrs, const ByteOffset flux0po_offsets[3]);
  static bool gpu_implementation(const Task *task, Context ctx, Runtime *runtime,
     const std::vector<double*> &flux0_ptrs, const ByteOffset flux0_offsets[3],
     const std::vector<double*> &flux0po_ptrs, const ByteOffset flux0po_offsets[3]);
};

#endif // __OUTER_H__


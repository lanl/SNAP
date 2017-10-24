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

#ifndef __SNAP_CONVERGENCE_H__
#define __SNAP_CONVERGENCE_H__

#include "snap.h"
#include "legion.h"

using namespace Legion;

// This class will issue a chain of single task launches
// that when executed will emulate the execution of a 
// monad. We use them in SNAP to print out the convergence
// and timing information for individual iterations of the
// various loops of SNAP
class ConvergenceMonad {
public:
  // This is the type data actually stored in the Monad
  struct MonadData {
  public:
    long long step_start;
    long long outer_start;
    long long inner_start;
    int time_step_number;
    int inner_loop_number;
    int outer_loop_number;
  public:
    int total_inner_loops;
    int total_outer_loops;
    long long total_inner_time;
    long long total_outer_time;
    long long total_step_time;
  };
public:
  ConvergenceMonad(Context ctx, Runtime *runtime);
  ConvergenceMonad(const ConvergenceMonad &rhs);
  ~ConvergenceMonad(void);
public:
  ConvergenceMonad& operator=(const ConvergenceMonad &rhs);
public:
  void bind_inner(const Predicate &pred, const Future &inner_converged);
  void bind_outer(const Predicate &pred, const Future &outer_converged);
public:
  const Context ctx;
  Runtime *const runtime;
protected:
  Future monad_future;
public:
  static void preregister_cpu_variants(void);
public:
  static MonadData bind_inner_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static MonadData bind_outer_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void summary_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

#endif // __SNAP_CONVERGENCE_H__


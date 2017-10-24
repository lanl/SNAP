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

#include "snap.h"
#include "convergence.h"

extern LegionRuntime::Logger::Category log_snap;

//------------------------------------------------------------------------------
ConvergenceMonad::ConvergenceMonad(Context c, Runtime *rt)
  : ctx(c), runtime(rt)
//------------------------------------------------------------------------------
{
  Future f = runtime->get_current_time_in_microseconds(ctx);
  long long init_time = f.get_result<long long>(true/*silence warnings*/);
  // Create the initial future
  MonadData init_data;
  init_data.step_start = init_time;
  init_data.outer_start = init_time;
  init_data.inner_start = init_time;
  init_data.time_step_number = 0;
  init_data.inner_loop_number = 0;
  init_data.outer_loop_number = 0;
  init_data.total_inner_loops = 0;
  init_data.total_outer_loops = 0;
  init_data.total_inner_time = 0;
  init_data.total_outer_time = 0;
  init_data.total_step_time = 0;

  monad_future = Future::from_untyped_pointer(runtime, &init_data, 
                                              sizeof(init_data));
}

//------------------------------------------------------------------------------
ConvergenceMonad::ConvergenceMonad(const ConvergenceMonad &rhs)
  : ctx(rhs.ctx), runtime(rhs.runtime)
//------------------------------------------------------------------------------
{
  // should never be called
  assert(false);
}

//------------------------------------------------------------------------------
ConvergenceMonad::~ConvergenceMonad(void)
//------------------------------------------------------------------------------
{
  // Launch the summary task
  TaskLauncher launcher(Snap::SUMMARY_TASK_ID, TaskArgument(NULL, 0));
  launcher.add_future(monad_future);

  runtime->execute_task(ctx, launcher);
}

//------------------------------------------------------------------------------
ConvergenceMonad& ConvergenceMonad::operator=(const ConvergenceMonad &rhs)
//------------------------------------------------------------------------------
{
  // should never be called
  assert(false);
  return *this;
}

//------------------------------------------------------------------------------
void ConvergenceMonad::bind_inner(const Predicate &pred,
                                  const Future &inner_converged)
//------------------------------------------------------------------------------
{
  Future timing_future = runtime->get_current_time_in_microseconds(ctx, 
                                                      inner_converged);

  TaskLauncher launcher(Snap::BIND_INNER_CONVERGENCE_TASK_ID,
                        TaskArgument(NULL, 0), pred);
  launcher.add_future(monad_future);
  launcher.add_future(inner_converged);
  launcher.add_future(timing_future);
  launcher.predicate_false_future = monad_future;

  monad_future = runtime->execute_task(ctx, launcher);
}

//------------------------------------------------------------------------------
void ConvergenceMonad::bind_outer(const Predicate &pred,
                                  const Future &outer_converged)
//------------------------------------------------------------------------------
{
  Future timing_future = runtime->get_current_time_in_microseconds(ctx, 
                                                      outer_converged);

  TaskLauncher launcher(Snap::BIND_OUTER_CONVERGENCE_TASK_ID,
                        TaskArgument(NULL, 0), pred);
  launcher.add_future(monad_future);
  launcher.add_future(outer_converged);
  launcher.add_future(timing_future);
  launcher.predicate_false_future = monad_future;

  monad_future = runtime->execute_task(ctx, launcher);
}

//------------------------------------------------------------------------------
/*static*/ void ConvergenceMonad::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  char variant_name[128];
  strcpy(variant_name, "CPU ");
  strncat(variant_name, 
      Snap::task_names[Snap::BIND_INNER_CONVERGENCE_TASK_ID], 123);
  TaskVariantRegistrar inner_registrar(
      Snap::BIND_INNER_CONVERGENCE_TASK_ID, true/*global*/,
      NULL/*generator*/, variant_name);
  inner_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  inner_registrar.leaf_variant = true;
  inner_registrar.inner_variant = false;
  Runtime::preregister_task_variant<MonadData, bind_inner_implementation>(
      inner_registrar, Snap::task_names[Snap::BIND_INNER_CONVERGENCE_TASK_ID]);

  strcpy(variant_name, "CPU ");
  strncat(variant_name, 
      Snap::task_names[Snap::BIND_OUTER_CONVERGENCE_TASK_ID], 123);
  TaskVariantRegistrar outer_registrar(
      Snap::BIND_OUTER_CONVERGENCE_TASK_ID, true/*global*/,
      NULL/*generator*/, variant_name);
  outer_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  outer_registrar.leaf_variant = true;
  outer_registrar.inner_variant = false;
  Runtime::preregister_task_variant<MonadData, bind_outer_implementation>(
      outer_registrar, Snap::task_names[Snap::BIND_OUTER_CONVERGENCE_TASK_ID]);

  strcpy(variant_name, "CPU ");
  strncat(variant_name,
      Snap::task_names[Snap::SUMMARY_TASK_ID], 123);
  TaskVariantRegistrar summary_registrar(
      Snap::SUMMARY_TASK_ID, true/*global*/, NULL/*generator*/, variant_name);
  summary_registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  summary_registrar.leaf_variant = true;
  summary_registrar.inner_variant = false;
  Runtime::preregister_task_variant<summary_implementation>(summary_registrar,
      Snap::task_names[Snap::SUMMARY_TASK_ID]);
}

//------------------------------------------------------------------------------
/*static*/ ConvergenceMonad::MonadData
              ConvergenceMonad::bind_inner_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  // Should always have three futures
  assert(task->futures.size() == 3);
  // First is the monad data
  MonadData data = 
    task->futures[0].get_result<MonadData>(true/*silence warnings*/);
  // Second is the convergence result
  bool converged = task->futures[1].get_result<bool>(true/*silence warnings*/);
  // Third is the timing information for when the convergence result was ready
  long long time = 
    task->futures[2].get_result<long long>(true/*silence warnings*/);

  const long long loop_time = time - data.inner_start;
  if (converged) {
    log_snap.print("Inner loop %d of outer loop %d of "
                   "time step %d CONVERGED in %lld microseconds",
                   data.inner_loop_number, data.outer_loop_number,
                   data.time_step_number, loop_time);
    // Inner count goes back to zero
    data.inner_loop_number = 0;
  }
  else {
    log_snap.print("Inner loop %d of outer loop %d of "
                   "time step %d did not converge in %lld microseconds",
                   data.inner_loop_number, data.outer_loop_number,
                   data.time_step_number, loop_time);
    data.inner_loop_number++;
  }
  // Remember the results
  data.total_inner_loops++;
  data.total_inner_time += loop_time;
  // Reset the timer
  data.inner_start = time;
  return data;
}

//------------------------------------------------------------------------------
/*static*/ ConvergenceMonad::MonadData 
              ConvergenceMonad::bind_outer_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  // Should always have three futures
  assert(task->futures.size() == 3);
  // First is the monad data
  MonadData data = 
    task->futures[0].get_result<MonadData>(true/*silence warnings*/);
  // Second is the convergence result
  bool converged = task->futures[1].get_result<bool>(true/*silence warnings*/);
  // Third is the timing information for when the convergence result was ready
  long long time = 
    task->futures[2].get_result<long long>(true/*silence warnings*/);

  const long long loop_time = time - data.outer_start;
  if (converged) {
    log_snap.print("Outer loop %d of time step %d CONVERGED in %lld "
                   "microseconds", data.outer_loop_number,
                   data.time_step_number, loop_time);
    const long long step_time = time - data.step_start;
    log_snap.print("Time step %d took %lld microseconds", 
                   data.time_step_number, step_time);
    data.time_step_number++;
    data.outer_loop_number = 0;
    data.step_start = time;
    data.total_step_time += step_time;
  } else {
    log_snap.print("Outer loop %d of time step %d did not converge "
                   "in %lld microsecond", data.outer_loop_number,
                   data.time_step_number, loop_time);
    data.outer_loop_number++;
  }
  data.total_outer_loops++;
  data.total_outer_time += loop_time;
  // Reset the timer
  data.outer_start = time;
  // And the inner loop number
  data.inner_loop_number = 0;

  return data;
}

//------------------------------------------------------------------------------
/*static*/ void ConvergenceMonad::summary_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  // Should always have one future
  assert(task->futures.size() == 1);
  // Get the monad data
  MonadData data = 
    task->futures[0].get_result<MonadData>(true/*silence warnings*/);

  log_snap.print("---------------------------------------------------------");
  log_snap.print("SNAP Execution Summary");
  log_snap.print("---------------------------------------------------------");
  log_snap.print("  Execution Time: %lld us", data.total_step_time);
  log_snap.print("  Total Time Steps: %d (avg %.8g us / iter)",
      data.time_step_number,
      double(data.total_step_time) / double(data.time_step_number));
  log_snap.print("  Total Outer Loops: %d (avg %.8g us / iter)",
      data.total_outer_loops,
      double(data.total_outer_time) / double(data.total_outer_loops));
  log_snap.print("  Total Inner Loops: %d (avg %.8g us / iter)", 
      data.total_inner_loops, 
      double(data.total_inner_time) / double(data.total_inner_loops));
  log_snap.print("---------------------------------------------------------");
}


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
#include "init.h"
#include "outer.h"
#include "inner.h"
#include "sweep.h"
#include "expxs.h"
#include "mms.h"
#include "convergence.h"

#include <cstdio>

#ifndef MIN
#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#endif

using namespace LegionRuntime::Accessor;

LegionRuntime::Logger::Category log_snap("snap");

const char* Snap::task_names[LAST_TASK_ID] = { SNAP_TASK_NAMES };

//------------------------------------------------------------------------------
void Snap::setup(void)
//------------------------------------------------------------------------------
{
  // This is the index space for the spatial simulation 
  const int upper[3] = { nx_chunks*nx_per_chunk - 1,
                         ny_chunks*ny_per_chunk - 1,
                         nz_chunks*nz_per_chunk - 1 };
  simulation_bounds = Rect<3>(Point<3>::ZEROES(), Point<3>(upper));
  simulation_is = 
    runtime->create_index_space(ctx, Domain::from_rect<3>(simulation_bounds));
  runtime->attach_name(simulation_is, "Simulation Space");
  // Create the disjoint partition of the index space 
  {
    const int bf[3] = { nx_per_chunk, ny_per_chunk, nz_per_chunk };
    Point<3> blocking_factor(bf);
    Blockify<3> spatial_map(blocking_factor);
    spatial_ip = 
      runtime->create_index_partition(ctx, simulation_is,
                                      spatial_map, DISJOINT_PARTITION);
    runtime->attach_name(spatial_ip, "Spatial Partition");
  }
  // Launch bounds though ignore the boundary condition chunks
  // so they start at 1 and go to number of chunks, just like Fortran!
  const int chunks[3] = { nx_chunks, ny_chunks, nz_chunks };
  launch_bounds = Rect<3>(Point<3>::ZEROES(), 
                          Point<3>(chunks) - Point<3>::ONES()); 
  // Make the index spaces for the flux exchanges
  {
    const int upper_xy[2] = { nx_chunks * nx_per_chunk - 1, 
                              ny_chunks * ny_per_chunk - 1};
    xy_flux_is = runtime->create_index_space(ctx, Domain::from_rect<2>(
          Rect<2>(Point<2>::ZEROES(), Point<2>(upper_xy))));
    runtime->attach_name(xy_flux_is, "XY Flux");
    const int bf[2] = { nx_per_chunk, ny_per_chunk };
    Point<2> flux_bf(bf);
    Blockify<2> flux_map(flux_bf);
    xy_flux_ip = runtime->create_index_partition(ctx, xy_flux_is, 
                                    flux_map, DISJOINT_PARTITION);
    runtime->attach_name(xy_flux_ip, "XY Flux Partition");
  }
  {
    const int upper_yz[2] = { ny_chunks * ny_per_chunk - 1, 
                              nz_chunks * nz_per_chunk - 1};
    yz_flux_is = runtime->create_index_space(ctx, Domain::from_rect<2>(
          Rect<2>(Point<2>::ZEROES(), Point<2>(upper_yz))));
    runtime->attach_name(yz_flux_is, "YZ Flux");
    const int bf[2] = { ny_per_chunk, nz_per_chunk };
    Point<2> flux_bf(bf);
    Blockify<2> flux_map(bf);
    yz_flux_ip = runtime->create_index_partition(ctx, yz_flux_is,
                                      flux_map, DISJOINT_PARTITION);
    runtime->attach_name(yz_flux_ip, "YZ Flux Partition");
  }
  {
    const int upper_xz[2] = { nx_chunks * nx_per_chunk - 1, 
                              nz_chunks * nz_per_chunk - 1};
    xz_flux_is = runtime->create_index_space(ctx, Domain::from_rect<2>(
          Rect<2>(Point<2>::ZEROES(), Point<2>(upper_xz))));
    runtime->attach_name(xz_flux_is, "XZ Flux");
    const int bf[2] = { nx_per_chunk, nz_per_chunk };
    Point<2> flux_bf(bf);
    Blockify<2> flux_map(bf);
    xz_flux_ip = runtime->create_index_partition(ctx, xz_flux_is,
                                      flux_map, DISJOINT_PARTITION);
    runtime->attach_name(xz_flux_ip, "XZ Flux Partition");
  }
  // Make some of our other field spaces
  const int nmat = (material_layout == HOMOGENEOUS_LAYOUT) ? 1 : 2;
  material_is = runtime->create_index_space(ctx,
      Domain::from_rect<1>(Rect<1>(Point<1>(1), Point<1>(nmat))));
  const int slgg_lower[2] = { 1, 0 };
  const int slgg_upper[2] = { nmat, num_groups-1 };
  Rect<2> slgg_bounds((Point<2>(slgg_lower)), Point<2>(slgg_upper));
  slgg_is = runtime->create_index_space(ctx, Domain::from_rect<2>(slgg_bounds));
  runtime->attach_name(slgg_is, "Scattering Index Space");
  point_is = runtime->create_index_space(ctx, 
      Domain::from_rect<1>(Rect<1>(Point<1>::ZEROES(), Point<1>::ZEROES())));
  runtime->attach_name(point_is, "Point Index Space");
  // Make a field space for all the energy groups
  group_fs = runtime->create_field_space(ctx); 
  runtime->attach_name(group_fs, "Energy Group Field Space");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, group_fs);
    assert(num_groups <= SNAP_MAX_ENERGY_GROUPS);
    std::vector<FieldID> group_fields(num_groups);
    for (int idx = 0; idx < num_groups; idx++)
      group_fields[idx] = SNAP_ENERGY_GROUP_FIELD(idx);
    std::vector<size_t> group_sizes(num_groups, sizeof(double));
    allocator.allocate_fields(group_sizes, group_fields);
    char name_buffer[64];
    for (int idx = 0; idx < num_groups; idx++)
    {
      snprintf(name_buffer,63,"Energy Group %d", idx);
      runtime->attach_name(group_fs, group_fields[idx], name_buffer);
    }
  }
  // This field space contains fields for all 8 corners for each group
  flux_fs = runtime->create_field_space(ctx);
  runtime->attach_name(flux_fs, "Flux Exchange Field Space");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, flux_fs);
    // Normal energy group fields
    assert(num_groups <= SNAP_MAX_ENERGY_GROUPS);
    std::vector<FieldID> group_fields(num_groups*8/*num corners*/);
    unsigned idx = 0;
    for (int g = 0; g < num_groups; g++) {
      for (int c = 0; c < 8/*corners*/; c++)
        group_fields[idx++] = SNAP_FLUX_GROUP_FIELD(g, c);
    }
    std::vector<size_t> group_sizes(num_groups*8/*num corners*/, num_angles*sizeof(double));
    allocator.allocate_fields(group_sizes, group_fields);
    char name_buffer[64];
    idx = 0;
    for (int g = 0; g < num_groups; g++){
      for (int c = 0; c < 8/*corners*/; c++) { 
        snprintf(name_buffer,63,"Flux Group %d for Corner %d", g, c);
        runtime->attach_name(flux_fs, group_fields[idx++], name_buffer);
      }
    }
  }
  // Make a fields space for the moments for each energy group
  moment_fs = runtime->create_field_space(ctx);
  runtime->attach_name(moment_fs, "Moment Field Space");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, moment_fs);
    assert(num_groups <= SNAP_MAX_ENERGY_GROUPS);
    std::vector<FieldID> moment_fields(num_groups);
    for (int idx = 0; idx < num_groups; idx++)
      moment_fields[idx] = SNAP_ENERGY_GROUP_FIELD(idx);
    // Notice that the field size is as big as necessary to store all moments
    std::vector<size_t> moment_sizes(num_groups, sizeof(MomentQuad));
    allocator.allocate_fields(moment_sizes, moment_fields);
    char name_buffer[64];
    for (int idx = 0; idx < num_groups; idx++)
    {
      snprintf(name_buffer,63,"Moment Energy Group %d", idx);
      runtime->attach_name(moment_fs, moment_fields[idx], name_buffer);
    }
  }
  flux_moment_fs = runtime->create_field_space(ctx);
  runtime->attach_name(flux_moment_fs, "Flux Moment Field Space");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, flux_moment_fs);
    assert(num_groups <= SNAP_MAX_ENERGY_GROUPS);
    std::vector<FieldID> moment_fields(num_groups);
    for (int idx = 0; idx < num_groups; idx++)
      moment_fields[idx] = SNAP_ENERGY_GROUP_FIELD(idx);
    // Storing number of moments - 1
    std::vector<size_t> moment_sizes(num_groups,sizeof(MomentTriple)); 
    allocator.allocate_fields(moment_sizes, moment_fields);
    char name_buffer[64];
    for (int idx = 0; idx < num_groups; idx++)
    {
      snprintf(name_buffer,63,"Moment Flux Energy Group %d", idx);
      runtime->attach_name(flux_moment_fs, moment_fields[idx], name_buffer);
    }
  }
  mat_fs = runtime->create_field_space(ctx);
  runtime->attach_name(mat_fs, "Material Field Space");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, mat_fs);
    allocator.allocate_field(sizeof(int), FID_SINGLE);
    runtime->attach_name(mat_fs, FID_SINGLE, "Material Field");
  }
  angle_fs = runtime->create_field_space(ctx);
  runtime->attach_name(angle_fs, "Denominator Inverse Field Space");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, angle_fs);
    std::vector<FieldID> dinv_fields(num_groups);
    for (int idx = 0; idx < num_groups; idx++)
      dinv_fields[idx] = SNAP_ENERGY_GROUP_FIELD(idx);
    std::vector<size_t> dinv_sizes(num_groups, 
                                   Snap::num_angles*sizeof(double));
    allocator.allocate_fields(dinv_sizes, dinv_fields);
    char name_buffer[64];
    for (int idx = 0; idx < num_groups; idx++)
    {
      snprintf(name_buffer,63,"Dinv Group %d", idx);
      runtime->attach_name(angle_fs, dinv_fields[idx], name_buffer);
    }
  }
  // Compute the wavefronts for our sweeps
  for (int corner = 0; corner < num_corners; corner++)
  {
    assert(!wavefront_map[corner].empty());
    for (std::vector<std::vector<DomainPoint> >::const_iterator it = 
         wavefront_map[corner].begin(); it != wavefront_map[corner].end(); it++)
    {
      const size_t wavefront_points = it->size();
      assert(wavefront_points > 0);
      wavefront_domains[corner].push_back(Domain::from_rect<1>(
            Rect<1>(Point<1>(0), Point<1>(wavefront_points-1))));
    }
  }
}

//------------------------------------------------------------------------------
void Snap::transport_solve(void)
//------------------------------------------------------------------------------
{
  // Use a tunable variable to decide how far ahead the outer loop gets
  Future outer_runahead_future = 
    runtime->select_tunable_value(ctx, OUTER_RUNAHEAD_TUNABLE);
  // Same thing with the inner loop
  Future inner_runahead_future = 
    runtime->select_tunable_value(ctx, INNER_RUNAHEAD_TUNABLE);
  // Also get the energy group chunk factor for sweeps
  Future sweep_energy_chunks_future = 
    runtime->select_tunable_value(ctx, SWEEP_ENERGY_CHUNKS_TUNABLE);

  // Create our important arrays
  SnapArray flux0(simulation_is, spatial_ip, group_fs, 
                  ctx, runtime, "flux0");
  SnapArray flux0po(simulation_is,spatial_ip, group_fs, 
                    ctx, runtime, "flux0po");
  SnapArray flux0pi(simulation_is, spatial_ip, group_fs, 
                    ctx, runtime, "flux0pi");
  SnapArray fluxm(simulation_is, spatial_ip, flux_moment_fs, 
                  ctx, runtime, "fluxm");
  SnapArray flux_xy(xy_flux_is, xy_flux_ip, flux_fs,
                    ctx, runtime, "fluxXY");
  SnapArray flux_yz(yz_flux_is, yz_flux_ip, flux_fs,
                    ctx, runtime, "fluxYZ");
  SnapArray flux_xz(xz_flux_is, xz_flux_ip, flux_fs,
                    ctx, runtime, "fluxXZ");

  SnapArray qi(simulation_is, spatial_ip, group_fs, ctx, runtime, "qi");
  SnapArray q2grp0(simulation_is, spatial_ip, group_fs, ctx, runtime, "q2grp0");
  SnapArray q2grpm(simulation_is, spatial_ip, flux_moment_fs, ctx, runtime,"q2grpm");
  SnapArray qtot(simulation_is, spatial_ip, moment_fs, ctx, runtime, "qtot");

  SnapArray mat(simulation_is, spatial_ip, mat_fs, ctx, runtime, "mat");
  SnapArray sigt(material_is, IndexPartition::NO_PART, group_fs, 
                 ctx, runtime, "sigt");
  SnapArray siga(material_is, IndexPartition::NO_PART, group_fs,
                 ctx, runtime, "siga");
  SnapArray sigs(material_is, IndexPartition::NO_PART, group_fs,
                 ctx, runtime, "sigs");
  SnapArray slgg(slgg_is, IndexPartition::NO_PART, moment_fs, 
                 ctx, runtime, "slgg");

  SnapArray t_xs(simulation_is, spatial_ip, group_fs, ctx, runtime, "t_xs");
  SnapArray a_xs(simulation_is, spatial_ip, group_fs, ctx, runtime, "a_xs");
  SnapArray s_xs(simulation_is, spatial_ip, moment_fs, ctx, runtime, "s_xs");

  SnapArray vel(point_is, IndexPartition::NO_PART, group_fs, 
                ctx, runtime, "vel");
  SnapArray vdelt(point_is, IndexPartition::NO_PART, group_fs, 
                  ctx, runtime, "vdelt");
  SnapArray dinv(simulation_is, spatial_ip, angle_fs, 
                 ctx, runtime, "dinv"); 

  SnapArray *time_flux_even[8];
  SnapArray *time_flux_odd[8]; 
  for (int i = 0; i < 8; i++) {
    char name_buffer[64];
    snprintf(name_buffer, 63, "time flux even %d", i);
    time_flux_even[i] = new SnapArray(simulation_is, spatial_ip, angle_fs, 
                                      ctx, runtime, name_buffer);
    snprintf(name_buffer, 63, "time flux odd %d", i);
    time_flux_odd[i] = new SnapArray(simulation_is, spatial_ip, angle_fs,
                                     ctx, runtime, name_buffer);
  }
  // Only necessary for MMS
  SnapArray *qim[8];
  SnapArray ref_flux(simulation_is, spatial_ip, group_fs, 
                     ctx, runtime, "ref_flux");
  SnapArray ref_fluxm(simulation_is, spatial_ip, flux_moment_fs, 
                      ctx, runtime, "ref_fluxm");
  const bool do_mms = (source_layout == MMS_SOURCE);
  for (int i = 0; i < 8; i++) {
    char name_buffer[64];
    snprintf(name_buffer, 63, "qim %d", i);
    qim[i] = new SnapArray(simulation_is, spatial_ip, angle_fs,
                           ctx, runtime, name_buffer);
  }

  // Initialize our data
  flux0.initialize();
  flux0po.initialize();
  flux0pi.initialize();
  if (num_moments > 1)
    fluxm.initialize();

  qi.initialize();
  q2grp0.initialize();
  if (num_moments > 1)
    q2grpm.initialize();
  qtot.initialize();

  mat.initialize<int>(1);
  sigt.initialize();
  siga.initialize();
  sigs.initialize();
  slgg.initialize();

  t_xs.initialize();
  a_xs.initialize();
  s_xs.initialize();

  vel.initialize();
  vdelt.initialize();
  dinv.initialize();

  for (int i = 0; i < 8; i++) {
    time_flux_even[i]->initialize();
    time_flux_odd[i]->initialize();
  } 

  // Launch some tasks to initialize the application data
  if (material_layout != HOMOGENEOUS_LAYOUT)
  {
    InitMaterial init_material(*this, mat);
    init_material.dispatch(ctx, runtime);
  }
  if (!do_mms)
  {
    InitSource init_source(*this, qi);
    init_source.dispatch(ctx, runtime);
  }
#ifdef USE_GPU_KERNELS
  {
    Future f_gpus = runtime->select_tunable_value(ctx, 
        DefaultMapper::DEFAULT_TUNABLE_GLOBAL_GPUS);
    int num_gpus = f_gpus.get_result<int>(true/*silence warnings*/);
    assert(num_gpus > 0);
    Rect<3> gpu_bounds(Point<3>::ZEROES(), Point<3>::ZEROES());
    gpu_bounds.hi.x[0] = num_gpus - 1;
    InitGPUSweep init_sweep(*this, gpu_bounds);
    init_sweep.dispatch(ctx, runtime, true/*block*/);
  }
#endif
  initialize_scattering(sigt, siga, sigs, slgg);
  initialize_velocity(vel, vdelt);

  if (do_mms) {
    ref_flux.initialize();
    ref_fluxm.initialize();
    MMSInitFlux init_mms_flux(*this, ref_flux, ref_fluxm);
    init_mms_flux.dispatch(ctx, runtime);
    for (int i = 0; i < 8; i++) {
      qim[i]->initialize();
      MMSInitSource init_mms_source(*this, ref_flux, ref_fluxm, mat,
                                    sigt, slgg, *(qim[i]), i);
      init_mms_source.dispatch(ctx, runtime);
    }
    if (time_dependent) {
      MMSInitTimeDependent mms_time_dependent(*this, vel, ref_flux, qi); 
      mms_time_dependent.dispatch(ctx, runtime);
    }
  }

  // Tunables should be ready by now
  const unsigned outer_runahead = 
    outer_runahead_future.get_result<unsigned>(true/*silence warnings*/);
  assert(outer_runahead > 0);
  const unsigned inner_runahead = 
    inner_runahead_future.get_result<unsigned>(true/*silence warnings*/);
  assert(inner_runahead > 0);
  const int energy_group_chunks = 
    sweep_energy_chunks_future.get_result<int>(true/*silence warnings*/);
  // Loop over time steps
  std::deque<Future> outer_converged_tests;
  std::deque<Future> inner_converged_tests;
  // Use this for when predicates evaluate to false, tasks can then
  // return true to indicate convergence
  const Future true_future = Future::from_value<bool>(runtime, true);
  // Use this for printing convergence and timing information
  // in a deferred execution environment with predication
  ConvergenceMonad convergence(ctx, runtime);
  // Iterate over time steps
  bool even_time_step = false;
  for (int cy = 0; cy < num_steps; ++cy)
  {
    even_time_step = !even_time_step;
    // Some of this is a little weird, you can in theory lift some
    // of this out the time stepping loop because the mock velocity 
    // array and the material array aren't changing, but I think that 
    // is just an artifact off SNAP and not a more general property of PARTISN, 
    // SNAP developers have now confirmed this so we'll leave this
    // here to be consistent with the original implementation of SNAP
    for (int g = 0; g < num_groups; g += energy_group_chunks)
    {
      int group_stop = g + energy_group_chunks - 1;
      if (group_stop >= num_groups)
        group_stop = num_groups - 1;
      ExpandCrossSection expxs(*this, siga, mat, a_xs, g, group_stop);
      expxs.dispatch(ctx, runtime);
    }
    for (int g = 0; g < num_groups; g += energy_group_chunks)
    {
      int group_stop = g + energy_group_chunks - 1;
      if (group_stop >= num_groups)
        group_stop = num_groups - 1;
      ExpandCrossSection expxs(*this, sigt, mat, t_xs, g, group_stop);
      expxs.dispatch(ctx, runtime);
    }
    for (int g = 0; g < num_groups; g += energy_group_chunks)
    {
      int group_stop = g + energy_group_chunks - 1;
      if (group_stop >= num_groups)
        group_stop = num_groups - 1;
      ExpandScatteringCrossSection expxs(*this, slgg, mat, s_xs, g, group_stop);
      expxs.dispatch(ctx, runtime);
    }
    for (int g = 0; g < num_groups; g += energy_group_chunks)
    {
      int group_stop = g + energy_group_chunks - 1;
      if (group_stop >= num_groups)
        group_stop = num_groups - 1;
      CalculateGeometryParam geom(*this, t_xs, vdelt, dinv, g, group_stop);
      geom.dispatch(ctx, runtime);
    }
    // Scale the manufactured solution for time
    if (do_mms) 
    {
      if (cy == 0) {
        const double time = dt * 0.5;
        for (int i = 0; i < 8; i++) {
          MMSScale scale_mms(*this, *(qim[i]), time);
          scale_mms.dispatch(ctx, runtime);
        }
      } else {
        const double sf = double(2 * cy + 1) / double(2 * cy - 1);
        for (int i = 0; i < 8; i++) {
          MMSScale scale_mms(*this, *(qim[i]), sf);
          scale_mms.dispatch(ctx, runtime);
        }
      }
    }
    outer_converged_tests.clear();
    Predicate outer_pred = Predicate::TRUE_PRED;
    Future timing_future_precondition;
    // The outer solve loop    
    for (int otno = 0; otno < max_outer_iters; ++otno)
    {
      // Do the outer source calculation 
      CalcOuterSource outer_src(*this, outer_pred, qi, slgg, mat, 
                                q2grp0, q2grpm, flux0, fluxm);
      outer_src.dispatch(ctx, runtime);
      // Save the fluxes
      save_fluxes(outer_pred, flux0, flux0po);
      // Do the inner solve
      inner_converged_tests.clear();
      Predicate inner_pred = outer_pred;
      Future inner_converged;
      // The inner solve loop
      for (int inno=0; inno < max_inner_iters; ++inno)
      {
        // Do the inner source calculation
        CalcInnerSource inner_src(*this, inner_pred, s_xs, flux0, fluxm,
                                  q2grp0, q2grpm, qtot);
        inner_src.dispatch(ctx, runtime);
        // Save the fluxes
        save_fluxes(inner_pred, flux0, flux0pi);
        flux0.initialize(inner_pred);
        // Perform the sweeps
        perform_sweeps(inner_pred, flux0, fluxm, qtot, vdelt, dinv, t_xs,
                       even_time_step ? time_flux_even : time_flux_odd,
                       even_time_step ? time_flux_odd : time_flux_even, 
                       qim, flux_xy, flux_yz, flux_xz, energy_group_chunks); 
        // Test for inner convergence
        Predicate converged = test_inner_convergence(inner_pred, flux0, 
                             flux0pi, true_future, energy_group_chunks);
        inner_converged = runtime->get_predicate_future(ctx, converged);
        inner_converged_tests.push_back(inner_converged);
        convergence.bind_inner(inner_pred, inner_converged);
        // Update the next predicate
        inner_pred = runtime->predicate_not(ctx, converged);
        // See if we've run far enough ahead
        if (inner_converged_tests.size() == inner_runahead)
        {
          Future f = inner_converged_tests.front();
          inner_converged_tests.pop_front();
          if (f.get_result<bool>(true/*silence warnings*/))
            break;
        }
      }
      // Test for outer convergence
      // Original SNAP says to skip this on the first iteration
      if (otno == 0)
        continue;
      Predicate converged = test_outer_convergence(outer_pred, flux0,
           flux0po, inner_converged, true_future, energy_group_chunks);
      Future outer_converged = runtime->get_predicate_future(ctx, converged);
      outer_converged_tests.push_back(outer_converged);
      convergence.bind_outer(outer_pred, outer_converged);
      // Update the next predicate
      outer_pred = runtime->predicate_not(ctx, converged);
      // See if we've run far enough ahead
      if (outer_converged_tests.size() == outer_runahead)
      {
        Future f = outer_converged_tests.front();
        outer_converged_tests.pop_front();
        if (f.get_result<bool>(true/*silence warnings*/))
          break;
      }
    }
  }
  if (do_mms) {
    MMSCompare compare_mms(*this, flux0, ref_flux); 
    Future f = compare_mms.dispatch<MMSReduction>(ctx, runtime);
    MomentTriple result = f.get_result<MomentTriple>(true/*silence warnings*/);
    log_snap.print("MMS Max Diff: %.8g", result[0]);
    log_snap.print("MMS Min Diff: %.8g", result[1]);
    const size_t total_cells = nx * nx_chunks * ny * ny_chunks * nz * nz_chunks;
    const double avg_diff = result[2] / double(total_cells * Snap::num_groups);
    log_snap.print("MMS Avg Diff: %.8g", avg_diff);
    if (result[0] > 0.1) {
      log_snap.error("MMS FAILURE! MAX IS LARGER THAN 0.1!");
      assert(false);
    }
    if (avg_diff > 0.001) {
      log_snap.error("MMS FAILURE! AVG IS LARGER THAN 0.001!\n");
      assert(false);
    }
  }
  for (int i = 0; i < 8; i++) {
    delete time_flux_even[i];
    delete time_flux_odd[i];
  }
  if (do_mms) {
    for (int i = 0; i < 8; i++)
      delete qim[i];
  }
}

//------------------------------------------------------------------------------
void Snap::initialize_scattering(const SnapArray &sigt, const SnapArray &siga,
                             const SnapArray &sigs, const SnapArray &slgg) const
//------------------------------------------------------------------------------
{
  PhysicalRegion sigt_region = sigt.map();
  PhysicalRegion siga_region = siga.map();
  PhysicalRegion sigs_region = sigs.map();
  PhysicalRegion slgg_region = slgg.map();
  sigt_region.wait_until_valid(true/*ignore warnings*/);
  siga_region.wait_until_valid(true/*ignore warnings*/);
  sigs_region.wait_until_valid(true/*ignore warnings*/);
  slgg_region.wait_until_valid(true/*ignore warnings*/);

  const DomainPoint one = DomainPoint::from_point<1>(Point<1>(1));
  const DomainPoint two = DomainPoint::from_point<1>(Point<1>(2));
  std::vector<RegionAccessor<AccessorType::Generic,double> > fa_sigt(num_groups);
  std::vector<RegionAccessor<AccessorType::Generic,double> > fa_siga(num_groups);
  std::vector<RegionAccessor<AccessorType::Generic,double> > fa_sigs(num_groups);
  for (int g = 0; g < num_groups; g++)
  {
    fa_sigt[g] = sigt_region.get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(g)).typeify<double>();
    fa_siga[g] = siga_region.get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(g)).typeify<double>();
    fa_sigs[g] = sigs_region.get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(g)).typeify<double>();
  }

  fa_sigt[0].write(one, 1.0);
  fa_siga[0].write(one, 0.5);
  fa_sigs[0].write(one, 0.5);
  for (int g = 1; g < num_groups; g++)
  {
    fa_sigt[g].write(one, 0.01  * fa_sigt[g-1].read(one));
    fa_siga[g].write(one, 0.005 * fa_siga[g-1].read(one));
    fa_sigs[g].write(one, 0.005 * fa_sigs[g-1].read(one));
  }

  if (material_layout != HOMOGENEOUS_LAYOUT) {
    fa_sigt[0].write(two, 2.0);
    fa_siga[0].write(two, 0.8);
    fa_sigs[0].write(two, 1.2);
    for (int g = 1; g < num_groups; g++)
    {
      fa_sigt[g].write(two, 0.01  * fa_sigt[g-1].read(two));
      fa_siga[g].write(two, 0.005 * fa_siga[g-1].read(two));
      fa_sigs[g].write(two, 0.005 * fa_sigs[g-1].read(two));
    }
  }

  std::vector<RegionAccessor<AccessorType::Generic,MomentQuad> > fa_slgg(num_groups); 
  for (int g = 0; g < num_groups; g++)
    fa_slgg[g] = slgg_region.get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(g)).typeify<MomentQuad>();

  Point<2> p2 = Point<2>::ZEROES();
  p2.x[0] = 1;
  if (num_groups == 1) {
    MomentQuad local;
    local[0] = fa_sigs[0].read(one);
    fa_slgg[0].write(DomainPoint::from_point<2>(p2), local); 
    if (material_layout != HOMOGENEOUS_LAYOUT) {
      p2.x[1] = 1; 
      local[0] = fa_sigs[0].read(two);
      fa_slgg[0].write(DomainPoint::from_point<2>(p2), local);
    }
  } else {
    MomentQuad local;
    for (int g = 0; g < num_groups; g++) {
      p2.x[1] = g; 
      const DomainPoint dp = DomainPoint::from_point<2>(p2);
      local[0] = 0.2 * fa_sigs[g].read(one);
      fa_slgg[g].write(dp, local);
      if (g > 0) {
        const double t = 1.0 / double(g);
        for (int g2 = 0; g2 < g; g2++) {
          local[0] = 0.1 * fa_sigs[g].read(one) * t;
          fa_slgg[g2].write(dp, local);
        }
      } else {
        local[0] = 0.3 * fa_sigs[g].read(one);
        fa_slgg[g].write(dp, local); 
      }

      if (g < (num_groups-1)) {
        const double t = 1.0 / double(num_groups-(g+1));
        for (int g2 = g+1; g2 < num_groups; g2++) {
          local[0] = 0.7 * fa_sigs[g].read(one) * t;
          fa_slgg[g2].write(dp, local);
        }
      } else {
        local[0] = 0.9 * fa_sigs[g].read(one);
        fa_slgg[g].write(dp, local);
      }
    }
    if (material_layout != HOMOGENEOUS_LAYOUT) {
      p2.x[0] = 2;
      for (int g = 0; g < num_groups; g++) {
        p2.x[1] = g; 
        const DomainPoint dp = DomainPoint::from_point<2>(p2);
        local[0] = 0.5 * fa_sigs[g].read(two);
        fa_slgg[g].write(dp, local);
        if (g > 0) {
          const double t = 1.0 / double(g);
          for (int g2 = 0; g2 < g; g2++) {
            local[0] = 0.1 * fa_sigs[g].read(two) * t;
            fa_slgg[g2].write(dp, local);
          }
        } else {
          local[0] = 0.6 * fa_sigs[g].read(two);
          fa_slgg[g].write(dp, local); 
        }

        if (g < (num_groups-1)) {
          const double t = 1.0 / double(num_groups-(g+1));
          for (int g2 = g+1; g2 < num_groups; g2++) {
            local[0] = 0.4 * fa_sigs[g].read(two) * t;
            fa_slgg[g2].write(dp, local);
          }
        } else {
          local[0] = 0.9 * fa_sigs[g].read(two);
          fa_slgg[g].write(dp, local);
        }
      }
    }
  }
  if (num_moments > 1) 
  {
    p2 = Point<2>::ZEROES();
    p2.x[0] = 1;
    for (int m = 1; m < num_moments; m++) {
      for (int g = 0; g < num_groups; g++) {
        p2.x[1] = g;
        DomainPoint dp = DomainPoint::from_point<2>(p2);
        for (int g2 = 0; g2 < num_groups; g2++) {
          MomentQuad quad = fa_slgg[g2].read(dp);
          quad[m] = ((m == 1) ? 0.1 : 0.5) * quad[m-1];
          fa_slgg[g2].write(dp, quad);
        }
      }
    }
    if (material_layout != HOMOGENEOUS_LAYOUT) {
      p2.x[0] = 2;
      for (int m = 1; m < num_moments; m++) {
        for (int g = 0; g < num_groups; g++) {
          p2.x[1] = g;
          DomainPoint dp = DomainPoint::from_point<2>(p2);
          for (int g2 = 0; g2 < num_groups; g2++) {
            MomentQuad quad = fa_slgg[g2].read(dp);
            quad[m] = ((m == 1) ? 0.8 : 0.6) * quad[m-1];
            fa_slgg[g2].write(dp, quad);
          }
        }
      }
    }
  }

  sigt.unmap(sigt_region);
  siga.unmap(siga_region);
  sigs.unmap(sigs_region);
  slgg.unmap(slgg_region);
}

//------------------------------------------------------------------------------
void Snap::initialize_velocity(const SnapArray &vel, 
                               const SnapArray &vdelt) const
//------------------------------------------------------------------------------
{
  PhysicalRegion vel_region = vel.map();
  PhysicalRegion vdelt_region = vdelt.map();
  vel_region.wait_until_valid(true/*ignore warnings*/);
  vdelt_region.wait_until_valid(true/*ignore warnings*/);
  const DomainPoint dp = DomainPoint::from_point<1>(Point<1>(0));
  for (int g = 0; g < num_groups; g++) 
  {
    RegionAccessor<AccessorType::Generic,double> fa_vel = 
      vel_region.get_field_accessor(SNAP_ENERGY_GROUP_FIELD(g)).typeify<double>();
    RegionAccessor<AccessorType::Generic,double> fa_vdelt = 
      vdelt_region.get_field_accessor(SNAP_ENERGY_GROUP_FIELD(g)).typeify<double>();
    const double v = double(Snap::num_groups - g);
    fa_vel.write(dp, v);
    if (Snap::time_dependent)
      fa_vdelt.write(dp, 2.0 / (Snap::dt * v));
    else
      fa_vdelt.write(dp, 0.0);
  }
  vel.unmap(vel_region);
  vdelt.unmap(vdelt_region);
}

//------------------------------------------------------------------------------
void Snap::save_fluxes(const Predicate &pred, 
                       const SnapArray &src, const SnapArray &dst) const
//------------------------------------------------------------------------------
{
  // Use this macro to disable index space copy launches
#ifdef NO_INDEX_SPACE_COPIES
  // Build the CopyLauncher
  CopyLauncher launcher(pred);
  launcher.add_copy_requirements(
      RegionRequirement(LogicalRegion::NO_REGION, READ_ONLY, 
                        EXCLUSIVE, src.get_region()),
      RegionRequirement(LogicalRegion::NO_REGION, WRITE_DISCARD,
                        EXCLUSIVE, dst.get_region()));
  const std::set<FieldID> &src_fields = src.get_all_fields();
  RegionRequirement &src_req = launcher.src_requirements.back();
  src_req.privilege_fields = src_fields;
  src_req.instance_fields.insert(src_req.instance_fields.end(),
      src_fields.begin(), src_fields.end());
  const std::set<FieldID> &dst_fields = dst.get_all_fields();
  RegionRequirement &dst_req = launcher.dst_requirements.back();
  dst_req.privilege_fields = dst_fields;
  dst_req.instance_fields.insert(dst_req.instance_fields.end(),
      dst_fields.begin(), dst_fields.end());
  // Iterate over the sub-regions and issue copies for each separately
  for (GenericPointInRectIterator<3> color_it(launch_bounds); 
        color_it; color_it++)   
  {
    DomainPoint dp = DomainPoint::from_point<3>(color_it.p);
    src_req.region = src.get_subregion(dp);
    dst_req.region = dst.get_subregion(dp);
    runtime->issue_copy_operation(ctx, launcher);
  }
#else
  IndexCopyLauncher launcher(Domain::from_rect<3>(get_launch_bounds()), pred);
  launcher.add_copy_requirements(
      RegionRequirement(src.get_partition(), 0/*projection id*/, 
                        READ_ONLY, EXCLUSIVE, src.get_region()),
      RegionRequirement(dst.get_partition(), 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, dst.get_region()));
  const std::set<FieldID> &src_fields = src.get_all_fields();
  RegionRequirement &src_req = launcher.src_requirements.back();
  src_req.privilege_fields = src_fields;
  src_req.instance_fields.insert(src_req.instance_fields.end(),
      src_fields.begin(), src_fields.end());
  const std::set<FieldID> &dst_fields = dst.get_all_fields();
  RegionRequirement &dst_req = launcher.dst_requirements.back();
  dst_req.privilege_fields = dst_fields;
  dst_req.instance_fields.insert(dst_req.instance_fields.end(),
      dst_fields.begin(), dst_fields.end());
  runtime->issue_copy_operation(ctx, launcher);
#endif
}

//------------------------------------------------------------------------------
void Snap::perform_sweeps(const Predicate &pred, const SnapArray &flux,
                          const SnapArray &fluxm, const SnapArray &qtot, 
                          const SnapArray &vdelt, const SnapArray &dinv, 
                          const SnapArray &t_xs, SnapArray *time_flux_in[8],
                          SnapArray *time_flux_out[8], SnapArray *qim[8], 
                          const SnapArray &flux_xy, const SnapArray &flux_yz,
                          const SnapArray &flux_xz, int energy_group_chunks) const
//------------------------------------------------------------------------------
{
  // Boundary fluxes always get initialized to zero before sweeps
  flux_xy.initialize(pred);
  flux_yz.initialize(pred);
  flux_xz.initialize(pred);
  // Loop over the corners
  for (int corner = 0; corner < num_corners; corner++)
  {
    // Compute the projection functions for this corner
    int ghost_offsets[3] = { 0, 0, 0 };
    for (int i = 0; i < num_dims; i++)
      ghost_offsets[i] = (corner & (0x1 << i)) >> i;
    const std::vector<Domain> &launch_domains = wavefront_domains[corner];
    // Then loop over the energy groups by chunks
    for (int group = 0; group < num_groups; group+=energy_group_chunks)
    {
      int group_stop = group + energy_group_chunks - 1;
      // Clamp to the upper bound
      if (group_stop >= num_groups)
        group_stop = num_groups-1;
      // Launch the sweep from this corner for the given set of fields
      MiniKBATask mini_kba(*this, pred, flux, fluxm, 
                           qtot, vdelt, dinv, t_xs, 
                           *time_flux_in[corner], *time_flux_out[corner],
                           *qim[corner], flux_xy, flux_yz, flux_xz,
                           group, group_stop, corner, ghost_offsets);
      for (unsigned idx = 0; idx < launch_domains.size(); idx++)
        mini_kba.dispatch_wavefront(idx, launch_domains[idx], ctx, runtime);
    }
  }
}

//------------------------------------------------------------------------------
Predicate Snap::test_inner_convergence(const Predicate &inner_pred, 
                                       const SnapArray &flux0,
                                       const SnapArray &flux0pi, 
                                       const Future &pred_false_result,
                                       int energy_group_chunks) const
//------------------------------------------------------------------------------
{
  PredicateLauncher launcher(true/*and predicate*/);
  // Iterate over the energy group chunks
  for (int group = 0; group < num_groups; group+=energy_group_chunks)
  {
    int group_stop = group + energy_group_chunks - 1;
    // Clamp to the upper bound
    if (group_stop >= num_groups)
      group_stop = num_groups-1;
    TestInnerConvergence inner_conv(*this, inner_pred, flux0, flux0pi,
                                    pred_false_result, group, group_stop);
    Future f = inner_conv.dispatch<AndReduction>(ctx, runtime);
    launcher.add_predicate(runtime->create_predicate(ctx, f));
  }
  return runtime->create_predicate(ctx, launcher);
}

//------------------------------------------------------------------------------
Predicate Snap::test_outer_convergence(const Predicate &outer_pred,
                                       const SnapArray &flux0,
                                       const SnapArray &flux0po,
                                       const Future &inner_converged,
                                       const Future &pred_false_result,
                                       int energy_group_chunks) const
//------------------------------------------------------------------------------
{
  PredicateLauncher launcher(true/*and predicate*/);
  // Iterate over the energy group chunks
  for (int group = 0; group < num_groups; group+=energy_group_chunks)
  {
    int group_stop = group + energy_group_chunks - 1;
    // Clamp to the upper bound
    if (group_stop >= num_groups)
      group_stop = num_groups-1;
    TestOuterConvergence outer_conv(*this, outer_pred, flux0, flux0po,
                inner_converged, pred_false_result, group, group_stop);
    Future f = outer_conv.dispatch<AndReduction>(ctx, runtime);
    launcher.add_predicate(runtime->create_predicate(ctx, f));
  }
  return runtime->create_predicate(ctx, launcher);
}

//------------------------------------------------------------------------------
/*static*/ void Snap::snap_top_level_task(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  printf("Welcome to Legion-SNAP!\n");
  report_arguments();
  Snap snap(ctx, runtime); 
  snap.setup();
  snap.transport_solve();
}

static void skip_line(FILE *f)
{
  char buffer[80];
  assert(fgets(buffer, 79, f) > 0);
}

static void read_int(FILE *f, const char *name, int &result)
{
  char format[80];
  sprintf(format,"  %s=%%d", name);
  assert(fscanf(f, format, &result) > 0);
}

static void read_bool(FILE *f, const char *name, bool &result)
{
  char format[80];
  sprintf(format,"  %s=%%d", name);
  int temp = 0;
  assert(fscanf(f, format, &temp) > 0);
  result = (temp != 0);
}

static void read_double(FILE *f, const char *name, double &result)
{
  char format[80];
  sprintf(format,"  %s=%%lf", name);
  assert(fscanf(f, format, &result) > 0);
}

int Snap::num_dims = 1;
int Snap::nx_chunks = 4;
int Snap::ny_chunks = 1;
int Snap::nz_chunks = 1;
int Snap::nx = 4;
double Snap::lx = 1.0;
int Snap::ny = 1;
double Snap::ly = 1.0;
int Snap::nz = 1;
double Snap::lz = 1.0;
int Snap::num_moments = 1;
int Snap::num_angles = 1;
int Snap::num_groups = 1;
double Snap::convergence_eps = 1e-4;
int Snap::max_inner_iters = 5;
int Snap::max_outer_iters = 100;
bool Snap::time_dependent = false;
double Snap::total_sim_time = 0.0;
int Snap::num_steps = 1;
Snap::MaterialLayout Snap::material_layout = HOMOGENEOUS_LAYOUT;
Snap::SourceLayout Snap::source_layout = EVERYWHERE_SOURCE; 
bool Snap::dump_scatter = false;
bool Snap::dump_iteration = false;
int Snap::dump_flux = 0;
bool Snap::flux_fixup = false;
bool Snap::dump_solution = false;
int Snap::dump_kplane = 0;
int Snap::dump_population = 0;
bool Snap::minikba_sweep = true;
bool Snap::single_angle_copy = true;

int Snap::num_corners = 1;
int Snap::nx_per_chunk;
int Snap::ny_per_chunk;
int Snap::nz_per_chunk;
std::vector<std::vector<DomainPoint> > Snap::wavefront_map[8];
double Snap::dt;
int Snap::cmom;
int Snap::num_octants;
double Snap::hi;
double Snap::hj;
double Snap::hk;
double* Snap::mu;
double* Snap::w;
double *Snap::wmu;
double* Snap::eta;
double* Snap::weta;
double* Snap::xi;
double* Snap::wxi;
double* Snap::ec;
int Snap::lma[4];

//------------------------------------------------------------------------------
/*static*/ void Snap::parse_arguments(int argc, char **argv)
//------------------------------------------------------------------------------
{
  if (argc < 2) {
    printf("No input file specified. Exiting.\n");
    exit(1);
  }
  printf("Reading input file %s\n", argv[1]);
  FILE *f = fopen(argv[1], "r");
  int dummy_int = 0;
  skip_line(f);
  skip_line(f);
  read_int(f, "npey", ny_chunks);
  read_int(f, "npez", nz_chunks);
  read_int(f, "ichunk", nx_chunks);
  read_int(f, "nthreads", dummy_int);
  read_int(f, "nnested", dummy_int);
  read_int(f, "ndimen", num_dims);
  read_int(f, "nx", nx);
  read_double(f, "lx", lx);
  read_int(f, "ny", ny);
  read_double(f, "ly", ly);
  read_int(f, "nz", nz);
  read_double(f, "lz", lz);
  read_int(f, "nmom", num_moments);
  read_int(f, "nang", num_angles);
  read_int(f, "ng", num_groups);
  read_double(f, "epsi", convergence_eps);
  read_int(f, "iitm", max_inner_iters);
  read_int(f, "oitm", max_outer_iters);
  read_bool(f, "timedep", time_dependent);
  read_double(f, "tf", total_sim_time);
  read_int(f, "nsteps", num_steps);
  int mat_opt = 0;
  read_int(f, "mat_opt", mat_opt);
  switch (mat_opt)
  {
    case 0:
      {
        material_layout = HOMOGENEOUS_LAYOUT;
        break;
      }
    case 1:
      {
        material_layout = CENTER_LAYOUT;
        break;
      }
    case 2:
      {
        material_layout = CORNER_LAYOUT;
        break;
      }
    default:
      assert(false);
  }
  int src_opt = 0;
  read_int(f, "src_opt", src_opt);
  switch (src_opt)
  {
    case 0:
      {
        source_layout = EVERYWHERE_SOURCE;
        break;
      }
    case 1:
      {
        source_layout = CENTER_SOURCE;
        break;
      }
    case 2:
      {
        source_layout = CORNER_SOURCE;
        break;
      }
    case 3:
      {
        source_layout = MMS_SOURCE;
        break;
      }
    default:
      assert(false);
  }
  read_bool(f, "scatp", dump_scatter);
  read_bool(f, "it_det", dump_iteration);
  read_int(f, "fluxp", dump_flux);
  read_bool(f, "fixup", flux_fixup);
  read_bool(f, "soloutp", dump_solution);
  read_int(f, "kplane", dump_kplane);
  read_int(f, "popout", dump_population);
  read_bool(f, "swp_typ", minikba_sweep);
  int angle_copies;
  read_int(f, "angcpy", angle_copies);
  single_angle_copy = (angle_copies == 1);
  if (single_angle_copy)
  {
    printf("ERROR: angcpy=1 is not currently supported.\n");
    printf("Legion SNAP currently requires two copies of angle flux\n");
    exit(1);
  }
  fclose(f);
  if (!minikba_sweep)
  {
    printf("ERROR: swp_type=0 is not currently supported.\n");
    printf("Legion SNAP currently implements only mini-kba sweeps\n");
    exit(1);
  }
  // Check all the conditions
  assert((1 <= num_dims) && (num_dims <= 3));
  assert((1 <= nx_chunks) && (nx_chunks <= nx));
  assert((1 <= ny_chunks) && (ny_chunks <= ny));
  assert((1 <= nz_chunks) && (nz_chunks <= nz));
  assert(4 <= nx);
  assert(0.0 < lx);
  assert(4 <= ny);
  assert(0.0 < ly);
  assert(4 <= ny);
  assert(0.0 < lz);
  assert((nx % nx_chunks) == 0);
  assert((ny % ny_chunks) == 0);
  assert((nz % nz_chunks) == 0);
  assert((1 <= num_moments) && (num_moments <= 4));
  assert(1 <= num_angles);
  assert(1 <= num_groups);
  assert((0.0 < convergence_eps) && (convergence_eps < 1e-2));
  assert(1 <= max_inner_iters);
  assert(1 <= max_outer_iters);
  if (time_dependent)
    assert(0.0 <= total_sim_time);
  assert(1 <= num_steps);
  // Some derived quantities
  for (int i = 0; i < num_dims; i++)
    num_corners *= 2;
  nx_per_chunk = nx / nx_chunks;
  ny_per_chunk = ny / ny_chunks;
  nz_per_chunk = nz / nz_chunks;
  compute_wavefronts();
  compute_derived_globals();
}

static bool contains_point(Point<3> &point, int xlo, int xhi, 
                           int ylo, int yhi, int zlo, int zhi)
{
  if ((point[0] < xlo) || (point[0] > xhi))
    return false;
  if ((point[1] < ylo) || (point[1] > yhi))
    return false;
  if ((point[2] < zlo) || (point[2] > zhi))
    return false;
  return true;
}

//------------------------------------------------------------------------------
/*static*/ void Snap::compute_wavefronts(void)
//------------------------------------------------------------------------------
{
  const int chunks[3] = { nx_chunks, ny_chunks, nz_chunks };
  // Compute the mapping from corners to wavefronts
  for (int corner = 0; corner < num_corners; corner++)
  {
    Point<3> strides[3] = 
      { Point<3>::ZEROES(), Point<3>::ZEROES(), Point<3>::ZEROES() };
    Point<3> start;
    for (int i = 0; i < num_dims; i++)
    {
      start.x[i] = ((corner & (0x1 << i)) ? 0 : chunks[i]-1);
      strides[i].x[i] = ((corner & (0x1 << i)) ? 1 : -1);
    }
    std::set<DomainPoint> current_points;
    current_points.insert(DomainPoint::from_point<3>(Point<3>(start)));
    // Do a little BFS to handle weird rectangle shapes correctly
    unsigned wavefront_number = 0;
    while (!current_points.empty())
    {
      wavefront_map[corner].push_back(std::vector<DomainPoint>());
      std::vector<DomainPoint> &wavefront_points = 
                          wavefront_map[corner][wavefront_number];
      std::set<DomainPoint> next_points;
      for (std::set<DomainPoint>::const_iterator it = current_points.begin();
            it != current_points.end(); it++)
      {
        // Save the point in this wavefront
        wavefront_points.push_back(*it);
        Point<3> point = it->get_point<3>();
        for (int i = 0; i < num_dims; i++)
        {
          Point<3> next = point + strides[i];
          if (contains_point(next, 0, nx_chunks-1, 0, 
                             ny_chunks-1, 0, nz_chunks-1))
            next_points.insert(DomainPoint::from_point<3>(next));
        }
      }
      current_points = next_points;
      wavefront_number++;
    }
  }
}

//------------------------------------------------------------------------------
/*static*/ void Snap::compute_derived_globals(void)
//------------------------------------------------------------------------------
{
  dt = total_sim_time / double(num_steps);

  cmom = num_moments;
  num_octants = 2;
  hi = 2.0 / (lx / double(nx));
  hj = 2.0 / (ly / double(ny));
  hk = 2.0 / (lz / double(nz));
  const size_t buffer_size = num_angles * sizeof(double);
  mu = (double*)malloc(buffer_size);
  w = (double*)malloc(buffer_size);
  wmu = (double*)malloc(buffer_size);
  eta = (double*)malloc(buffer_size);
  weta = (double*)malloc(buffer_size);
  xi = (double*)malloc(buffer_size);
  wxi = (double*)malloc(buffer_size);

  memset(mu, 0, buffer_size);
  memset(w, 0, buffer_size);
  memset(wmu, 0, buffer_size);
  memset(eta, 0, buffer_size);
  memset(weta, 0, buffer_size);
  memset(xi, 0, buffer_size);
  memset(wxi, 0, buffer_size);

  if (num_dims > 1) {
    cmom = num_moments * (num_moments+1) / 2;
    num_octants = 4;
  }

  if (num_dims > 2) { 
    cmom = num_moments * num_moments;
    num_octants = 8;
  }

  const double dm = 1.0 / double(num_angles);

  mu[0] = 0.5 * dm;
  for (int i = 1; i < num_angles; i++)
    mu[i] = mu[i-1] + dm;
  if (num_dims > 1) { 
    eta[0] = 1.0 - 0.5 * dm;
    for (int i = 1; i < num_angles; i++)
      eta[i] = eta[i-1] - dm;

    if (num_dims > 2) {
      for (int i = 0; i < num_angles; i++) {
        const double t = mu[i]*mu[i] + eta[i]*eta[i];
        xi[i] = sqrt( 1.0 - t );
      }
    }
  }

  if (num_dims == 1) {
    for (int i = 0; i < num_angles; i++)
      w[i] = 0.5 / double(num_angles);
  } else if (num_dims == 2) {
    for (int i = 0; i < num_angles; i++)
      w[i] = 0.25 / double(num_angles);
  } else if (num_dims == 3) {
    for (int i = 0; i < num_angles; i++)
      w[i] = 0.125 / double(num_angles);
  } else 
    assert(false);

  const size_t ec_size = num_angles * cmom * num_octants * sizeof(double);
  ec = (double*)malloc(ec_size);
  memset(ec, 0, ec_size);

  for (int i = 0; i < 4; i++)
    lma[i] = 0;

  switch (num_dims)
  {
    case 1:
      {
        for (int i = 0; i < num_angles; i++)
          wmu[i] = w[i] * mu[i];
        for (int i = 0; i < 4; i++)
          lma[i] = 1;
        for (int id = 0; id < 2; id++) {
          int is = -1;
          if (id == 1) 
            is = 1;
          for (int idx = 0; idx < num_angles; idx++)
            ec[id * num_angles * num_moments + idx] = 1.0;
          for (int l = 1; l < num_moments; l++)
            for (int idx = 0; idx < num_angles; idx++)
              ec[id * num_angles * num_moments + l * num_angles + idx] = 
                pow(is*mu[idx], double(2*(l+1)-3)); 
        }
        break;
      }
    case 2:
      {
        for (int i = 0; i < num_angles; i++)
          wmu[i] = w[i] * mu[i];
        for (int i = 0; i < num_angles; i++)
          weta[i] = w[i] * eta[i];
        for (int l = 0; l < num_moments; l++)
          lma[l] = l+1;
        for (int jd = 0; jd < 2; jd++) {
          int js = -1;
          if (jd == 1)
            js = 1;
          for (int id = 0; id < 2; id++) {
            int is = -1;
            if (id == 1)
              is = 1;
            int oct = 2*jd + id;
            for (int idx = 0; idx < num_angles; idx++)
              ec[oct * num_angles * num_moments + idx] = 1.0;
            int moment = 1;
            for (int l = 1; l < num_moments; l++) {
              for (int m = 0; m < l; m++) {
                for (int idx = 0; idx < num_angles; idx++)
                  ec[oct * num_angles * num_moments + moment * num_angles + idx] = 
                    pow(is*mu[idx],2*(l+1)-3) * pow(js*eta[idx],m);
                moment++;
              }
            }
          }
        }
        break;
      }
    case 3:
      {
        for (int i = 0; i < num_angles; i++)
          wmu[i] = w[i] * mu[i];
        for (int i = 0; i < num_angles; i++)
          weta[i] = w[i] * eta[i];
        for (int i = 0; i < num_angles; i++)
          wxi[i] = w[i] * xi[i];

        for (int l = 0; l < num_moments; l++)
          lma[l] = 2*(l+1) - 1;

        for (int kd = 0; kd < 2; kd++) {
          int ks = -1;
          if (kd == 1)
            ks = 1;
          for (int jd = 0; jd < 2; jd++) {
            int js = -1;
            if (jd == 1)
              js = 1;
            for (int id = 0; id < 2; id++) {
              int is = -1;
              if (id == 1)
                is = 1;
              int oct = 4 * kd + 2 * jd + id;
              for (int idx = 0; idx < num_angles; idx++)
                ec[oct * num_angles * num_moments + idx] = 1.0;
              int moment = 1;
              for (int l = 1; l < num_moments; l++) {
                for (int m = 0; m < (2*(l+1)-1); m++) {
                  for (int idx = 0; idx < num_angles; idx++)
                    ec[oct * num_angles * num_moments + moment * num_angles + idx] = 
                      pow(is*mu[idx], 2*(l+1)-3) * pow(ks * xi[idx] * js * eta[idx], m);
                  moment++;
                }
              }
            }
          }
        }
        break;
      }
    default:
      assert(false);
  }
}

//------------------------------------------------------------------------------
/*static*/ void Snap::report_arguments(void)
//------------------------------------------------------------------------------
{
  printf("Dimensions: %d\n", num_dims);
  printf("X-Chunks: %d\n", nx_chunks);
  printf("Y-Chunks: %d\n", ny_chunks);
  printf("Z-Chunks: %d\n", nz_chunks);
  printf("nx,ny,nz: %d,%d,%d\n", nx, ny, nz);
  printf("lx,ly,lz: %.8g,%.8g,%.8g\n", lx, ly, lz);
  printf("Moments: %d\n", num_moments);
  printf("Angles: %d\n", num_angles);
  printf("Groups: %d\n", num_groups);
  printf("Convergence: %.8g\n", convergence_eps);
  printf("Max Inner Iterations: %d\n", max_inner_iters);
  printf("Max Outer Iterations: %d\n", max_outer_iters);
  printf("Time Dependent: %s\n", (time_dependent ? "Yes" : "No"));
  printf("Total Simulation Time: %.8g\n", total_sim_time);
  printf("Total Steps: %d\n", num_steps);
  printf("Material Layout: %s\n", ((material_layout == HOMOGENEOUS_LAYOUT) ? 
        "Homogenous Layout" : (material_layout == CENTER_LAYOUT) ? 
        "Center Layout" : "Corner Layout"));
  printf("Source Layout: %s\n", ((source_layout == EVERYWHERE_SOURCE) ? 
        "Everywhere" : (source_layout == CENTER_SOURCE) ? "Center" :
        (source_layout == CORNER_SOURCE) ? "Corner" : "MMS"));
  printf("Dump Scatter: %s\n", dump_scatter ? "Yes" : "No");
  printf("Dump Iteration: %s\n", dump_iteration ? "Yes" : "No");
  printf("Dump Flux: %s\n", (dump_flux == 2) ? "All" : 
      (dump_flux == 1) ? "Scalar" : "No");
  printf("Fixup Flux: %s\n", flux_fixup ? "Yes" : "No");
  printf("Dump Solution: %s\n", dump_solution ? "Yes" : "No");
  printf("Dump Kplane: %d\n", dump_kplane);
  printf("Dump Population: %s\n", (dump_population == 2) ? "All" : 
      (dump_population == 1) ? "Final" : "No");
  printf("Mini-KBA Sweep: %s\n", minikba_sweep ? "Yes" : "No");
  printf("Single Angle Copy: %s\n", single_angle_copy ? "Yes" : "No");
}

//------------------------------------------------------------------------------
/*static*/ void Snap::perform_registrations(void)
//------------------------------------------------------------------------------
{
  TaskVariantRegistrar registrar(SNAP_TOP_LEVEL_TASK_ID, "snap_main_variant");
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  Runtime::preregister_task_variant<snap_top_level_task>(registrar,
                          Snap::task_names[SNAP_TOP_LEVEL_TASK_ID]);
  Runtime::set_top_level_task_id(SNAP_TOP_LEVEL_TASK_ID);
  Runtime::set_registration_callback(mapper_registration);
  // Now register all the task variants
  InitMaterial::preregister_cpu_variants();
  InitSource::preregister_cpu_variants();
#ifdef USE_GPU_KERNELS
  InitGPUSweep::preregister_gpu_variants();
  CalcOuterSource::preregister_all_variants();
  TestOuterConvergence::preregister_all_variants();
  CalcInnerSource::preregister_all_variants();
  TestInnerConvergence::preregister_all_variants();
  MiniKBATask::preregister_all_variants();
  ExpandCrossSection::preregister_all_variants();
  ExpandScatteringCrossSection::preregister_all_variants();
  CalculateGeometryParam::preregister_all_variants();
#else
  CalcOuterSource::preregister_cpu_variants();
  TestOuterConvergence::preregister_cpu_variants();
  CalcInnerSource::preregister_cpu_variants();
  TestInnerConvergence::preregister_cpu_variants();
  MiniKBATask::preregister_cpu_variants();
  ExpandCrossSection::preregister_cpu_variants();
  ExpandScatteringCrossSection::preregister_cpu_variants();
  CalculateGeometryParam::preregister_cpu_variants();
#endif
  MMSInitFlux::preregister_cpu_variants();
  MMSInitSource::preregister_cpu_variants();
  MMSInitTimeDependent::preregister_cpu_variants();
  MMSScale::preregister_cpu_variants();
  MMSCompare::preregister_cpu_variants();
  ConvergenceMonad::preregister_cpu_variants();
  // Register projection functors
  Runtime::preregister_projection_functor(SWEEP_PROJECTION, 
                        new SnapSweepProjectionFunctor());
  Runtime::preregister_projection_functor(XY_PROJECTION,
                        new FluxProjectionFunctor(XY_PROJECTION));
  Runtime::preregister_projection_functor(YZ_PROJECTION,
                        new FluxProjectionFunctor(YZ_PROJECTION));
  Runtime::preregister_projection_functor(XZ_PROJECTION,
                        new FluxProjectionFunctor(XZ_PROJECTION));
  // Finally register our reduction operators
  Runtime::register_reduction_op<AndReduction>(AndReduction::REDOP);
  Runtime::register_reduction_op<SumReduction>(SumReduction::REDOP);
  Runtime::register_reduction_op<TripleReduction>(TripleReduction::REDOP);
  Runtime::register_reduction_op<MMSReduction>(MMSReduction::REDOP);
}

//------------------------------------------------------------------------------
/*static*/ void Snap::mapper_registration(Machine machine, Runtime *runtime,
                                         const std::set<Processor> &local_procs)
//------------------------------------------------------------------------------
{
  MapperRuntime *mapper_rt = runtime->get_mapper_runtime();
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    runtime->replace_default_mapper(new SnapMapper(mapper_rt, machine, 
                                                   *it, "SNAP Mapper"), *it);
  }
}

//------------------------------------------------------------------------------
/*static*/ LayoutConstraintID Snap::get_soa_layout(void)
//------------------------------------------------------------------------------
{
  static LayoutConstraintID layout_id = 0;
  if (layout_id > 0)
    return layout_id;
  // If we haven't made it yet, make it now
  LayoutConstraintRegistrar constraints;
  // This should be a normal instance
  constraints.add_constraint(SpecializedConstraint(NORMAL_SPECIALIZE));
  // Want fortran ordering of dimensions
  std::vector<DimensionKind> dim_order(4);
  dim_order[0] = DIM_X;
  dim_order[1] = DIM_Y;
  dim_order[2] = DIM_Z;
  dim_order[3] = DIM_F; // SOA: fields are least quickly changing
  constraints.add_constraint(OrderingConstraint(dim_order, true/*contiguous*/));
  layout_id = Runtime::preregister_layout(constraints);
  return layout_id;
}

//------------------------------------------------------------------------------
/*static*/ LayoutConstraintID Snap::get_reduction_layout(void)
//------------------------------------------------------------------------------
{
  static LayoutConstraintID layout_id = 0;
  if (layout_id > 0)
    return layout_id;
  // If we haven't made it yet, make it now
  LayoutConstraintRegistrar constraints;
  // This should be a normal instance
  constraints.add_constraint(
      SpecializedConstraint(REDUCTION_FOLD_SPECIALIZE, SumReduction::REDOP));
  // Want fortran ordering of dimensions
  std::vector<DimensionKind> dim_order(4);
  dim_order[0] = DIM_X;
  dim_order[1] = DIM_Y;
  dim_order[2] = DIM_Z;
  dim_order[3] = DIM_F; // SOA: fields are least quickly changing
  constraints.add_constraint(OrderingConstraint(dim_order, true/*contiguous*/));
  layout_id = Runtime::preregister_layout(constraints);
  return layout_id;
}

//------------------------------------------------------------------------------
SnapArray::SnapArray(IndexSpace is, IndexPartition ip, FieldSpace fs,
                     Context c, Runtime *rt, const char *name)
  : ctx(c), runtime(rt)
//------------------------------------------------------------------------------
{
  char name_buffer[64];
  snprintf(name_buffer,63,"%s Logical Region", name);
  lr = runtime->create_logical_region(ctx, is, fs);
  runtime->attach_name(lr, name_buffer);
  if (ip.exists())
  {
    lp = runtime->get_logical_partition(lr, ip);
    snprintf(name_buffer,63,"%s Spatial Partition", name);
    runtime->attach_name(lp, name_buffer);
    // Also get the color space if we have one
    color_space = 
      runtime->get_index_partition_color_space(ctx, lp.get_index_partition());
  }
  runtime->get_field_space_fields(fs, all_fields);
  assert(!all_fields.empty());
  // Assume all the fields are the same size
  field_size = runtime->get_field_size(lr.get_field_space(),
                                       *(all_fields.begin()));
  assert(field_size > 0);
  fill_buffer = malloc(field_size);
  memset(fill_buffer, 0, field_size);
}

//------------------------------------------------------------------------------
SnapArray::SnapArray(const SnapArray &rhs)
  : ctx(rhs.ctx), runtime(rhs.runtime)
//------------------------------------------------------------------------------
{
  // should never be called
  assert(false);
}

//------------------------------------------------------------------------------
SnapArray::~SnapArray(void)
//------------------------------------------------------------------------------
{
  runtime->destroy_logical_region(ctx, lr);
  free(fill_buffer);
}

//------------------------------------------------------------------------------
SnapArray& SnapArray::operator=(const SnapArray &rhs)
//------------------------------------------------------------------------------
{
  // should never be called
  assert(false);
  return *this;
}

//------------------------------------------------------------------------------
LogicalRegion SnapArray::get_subregion(const DomainPoint &color) const
//------------------------------------------------------------------------------
{
  assert(lp.exists());
  // See if we already cached the result
  std::map<DomainPoint,LogicalRegion>::const_iterator finder = 
    subregions.find(color);
  if (finder != subregions.end())
    return finder->second;
  LogicalRegion result = runtime->get_logical_subregion_by_color(lp, color);
  // Save the result for later
  subregions[color] = result;
  return result;
}

//------------------------------------------------------------------------------
void SnapArray::initialize(Predicate pred) const
//------------------------------------------------------------------------------
{
#ifndef NO_INDEX_SPACE_FILLS
  // If we have partition it is better to do an index space fill for scalability
  if (lp.exists())
  {
    IndexFillLauncher launcher(color_space, lp, lr, 
                               TaskArgument(fill_buffer, field_size),
                               0/*identity*/, pred);
    launcher.fields = all_fields;
    runtime->fill_fields(ctx, launcher);
  }
  else
#endif
  {
    FillLauncher launcher(lr, lr, TaskArgument(fill_buffer, field_size), pred);
    launcher.fields = all_fields;
    runtime->fill_fields(ctx, launcher);
  }
}

//------------------------------------------------------------------------------
template<typename T>
void SnapArray::initialize(T value, Predicate pred) const
//------------------------------------------------------------------------------
{
#ifndef NO_INDEX_SPACE_FILLS
  // If we have partition it is better to do an index space fill for scalability
  if (lp.exists())
  {
    IndexFillLauncher launcher(color_space, lp, lr, 
                               TaskArgument(&value, sizeof(value)),
                               0/*identity*/, pred);
    launcher.fields = all_fields;
    runtime->fill_fields(ctx, launcher);
  }
  else
#endif
  {
    FillLauncher launcher(lr, lr, TaskArgument(&value, sizeof(value)), pred);
    launcher.fields = all_fields;
    runtime->fill_fields(ctx, launcher);
  }
}

//------------------------------------------------------------------------------
PhysicalRegion SnapArray::map(void) const
//------------------------------------------------------------------------------
{
  InlineLauncher launcher(RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
  launcher.requirement.privilege_fields = all_fields;
  return runtime->map_region(ctx, launcher);
}

//------------------------------------------------------------------------------
void SnapArray::unmap(const PhysicalRegion &region) const
//------------------------------------------------------------------------------
{
  runtime->unmap_region(ctx, region);
}

//------------------------------------------------------------------------------
SnapSweepProjectionFunctor::SnapSweepProjectionFunctor(void)
  : ProjectionFunctor()
//------------------------------------------------------------------------------
{
}

//------------------------------------------------------------------------------
LogicalRegion SnapSweepProjectionFunctor::project(const Mappable *mappable,
            unsigned index, LogicalRegion upper_bound, const DomainPoint &point)
//------------------------------------------------------------------------------
{
  // should never be called
  assert(false);
  return LogicalRegion::NO_REGION;
}

//------------------------------------------------------------------------------
LogicalRegion SnapSweepProjectionFunctor::project(const Mappable *mappable,
         unsigned index, LogicalPartition upper_bound, const DomainPoint &point)
//------------------------------------------------------------------------------
{
  const Task *task = mappable->as_task();
  assert(task != NULL);
  assert(task->task_id == Snap::MINI_KBA_TASK_ID);
  assert(point.get_dim() == 1);
  Point<1> p = point.get_point<1>();
  // Figure out which wavefront and corner we are in
  unsigned wavefront = ((const MiniKBATask::MiniKBAArgs*)task->args)->wavefront;
  unsigned corner = ((const MiniKBATask::MiniKBAArgs*)task->args)->corner;
  // Not valid, need to go get the result
  LogicalRegion result = runtime->get_logical_subregion_by_color(upper_bound,
                                Snap::wavefront_map[corner][wavefront][p[0]]);
  return result;
}

//------------------------------------------------------------------------------
FluxProjectionFunctor::FluxProjectionFunctor(Snap::SnapProjectionID k)
  : ProjectionFunctor(), projection_kind(k)
//------------------------------------------------------------------------------
{
}

//------------------------------------------------------------------------------
LogicalRegion FluxProjectionFunctor::project(const Mappable *mappble,
            unsigned index, LogicalRegion upper_bound, const DomainPoint &point)
//------------------------------------------------------------------------------
{
  // should never be called
  assert(false);
  return LogicalRegion::NO_REGION;
}

//------------------------------------------------------------------------------
LogicalRegion FluxProjectionFunctor::project(const Mappable *mappable,
         unsigned index, LogicalPartition upper_bound, const DomainPoint &point)
//------------------------------------------------------------------------------
{
  const Task *task = mappable->as_task();
  assert(task != NULL);
  assert(task->task_id == Snap::MINI_KBA_TASK_ID);
  assert(point.get_dim() == 1);
  Point<1> p = point.get_point<1>();
  // Figure out which wavefront and corner we are in
  unsigned wavefront = ((const MiniKBATask::MiniKBAArgs*)task->args)->wavefront;
  unsigned corner = ((const MiniKBATask::MiniKBAArgs*)task->args)->corner;
  // Not in the cache, let's go find it
  Point<3> spatial_point = 
    Snap::wavefront_map[corner][wavefront][p[0]].get_point<3>();
  // Get the right sub-partition by projection down to the proper color
  Point<2> color;
  switch (projection_kind)
  {
    case Snap::XY_PROJECTION:
      {
        color.x[0] = spatial_point.x[0];
        color.x[1] = spatial_point.x[1];
        return runtime->get_logical_subregion_by_color(upper_bound,
                                  DomainPoint::from_point<2>(color));
      }
    case Snap::YZ_PROJECTION:
      {
        color.x[0] = spatial_point.x[1];
        color.x[1] = spatial_point.x[2];
        return runtime->get_logical_subregion_by_color(upper_bound,
                                  DomainPoint::from_point<2>(color));
      }
    case Snap::XZ_PROJECTION:
      {
        color.x[0] = spatial_point.x[0];
        color.x[1] = spatial_point.x[2];
        return runtime->get_logical_subregion_by_color(upper_bound,
                                  DomainPoint::from_point<2>(color));
      }
    default:
      assert(false);
  }
  return LogicalRegion::NO_REGION;
}

//------------------------------------------------------------------------------
template<>
/*static*/ void AndReduction::apply<true>(LHS &lhs, RHS rhs)
//------------------------------------------------------------------------------
{
  // This is monotonic so no need for synchronization either way 
  if (!rhs)
    lhs = false;
}

//------------------------------------------------------------------------------
template<>
/*static*/ void AndReduction::apply<false>(LHS &lhs, RHS rhs)
//------------------------------------------------------------------------------
{
  // This is monotonic so no need for synchronization either way 
  if (!rhs)
    lhs = false;
}

//------------------------------------------------------------------------------
template<>
/*static*/ void AndReduction::fold<true>(RHS &rhs1, RHS rhs2)
//------------------------------------------------------------------------------
{
  // This is monotonic so no need for synchronization either way
  if (!rhs2)
    rhs1 = false;
}

//------------------------------------------------------------------------------
template<>
/*static*/ void AndReduction::fold<false>(RHS &rhs1, RHS rhs2)
//------------------------------------------------------------------------------
{
  // This is monotonic so no need for synchronization either way
  if (!rhs2)
    rhs1 = false;
}

const double SumReduction::identity = 0.0;

//------------------------------------------------------------------------------
template <>
void SumReduction::apply<true>(LHS &lhs, RHS rhs) 
//------------------------------------------------------------------------------
{
  lhs += rhs;
}

//------------------------------------------------------------------------------
template<>
void SumReduction::apply<false>(LHS &lhs, RHS rhs)
//------------------------------------------------------------------------------
{
  volatile long *target = (volatile long *)&lhs;
  union { long as_int; double as_float; } oldval, newval;
  do {
    oldval.as_int = *target;
    newval.as_float = oldval.as_float + rhs;
  } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}

//------------------------------------------------------------------------------
template <>
void SumReduction::fold<true>(RHS &rhs1, RHS rhs2) 
//------------------------------------------------------------------------------
{
  rhs1 += rhs2;
}

//------------------------------------------------------------------------------
template<>
void SumReduction::fold<false>(RHS &rhs1, RHS rhs2)
//------------------------------------------------------------------------------
{
  volatile long *target = (volatile long *)&rhs1;
  union { long as_int; double as_float; } oldval, newval;
  do {
    oldval.as_int = *target;
    newval.as_float = oldval.as_float + rhs2;
  } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}

const MomentTriple TripleReduction::identity = MomentTriple();

//------------------------------------------------------------------------------
template<>
void TripleReduction::apply<true>(LHS &lhs, RHS rhs)
//------------------------------------------------------------------------------
{
  for (int i = 0; i < 3; i++)
    lhs[i] += rhs[i];
}

//------------------------------------------------------------------------------
template<>
void TripleReduction::apply<false>(LHS &lhs, RHS rhs)
//------------------------------------------------------------------------------
{
  for (int i = 0; i < 3; i++)
  {
    volatile long *target = (volatile long *)&lhs[i];
    union { long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float + rhs[i];
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int)); 
  } 
}

//------------------------------------------------------------------------------
template<>
void TripleReduction::fold<true>(RHS &rhs1, RHS rhs2)
//------------------------------------------------------------------------------
{
  for (int i = 0; i < 3; i++)
    rhs1[i] += rhs2[i];
}

//------------------------------------------------------------------------------
template<>
void TripleReduction::fold<false>(RHS &rhs1, RHS rhs2)
//------------------------------------------------------------------------------
{
  for (int i = 0; i < 3; i++)
  {
    volatile long *target = (volatile long *)&rhs1[i];
    union { long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float + rhs2[i];
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int)); 
  }
}


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
#include "sweep.h"
#include "legion_stl.h"

#include <stdlib.h>
#include <x86intrin.h>

extern LegionRuntime::Logger::Category log_snap;

using namespace LegionRuntime::Accessor;
using namespace Legion::STL;

//------------------------------------------------------------------------------
MiniKBATask::MiniKBATask(const Snap &snap, const Predicate &pred,
                         const SnapArray &flux, const SnapArray &fluxm,
                         const SnapArray &qtot, const SnapArray &vdelt, 
                         const SnapArray &dinv, const SnapArray &t_xs,
                         const SnapArray &time_flux_in, 
                         const SnapArray &time_flux_out,
                         const SnapArray &qim, const SnapArray &flux_xy,
                         const SnapArray &flux_yz, const SnapArray &flux_xz,
                         int group_start, int group_stop, int corner, 
                         const int ghost_offsets[3])
  : SnapTask<MiniKBATask, Snap::MINI_KBA_TASK_ID>(
      snap, Rect<3>(Point<3>::ZEROES(), Point<3>::ZEROES()), pred), 
    mini_kba_args(MiniKBAArgs(corner, group_start, group_stop))
//------------------------------------------------------------------------------
{
  global_arg = TaskArgument(&mini_kba_args, sizeof(mini_kba_args));
  if (group_start == group_stop) {
    // Special case for a single field
    const Snap::SnapFieldID group_field = SNAP_ENERGY_GROUP_FIELD(group_start);
    qtot.add_projection_requirement(READ_ONLY, *this, 
                                    group_field, Snap::SWEEP_PROJECTION);
#ifndef SNAP_USE_RELAXED_COHERENCE
    // We need reduction privileges on the flux field since all sweeps
    // will be contributing to it
    flux.add_projection_requirement(*this, Snap::SUM_REDUCTION_ID,
                                    group_field, Snap::SWEEP_PROJECTION);
#else
    flux.add_projection_requirement(READ_WRITE, *this, 
                                    group_field, Snap::SWEEP_PROJECTION);
    region_requirements.back().prop = SIMULTANEOUS;
#endif
    if (Snap::source_layout == Snap::MMS_SOURCE) {
      qim.add_projection_requirement(READ_ONLY, *this,
                                     group_field, Snap::SWEEP_PROJECTION);
#ifndef SNAP_USE_RELAXED_COHERENCE
      fluxm.add_projection_requirement(*this, Snap::TRIPLE_REDUCTION_ID,
                                       group_field, Snap::SWEEP_PROJECTION);
#else
      fluxm.add_projection_requirement(READ_WRITE, *this,
                                       group_field, Snap::SWEEP_PROJECTION);
      region_requirements.back().prop = SIMULTANEOUS;
#endif
    }
    else {
      qim.add_projection_requirement(NO_ACCESS, *this,
                                     group_field, Snap::SWEEP_PROJECTION);
      fluxm.add_projection_requirement(NO_ACCESS, *this, 
                                       group_field, Snap::SWEEP_PROJECTION);
    }
    // Add the dinv array for this field
    dinv.add_projection_requirement(READ_ONLY, *this,
                                    group_field, Snap::SWEEP_PROJECTION);
    time_flux_in.add_projection_requirement(READ_ONLY, *this,
                                            group_field, Snap::SWEEP_PROJECTION);
    time_flux_out.add_projection_requirement(WRITE_DISCARD, *this,
                                             group_field, Snap::SWEEP_PROJECTION);
    t_xs.add_projection_requirement(READ_ONLY, *this,
                                    group_field, Snap::SWEEP_PROJECTION);
    // Now do our ghost requirements
    const Snap::SnapFieldID flux_field = SNAP_FLUX_GROUP_FIELD(group_start, corner);
    flux_xy.add_projection_requirement(READ_WRITE, *this,
                                       flux_field, Snap::XY_PROJECTION);
    flux_yz.add_projection_requirement(READ_WRITE, *this,
                                       flux_field, Snap::YZ_PROJECTION);
    flux_xz.add_projection_requirement(READ_WRITE, *this,
                                       flux_field, Snap::XZ_PROJECTION);
    // This one last since it's not a projection requirement
    vdelt.add_region_requirement(READ_ONLY, *this, group_field);
  } else {
    std::vector<Snap::SnapFieldID> group_fields((group_stop - group_start) + 1);
    for (int group = group_start; group <= group_stop; group++)
      group_fields[group-group_start] = SNAP_ENERGY_GROUP_FIELD(group);
    qtot.add_projection_requirement(READ_ONLY, *this, 
                                    group_fields, Snap::SWEEP_PROJECTION);
#ifndef SNAP_USE_RELAXED_COHERENCE
    // We need reduction privileges on the flux field since all sweeps
    // will be contributing to it
    flux.add_projection_requirement(*this, Snap::SUM_REDUCTION_ID,
                                    group_fields, Snap::SWEEP_PROJECTION);
#else
    flux.add_projection_requirement(READ_WRITE, *this,
                                    group_fields, Snap::SWEEP_PROJECTION);
    region_requirements.back().prop = SIMULTANEOUS;
#endif
    if (Snap::source_layout == Snap::MMS_SOURCE) {
      qim.add_projection_requirement(READ_ONLY, *this,
                                     group_fields, Snap::SWEEP_PROJECTION);
#ifndef SNAP_USE_RELAXED_COHERENCE
      fluxm.add_projection_requirement(*this, Snap::TRIPLE_REDUCTION_ID,
                                       group_fields, Snap::SWEEP_PROJECTION);
#else
      fluxm.add_projection_requirement(READ_WRITE, *this,
                                       group_fields, Snap::SWEEP_PROJECTION);
      region_requirements.back().prop = SIMULTANEOUS;
#endif
    }
    else {
      qim.add_projection_requirement(NO_ACCESS, *this,
                                     group_fields, Snap::SWEEP_PROJECTION);
      fluxm.add_projection_requirement(NO_ACCESS, *this,
                                       group_fields, Snap::SWEEP_PROJECTION);
    }
    // Add the dinv array for this field
    dinv.add_projection_requirement(READ_ONLY, *this,
                                    group_fields, Snap::SWEEP_PROJECTION);
    time_flux_in.add_projection_requirement(READ_ONLY, *this,
                                            group_fields, Snap::SWEEP_PROJECTION);
    time_flux_out.add_projection_requirement(WRITE_DISCARD, *this,
                                             group_fields, Snap::SWEEP_PROJECTION);
    t_xs.add_projection_requirement(READ_ONLY, *this,
                                    group_fields, Snap::SWEEP_PROJECTION);
    // Then do our ghost region requirements
    std::vector<Snap::SnapFieldID> flux_fields((group_stop - group_start) + 1);
    for (int group = group_start; group <= group_stop; group++)
      flux_fields[group-group_start] = SNAP_FLUX_GROUP_FIELD(group, corner);
    flux_xy.add_projection_requirement(READ_WRITE, *this,
                                       flux_fields, Snap::XY_PROJECTION);
    flux_yz.add_projection_requirement(READ_WRITE, *this,
                                       flux_fields, Snap::YZ_PROJECTION);
    flux_xz.add_projection_requirement(READ_WRITE, *this,
                                       flux_fields, Snap::XZ_PROJECTION);
    // This one last since it's not a projection requirement
    vdelt.add_region_requirement(READ_ONLY, *this, group_fields);
  }
}

//------------------------------------------------------------------------------
void MiniKBATask::dispatch_wavefront(int wavefront, const Domain &launch_d,
                                     Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  // Save our wavefront
  this->mini_kba_args.wavefront = wavefront;
  // Set our launch domain
  this->launch_domain = launch_d;
  // Then call the normal dispatch routine
  dispatch(ctx, runtime);
}

//------------------------------------------------------------------------------
/*static*/ void MiniKBATask::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need x86 CPU with SSE or AVX instructions
#ifdef __AVX__
  execution_constraints.add_constraint(ISAConstraint(X86_ISA | AVX_ISA));
#else
  execution_constraints.add_constraint(ISAConstraint(X86_ISA | SSE_ISA));
#endif
  TaskLayoutConstraintSet layout_constraints;
  // Most requirements are normal SOA, the others are reductions
  layout_constraints.add_layout_constraint(0/*index*/,
                                           Snap::get_soa_layout());
#ifndef SNAP_USE_RELAXED_COHERENCE
  layout_constraints.add_layout_constraint(1/*index*/,
                                           Snap::get_reduction_layout());
#else
  layout_constraints.add_layout_constraint(1/*index*/,
                                           Snap::get_soa_layout());
#endif
  layout_constraints.add_layout_constraint(2/*index*/,
                                           Snap::get_soa_layout());
#ifndef SNAP_USE_RELAXED_COHERENCE
  layout_constraints.add_layout_constraint(3/*index*/,
                                           Snap::get_reduction_layout());
#else
  layout_constraints.add_layout_constraint(3/*index*/,
                                           Snap::get_soa_layout());
#endif
  for (unsigned idx = 4; idx < 12; idx++)
    layout_constraints.add_layout_constraint(idx/*index*/,
                                             Snap::get_soa_layout());
#if defined(BOUNDS_CHECKS) || defined(PRIVILEGE_CHECKS)
  register_cpu_variant<cpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
#else
#ifdef __AVX__
  register_cpu_variant<
    raw_rect_task_wrapper<MomentQuad, 3, double, 3, double, 3, MomentTriple, 3,
                       double, 3, double, 3, double, 3, double, 3, double, 2,
                       double, 2, double, 2, double, 1, avx_implementation> >(
              execution_constraints, layout_constraints, true/*leaf*/);
#else
  register_cpu_variant<
    raw_rect_task_wrapper<MomentQuad, 3, double, 3, double, 3, MomentTriple, 3,
                       double, 3, double, 3, double, 3, double, 3, double, 2,
                       double, 2, double, 2, double, 1, sse_implementation> >(
              execution_constraints, layout_constraints, true/*leaf*/);
#endif
#endif
}

//------------------------------------------------------------------------------
/*static*/ void MiniKBATask::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need a CUDA GPU with at least SM 30
  execution_constraints.add_constraint(ISAConstraint(CUDA_ISA | SM_30_ISA));
  TaskLayoutConstraintSet layout_constraints;
  // Most requirements are normal SOA, the others are reductions
  layout_constraints.add_layout_constraint(0/*index*/,
                                           Snap::get_soa_layout());
#ifndef SNAP_USE_RELAXED_COHERENCE
  layout_constraints.add_layout_constraint(1/*index*/,
                                           Snap::get_reduction_layout());
#else
  layout_constraints.add_layout_constraint(1/*index*/,
                                           Snap::get_soa_layout());
#endif
  layout_constraints.add_layout_constraint(2/*index*/,
                                           Snap::get_soa_layout());
#ifndef SNAP_USE_RELAXED_COHERENCE
  layout_constraints.add_layout_constraint(3/*index*/,
                                           Snap::get_reduction_layout());
#else
  layout_constraints.add_layout_constraint(3/*index*/,
                                           Snap::get_soa_layout());
#endif
  for (unsigned idx = 4; idx < 12; idx++)
    layout_constraints.add_layout_constraint(idx/*index*/,
                                             Snap::get_soa_layout());
  register_gpu_variant<
    raw_rect_task_wrapper<MomentQuad, 3, double, 3, double, 3, MomentTriple, 3,
                       double, 3, double, 3, double, 3, double, 3, double, 2,
                       double, 2, double, 2, double, 1, gpu_implementation> >(
              execution_constraints, layout_constraints, true/*leaf*/);
}

static inline Point<2> ghostx_point(const Point<3> &local_point)
{
  Point<2> ghost;
  ghost.x[0] = local_point.x[1]; // y
  ghost.x[1] = local_point.x[2]; // z
  return ghost;
}

static inline Point<2> ghosty_point(const Point<3> &local_point)
{
  Point<2> ghost;
  ghost.x[0] = local_point.x[0]; // x
  ghost.x[1] = local_point.x[2]; // z
  return ghost;
}

static inline Point<2> ghostz_point(const Point<3> &local_point)
{
  Point<2> ghost;
  ghost.x[0] = local_point.x[0]; // x
  ghost.x[1] = local_point.x[1]; // y
  return ghost;
}

//------------------------------------------------------------------------------
/*static*/ void MiniKBATask::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running Mini-KBA Sweep");

  assert(task->arglen == sizeof(MiniKBAArgs));
  const MiniKBAArgs *args = reinterpret_cast<const MiniKBAArgs*>(task->args);
    
  // This implementation of the sweep assumes three dimensions
  assert(Snap::num_dims == 3);

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

  // Figure out the origin point based on which corner we are
  const bool stride_x_positive = ((args->corner & 0x1) != 0);
  const bool stride_y_positive = ((args->corner & 0x2) != 0);
  const bool stride_z_positive = ((args->corner & 0x4) != 0);
  const coord_t origin_ints[3] = { 
    (stride_x_positive ? subgrid_bounds.lo[0] : subgrid_bounds.hi[0]),
    (stride_y_positive ? subgrid_bounds.lo[1] : subgrid_bounds.hi[1]),
    (stride_z_positive ? subgrid_bounds.lo[2] : subgrid_bounds.hi[2]) };
  const Point<3> origin(origin_ints);

  // Local arrays
  const size_t angle_buffer_size = Snap::num_angles * sizeof(double);
  double *psi = (double*)malloc(angle_buffer_size);
  double *pc = (double*)malloc(angle_buffer_size);
  double *psii = (double*)malloc(angle_buffer_size);
  double *psij = (double*)malloc(angle_buffer_size);
  double *psik = (double*)malloc(angle_buffer_size);
  double *time_flux_in = (double*)malloc(angle_buffer_size);
  double *time_flux_out = (double*)malloc(angle_buffer_size);
  double *temp_array = (double*)malloc(angle_buffer_size);
  double *hv_x = (double*)malloc(angle_buffer_size);
  double *hv_y = (double*)malloc(angle_buffer_size);
  double *hv_z = (double*)malloc(angle_buffer_size);
  double *hv_t = (double*)malloc(angle_buffer_size);
  double *fx_hv_x = (double*)malloc(angle_buffer_size);
  double *fx_hv_y = (double*)malloc(angle_buffer_size);
  double *fx_hv_z = (double*)malloc(angle_buffer_size);
  double *fx_hv_t = (double*)malloc(angle_buffer_size);

  const double tolr = 1.0e-12;

  // There's no parallelism here, better to walk pencils in one direction
  // and try and maintain some locality. Hopefully the 2-D slices will 
  // remain in the last level cache
  // Assume y_range 16 and z_range 16
  // 2K angles * 8 bytes * 16 * 16 = 4MB
  // This is maybe not as good from a blocking standpoint (maybe better
  // to block for L2, something like 2x2x2 blocks), but it will result 
  // in linear strides through memory which will be better for the 
  // prefetchers and likely result in overall better performance
  // because the very small 2x2x2 size will be too small to warm up
  // the prefetchers and they will be confused by the access pattern
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1; 
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;
  double *yflux_pencil = (double*)malloc(x_range * angle_buffer_size);
  double *zflux_plane  = (double*)malloc(y_range * x_range * angle_buffer_size);

  // We could abstract these things into functions, but C++ compilers
  // get angsty about pointers and marking everything with restrict
  // is super annoying so just do all the inlining for the compiler
  for (int group = args->group_start; group <= args->group_stop; group++) {
    // Get all the accessors for this energy group
    RegionAccessor<AccessorType::Generic,MomentQuad> fa_qtot = 
      regions[0].get_field_accessor(
          SNAP_ENERGY_GROUP_FIELD(group)).typeify<MomentQuad>();
    RegionAccessor<AccessorType::Generic,double> fa_flux = 
      regions[1].get_field_accessor(
          SNAP_ENERGY_GROUP_FIELD(group)).typeify<double>();

    // Reduction accessors don't support structured points yet
    ByteOffset flux_offsets[3], fluxm_offsets[3];
    Rect<3> temp_bounds; 
    double *const flux = 
      fa_flux.raw_rect_ptr<3>(subgrid_bounds, temp_bounds, flux_offsets);
    assert(temp_bounds == subgrid_bounds);
    MomentTriple *fluxm = NULL;
    RegionAccessor<AccessorType::Generic> fa_qim;
    if (Snap::source_layout == Snap::MMS_SOURCE) {
      fa_qim = regions[2].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(group));
      RegionAccessor<AccessorType::Generic,MomentTriple> fa_fluxm = 
        regions[3].get_field_accessor(
            SNAP_ENERGY_GROUP_FIELD(group)).typeify<MomentTriple>();
      fluxm = fa_fluxm.raw_rect_ptr<3>(subgrid_bounds, temp_bounds, fluxm_offsets);
      assert(temp_bounds == subgrid_bounds);
    }
    
    // No types here since the size of these fields are dependent
    // on the number of angles
    RegionAccessor<AccessorType::Generic> fa_dinv = 
      regions[4].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(group));
    RegionAccessor<AccessorType::Generic> fa_time_flux_in = 
      regions[5].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(group));
    RegionAccessor<AccessorType::Generic> fa_time_flux_out = 
      regions[6].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(group));
    RegionAccessor<AccessorType::Generic,double> fa_t_xs = 
      regions[7].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(group)).typeify<double>();

    // Ghost regions
    RegionAccessor<AccessorType::Generic> fa_ghostz = 
      regions[8].get_field_accessor(SNAP_FLUX_GROUP_FIELD(group, args->corner));
    RegionAccessor<AccessorType::Generic> fa_ghostx = 
      regions[9].get_field_accessor(SNAP_FLUX_GROUP_FIELD(group, args->corner));
    RegionAccessor<AccessorType::Generic> fa_ghosty = 
      regions[10].get_field_accessor(SNAP_FLUX_GROUP_FIELD(group, args->corner));

    const double vdelt = regions[11].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(group)).typeify<double>().read(
        DomainPoint::from_point<1>(Point<1>::ZEROES()));
    
    // Now we do the sweeps over the points
    for (int z = 0; z < z_range; z++) {
      for (int y = 0; y < y_range; y++) {
        for (int x = 0; x < x_range; x++) {
          // Figure out the local point that we are working on    
          Point<3> local_point = origin;
          if (stride_x_positive)
            local_point.x[0] += x;
          else
            local_point.x[0] -= x;
          if (stride_y_positive)
            local_point.x[1] += y;
          else
            local_point.x[1] -= y;
          if (stride_z_positive)
            local_point.x[2] += z;
          else
            local_point.x[2] -= z;

          // Compute the angular source
          const DomainPoint dp = DomainPoint::from_point<3>(local_point);
          const MomentQuad quad = fa_qtot.read(dp);
          for (int ang = 0; ang < Snap::num_angles; ang++)
            psi[ang] = quad[0];
          if (Snap::num_moments > 1) {
            const int corner_offset = 
              args->corner * Snap::num_angles * Snap::num_moments;
            for (unsigned l = 1; 1 < Snap::num_moments; l++) {
              const int moment_offset = corner_offset + l * Snap::num_angles;
              for (int ang = 0; ang < Snap::num_angles; ang++) {
                psi[ang] += Snap::ec[moment_offset+ang] * quad[l];
              }
            }
          }

          // If we're doing MMS, there is an additional term
          if (Snap::source_layout == Snap::MMS_SOURCE)
          {
            fa_qim.read_untyped(DomainPoint::from_point<3>(local_point),
                                temp_array, angle_buffer_size);
            for (int ang = 0; ang < Snap::num_angles; ang++)
              psi[ang] += temp_array[ang];
          }

          // Compute the initial solution
          for (int ang = 0; ang < Snap::num_angles; ang++)
            pc[ang] = psi[ang];
          // X ghost cells
          if (x == 0) {
            // Ghost cell array
            Point<2> ghost_point = ghostx_point(local_point);
            fa_ghostx.read_untyped(DomainPoint::from_point<2>(ghost_point),   
                                   psii, angle_buffer_size);
          } // Else nothing: psii already contains next flux
          for (int ang = 0; ang < Snap::num_angles; ang++)
            pc[ang] += psii[ang] * Snap::mu[ang] * Snap::hi;
          // Y ghost cells
          if (y == 0) {
            // Ghost cell array
            Point<2> ghost_point = ghosty_point(local_point);
            fa_ghosty.read_untyped(DomainPoint::from_point<2>(ghost_point), 
                                   psij, angle_buffer_size);
          } else {
            // Local array
            const int offset = x * Snap::num_angles;
            memcpy(psij, yflux_pencil+offset, angle_buffer_size);
          }
          for (int ang = 0; ang < Snap::num_angles; ang++)
            pc[ang] += psij[ang] * Snap::eta[ang] * Snap::hj;
          // Z ghost cells
          if (z == 0) {
            // Ghost cell array
            Point<2> ghost_point = ghostz_point(local_point);
            fa_ghostz.read_untyped(DomainPoint::from_point<2>(ghost_point), 
                                   psik, angle_buffer_size);
          } else {
            // Local array
            const int offset = (y * x_range + x) * Snap::num_angles;
            memcpy(psik, zflux_plane+offset, angle_buffer_size);
          }
          for (int ang = 0; ang < Snap::num_angles; ang++)
            pc[ang] += psik[ang] * Snap::xi[ang] * Snap::hk;

          // See if we're doing anything time dependent
          if (vdelt != 0.0) 
          {
            fa_time_flux_in.read_untyped(DomainPoint::from_point<3>(local_point),
                                         time_flux_in, angle_buffer_size);
            for (int ang = 0; ang < Snap::num_angles; ang++)
              pc[ang] += vdelt * time_flux_in[ang];
          }
          // Multiple by the precomputed denominator inverse
          fa_dinv.read_untyped(DomainPoint::from_point<3>(local_point),
                               temp_array, angle_buffer_size);
          for (int ang = 0; ang < Snap::num_angles; ang++)
            pc[ang] *= temp_array[ang];

          if (Snap::flux_fixup) {
            // DO THE FIXUP
            unsigned old_negative_fluxes = 0;
            for (int ang = 0; ang < Snap::num_angles; ang++)
              hv_x[ang] = 1.0;
            for (int ang = 0; ang < Snap::num_angles; ang++)
              hv_y[ang] = 1.0;
            for (int ang = 0; ang < Snap::num_angles; ang++)
              hv_z[ang] = 1.0;
            for (int ang = 0; ang < Snap::num_angles; ang++)
              hv_t[ang] = 1.0;
            const double t_xs = fa_t_xs.read(DomainPoint::from_point<3>(local_point));
            while (true) {
              unsigned negative_fluxes = 0;
              // Figure out how many negative fluxes we have
              for (int ang = 0; ang < Snap::num_angles; ang++) {
                fx_hv_x[ang] = 2.0 * pc[ang] - psii[ang];
                if (fx_hv_x[ang] < 0.0) {
                  hv_x[ang] = 0.0;
                  negative_fluxes++;
                }
              }
              for (int ang = 0; ang < Snap::num_angles; ang++) {
                fx_hv_y[ang] = 2.0 * pc[ang] - psij[ang];
                if (fx_hv_y[ang] < 0.0) {
                  hv_y[ang] = 0.0;
                  negative_fluxes++;
                }
              }
              for (int ang = 0; ang < Snap::num_angles; ang++) {
                fx_hv_z[ang] = 2.0 * pc[ang] - psik[ang];
                if (fx_hv_z[ang] < 0.0) {
                  hv_z[ang] = 0.0;
                  negative_fluxes++;
                }
              }
              if (vdelt != 0.0) {
                for (int ang = 0; ang < Snap::num_angles; ang++) {
                  fx_hv_t[ang] = 2.0 * pc[ang] - time_flux_in[ang];
                  if (fx_hv_t[ang] < 0.0) {
                    hv_t[ang] = 0.0;
                    negative_fluxes++;
                  }
                }
              }
              if (negative_fluxes == old_negative_fluxes)
                break;
              old_negative_fluxes = negative_fluxes; 
              if (vdelt != 0.0) {
                for (int ang = 0; ang < Snap::num_angles; ang++) {
                  pc[ang] = psi[ang] + 0.5 * (
                      psii[ang] * Snap::mu[ang] * Snap::hi * (1.0 + hv_x[ang]) + 
                      psij[ang] * Snap::eta[ang] * Snap::hj * (1.0 + hv_y[ang]) + 
                      psik[ang] * Snap::xi[ang] * Snap::hk * (1.0 + hv_z[ang]) + 
                      time_flux_in[ang] * vdelt * (1.0 + hv_t[ang]) );
                  double den = (pc[ang] <= 0.0) ? 0.0 : (t_xs + 
                    Snap::mu[ang] * Snap::hi * hv_x[ang] + 
                    Snap::eta[ang] * Snap::hj * hv_y[ang] +
                    Snap::xi[ang] * Snap::hk * hv_z[ang] + vdelt * hv_t[ang]);
                  if (den < tolr)
                    pc[ang] = 0.0;
                  else
                    pc[ang] /= den;
                }
              } else {
                for (int ang = 0; ang < Snap::num_angles; ang++) {
                  pc[ang] = psi[ang] + 0.5 * (
                      psii[ang] * Snap::mu[ang] * Snap::hi * (1.0 + hv_x[ang]) + 
                      psij[ang] * Snap::eta[ang] * Snap::hj * (1.0 + hv_y[ang]) +
                      psik[ang] * Snap::xi[ang] * Snap::hk * (1.0 + hv_z[ang]) );
                  double den = (pc[ang] <= 0.0) ? 0.0 : (t_xs + 
                    Snap::mu[ang] * Snap::hi * hv_x[ang] + 
                    Snap::eta[ang] * Snap::hj * hv_y[ang] +
                    Snap::xi[ang] * Snap::hk * hv_z[ang]);
                  if (den < tolr)
                    pc[ang] = 0.0;
                  else
                    pc[ang] /= den;
                }
              }
            }
            // Fixup done so compute the updated values
            for (int ang = 0; ang < Snap::num_angles; ang++)
              psii[ang] = fx_hv_x[ang] * hv_x[ang];
            for (int ang = 0; ang < Snap::num_angles; ang++)
              psij[ang] = fx_hv_y[ang] * hv_y[ang];
            for (int ang = 0; ang < Snap::num_angles; ang++)
              psik[ang] = fx_hv_z[ang] * hv_z[ang];
            if (vdelt != 0.0)
            {
              for (int ang = 0; ang < Snap::num_angles; ang++)
                time_flux_out[ang] = fx_hv_t[ang] * hv_t[ang];
              fa_time_flux_out.write_untyped(DomainPoint::from_point<3>(local_point),
                                             time_flux_out, angle_buffer_size);
            }
          } else {
            // NO FIXUP
            for (int ang = 0; ang < Snap::num_angles; ang++)
              psii[ang] = 2.0 * pc[ang] - psii[ang]; 
            for (int ang = 0; ang < Snap::num_angles; ang++)
              psij[ang] = 2.0 * pc[ang] - psij[ang];
            for (int ang = 0; ang < Snap::num_angles; ang++)
              psik[ang] = 2.0 * pc[ang] - psik[ang];
            if (vdelt != 0.0) 
            {
              // Write out the outgoing temporal flux
              for (int ang = 0; ang < Snap::num_angles; ang++)
                time_flux_out[ang] = 2.0 * pc[ang] - time_flux_in[ang];
              fa_time_flux_out.write_untyped(DomainPoint::from_point<3>(local_point),
                                             time_flux_out, angle_buffer_size);
            }
          }

          // Write out the ghost regions 
          // X ghost
          if (x == (Snap::nx_per_chunk-1)) {
            Point<2> ghost_point = ghostx_point(local_point);
            // We write out on our own region
            fa_ghostx.write_untyped(DomainPoint::from_point<2>(ghost_point),
                                    psii, angle_buffer_size);
          } // Else nothing: psii just gets caried over to next iteration
          // Y ghost
          if (y == (Snap::ny_per_chunk-1)) {
            Point<2> ghost_point = ghosty_point(local_point);
            // Write out on our own region
            fa_ghosty.write_untyped(DomainPoint::from_point<2>(ghost_point),
                                    psij, angle_buffer_size);
          } else {
            // Write to the pencil 
            const int offset = x * Snap::num_angles;
            memcpy(yflux_pencil+offset, psij, angle_buffer_size);
          }
          // Z ghost
          if (z == (Snap::nz_per_chunk-1)) {
            Point<2> ghost_point = ghostz_point(local_point);
            // Write out on our own region
            fa_ghostz.write_untyped(DomainPoint::from_point<2>(ghost_point),
                                    psik, angle_buffer_size);
          } else {
            // Write to the plane
            const int offset = (y * x_range + x) * Snap::num_angles;
            memcpy(zflux_plane+offset, psik, angle_buffer_size);
          }

          // Finally we apply reductions to the flux moments
          double total = 0.0;
          for (int ang = 0; ang < Snap::num_angles; ang++) {
            psi[ang] = Snap::w[ang] * pc[ang]; 
            total += psi[ang];
          }
          double *local_flux = flux;
          // Have to convert to local coordinates for raw pointer
          const Point<3> local_flux_point = local_point - subgrid_bounds.lo;
          for (int i = 0; i < 3; i++)
            local_flux += local_flux_point[i] * flux_offsets[i];
#ifndef SNAP_USE_RELAXED_COHERENCE
          SumReduction::fold<true/*exclusive*/>(*local_flux, total);
#else
          SumReduction::apply<false/*exclusive*/>(*local_flux, total);
#endif
          if (Snap::num_moments > 1) {
            MomentTriple triple;
            for (int l = 1; l < Snap::num_moments; l++) {
              unsigned offset = l * Snap::num_angles + 
                args->corner * Snap::num_angles * Snap::num_moments;
              total = 0.0;
              for (int ang = 0; ang < Snap::num_angles; ang++) {
                total += Snap::ec[offset+ang] * psi[ang]; 
              }
              triple[l-1] = total;
            }
            MomentTriple *local_fluxm = fluxm;
            for (int i = 0; i < 3; i++)
              local_fluxm += local_flux_point[i] * fluxm_offsets[i];
#ifndef SNAP_USE_RELAXED_COHERENCE
            TripleReduction::fold<true/*exclusive*/>(*local_fluxm, triple);
#else
            TripleReduction::apply<false/*exclusive*/>(*local_fluxm, triple);
#endif
          }
        }
      }
    }
  }

  free(psi);
  free(pc);
  free(psii);
  free(psij);
  free(psik);
  free(time_flux_in);
  free(time_flux_out);
  free(temp_array);
  free(hv_x);
  free(hv_y);
  free(hv_z);
  free(hv_t);
  free(fx_hv_x);
  free(fx_hv_y);
  free(fx_hv_z);
  free(fx_hv_t);
  free(yflux_pencil);
  free(zflux_plane);
#endif
}

inline ByteOffset operator*(const ByteOffset offsets[2], const Point<2> &point)
{
  return (offsets[0] * point.x[0] + offsets[1] * point.x[1]);
}

inline ByteOffset operator*(const ByteOffset offsets[3], const Point<3> &point)
{
  return (offsets[0] * point.x[0] + offsets[1] * point.x[1] + offsets[2] * point.x[2]);
}

template<int DIM>
inline __m128d* get_sse_angle_ptr(void *ptr, const ByteOffset offsets[DIM],
                                  const Point<DIM> &point)
{
  return (__m128d*)(ptr + (offsets * point));
}

//------------------------------------------------------------------------------
/*static*/ void MiniKBATask::sse_implementation(
  const Task *task, Context ctx, Runtime *runtime,
  const std::vector<MomentQuad*> &qtot_ptrs, const ByteOffset qtot_offsets[3],
  const std::vector<double*> &flux_ptrs, const ByteOffset flux_offsets[3],
  const std::vector<double*> &qim_ptrs, const ByteOffset qim_offsets[3],
  const std::vector<MomentTriple*> &fluxm_ptrs, const ByteOffset fluxm_offsets[3],
  const std::vector<double*> &dinv_ptrs, const ByteOffset dinv_offsets[3],
  const std::vector<double*> &time_flux_in_ptrs, const ByteOffset time_flux_in_offsets[3],
  const std::vector<double*> &time_flux_out_ptrs, const ByteOffset time_flux_out_offsets[3],
  const std::vector<double*> &t_xs_ptrs, const ByteOffset t_xs_offsets[3],
  const std::vector<double*> &ghost_z_ptrs, const ByteOffset ghostz_offsets[2],
  const std::vector<double*> &ghost_x_ptrs, const ByteOffset ghostx_offsets[2],
  const std::vector<double*> &ghost_y_ptrs, const ByteOffset ghosty_offsets[2],
  const std::vector<double*> &vdelt_ptrs, const ByteOffset vdelt_offsets[1])
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running SSE Mini-KBA Sweep");

  assert(task->arglen == sizeof(MiniKBAArgs));
  const MiniKBAArgs *args = reinterpret_cast<const MiniKBAArgs*>(task->args);
    
  // This implementation of the sweep assumes three dimensions
  assert(Snap::num_dims == 3); 

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

  // Figure out the origin point based on which corner we are
  const bool stride_x_positive = ((args->corner & 0x1) != 0);
  const bool stride_y_positive = ((args->corner & 0x2) != 0);
  const bool stride_z_positive = ((args->corner & 0x4) != 0);
  // Convert to local coordinates
  const coord_t origin_ints[3] = { 
    (stride_x_positive ? subgrid_bounds.lo[0] : subgrid_bounds.hi[0]) - subgrid_bounds.lo[0],
    (stride_y_positive ? subgrid_bounds.lo[1] : subgrid_bounds.hi[1]) - subgrid_bounds.lo[1],
    (stride_z_positive ? subgrid_bounds.lo[2] : subgrid_bounds.hi[2]) - subgrid_bounds.lo[2]};
  const Point<3> origin(origin_ints);

  // Local arrays
  assert((Snap::num_angles % 2) == 0);
  const int num_vec_angles = Snap::num_angles/2;
  const size_t angle_buffer_size = num_vec_angles * sizeof(__m128d);
  __m128d *__restrict__ psi = (__m128d*)malloc(angle_buffer_size);
  __m128d *__restrict__ pc = (__m128d*)malloc(angle_buffer_size);
  __m128d *__restrict__ psii = (__m128d*)malloc(angle_buffer_size);
  __m128d *__restrict__ hv_x = (__m128d*)malloc(angle_buffer_size);
  __m128d *__restrict__ hv_y = (__m128d*)malloc(angle_buffer_size);
  __m128d *__restrict__ hv_z = (__m128d*)malloc(angle_buffer_size);
  __m128d *__restrict__ hv_t = (__m128d*)malloc(angle_buffer_size);
  __m128d *__restrict__ fx_hv_x = (__m128d*)malloc(angle_buffer_size);
  __m128d *__restrict__ fx_hv_y = (__m128d*)malloc(angle_buffer_size);
  __m128d *__restrict__ fx_hv_z = (__m128d*)malloc(angle_buffer_size);
  __m128d *__restrict__ fx_hv_t = (__m128d*)malloc(angle_buffer_size);

  const __m128d tolr = _mm_set1_pd(1.0e-12);

  // See note in the CPU implementation about why we do things this way
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1; 
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;
  __m128d *yflux_pencil = (__m128d*)malloc(x_range * angle_buffer_size);
  __m128d *zflux_plane  = (__m128d*)malloc(y_range * x_range * angle_buffer_size);

  for (unsigned group = 0; group < flux_ptrs.size(); group++) {
    const MomentQuad *const qtot_ptr = qtot_ptrs[group];
    double *const flux = flux_ptrs[group];

    void *const qim_ptr = qim_ptrs[group];
    MomentTriple *const fluxm = fluxm_ptrs[group];
 
    // No types here since the size of these fields are dependent
    // on the number of angles
    void *const dinv_ptr = dinv_ptrs[group];
    void *const time_flux_in_ptr = time_flux_in_ptrs[group]; 
    void *const time_flux_out_ptr = time_flux_out_ptrs[group]; 
    const double *const t_xs_ptr = t_xs_ptrs[group];

    // Ghost regions
    void *const ghostx_ptr = ghost_x_ptrs[group];
    void *const ghosty_ptr = ghost_y_ptrs[group];
    void *const ghostz_ptr = ghost_z_ptrs[group];

    const double vdelt = *(vdelt_ptrs[group]);

    for (int z = 0; z < z_range; z++) {
      for (int y = 0; y < y_range; y++) {
        for (int x = 0; x < x_range; x++) {
          // Figure out the local point that we are working on    
          Point<3> local_point = origin;
          if (stride_x_positive)
            local_point.x[0] += x;
          else
            local_point.x[0] -= x;
          if (stride_y_positive)
            local_point.x[1] += y;
          else
            local_point.x[1] -= y;
          if (stride_z_positive)
            local_point.x[2] += z;
          else
            local_point.x[2] -= z;

          // Compute the angular source
          MomentQuad quad = *(qtot_ptr + qtot_offsets * local_point);
          for (int ang = 0; ang < num_vec_angles; ang++)
            psi[ang] = _mm_set1_pd(quad[0]);
          if (Snap::num_moments > 1) {
            const int corner_offset = 
              args->corner * Snap::num_angles * Snap::num_moments;
            for (int l = 1; l < Snap::num_moments; l++) {
              const int moment_offset = corner_offset + l * Snap::num_angles;
              for (int ang = 0; ang < num_vec_angles; ang++) {
                psi[ang] = _mm_add_pd(psi[ang], _mm_mul_pd(
                      _mm_set_pd(Snap::ec[moment_offset+2*ang+1],
                                 Snap::ec[moment_offset+2*ang]),
                      _mm_set1_pd(quad[l])));
              }
            }
          }

          // If we're doing MMS, there is an additional term
          if (Snap::source_layout == Snap::MMS_SOURCE)
          {
            __m128d *__restrict__ qim = 
              (__m128d*)(qim_ptr + qim_offsets * local_point);
            for (int ang = 0; ang < num_vec_angles; ang++)
              psi[ang] = _mm_add_pd(psi[ang], qim[ang]);
          }

          // Compute the initial solution
          for (int ang = 0; ang < num_vec_angles; ang++)
            pc[ang] = psi[ang];
          // X ghost cells
          if (x == 0) {
            // Ghost cell array
            Point<2> ghost_point = ghostx_point(local_point);
            memcpy(psii, get_sse_angle_ptr<2>(ghostx_ptr, ghostx_offsets,
                                              ghost_point), angle_buffer_size);
          } // Else nothing: psii already contains next flux
          for (int ang = 0; ang < num_vec_angles; ang++)
            pc[ang] = _mm_add_pd(pc[ang], _mm_mul_pd( _mm_mul_pd(psii[ang], 
                    _mm_set_pd(Snap::mu[2*ang+1],Snap::mu[2*ang])), 
                    _mm_set1_pd(Snap::hi)));
          // Y ghost cells
          __m128d *__restrict__ psij = yflux_pencil + x * num_vec_angles;
          if (y == 0) {
            // Ghost cell array
            Point<2> ghost_point = ghosty_point(local_point);
            memcpy(psij, get_sse_angle_ptr<2>(ghosty_ptr, ghosty_offsets, 
                                              ghost_point), angle_buffer_size);
          } // Else nothing: psij already points at the flux
          for (int ang = 0; ang < num_vec_angles; ang++)
            pc[ang] = _mm_add_pd(pc[ang], _mm_mul_pd( _mm_mul_pd(psij[ang],
                    _mm_set_pd(Snap::eta[2*ang+1], Snap::eta[2*ang])),
                    _mm_set1_pd(Snap::hj)));
          // Z ghost cells
          __m128d *__restrict__ psik = zflux_plane + (y * x_range + x) * num_vec_angles;
          if (z == 0) {
            // Ghost cell array
            Point<2> ghost_point = ghostz_point(local_point);
            memcpy(psik, get_sse_angle_ptr<2>(ghostz_ptr, ghostz_offsets, 
                                              ghost_point), angle_buffer_size);
          } // Else nothing: psik already points at the flux
          for (int ang = 0; ang < num_vec_angles; ang++)
            pc[ang] = _mm_add_pd(pc[ang], _mm_mul_pd( _mm_mul_pd(psik[ang],
                    _mm_set_pd(Snap::xi[2*ang+1], Snap::xi[2*ang])),
                    _mm_set1_pd(Snap::hk)));
          // See if we're doing anything time dependent
          __m128d *__restrict__ time_flux_in = get_sse_angle_ptr<3>(time_flux_in_ptr,
                                                  time_flux_in_offsets, local_point);
          if (vdelt != 0.0) 
          {
            for (int ang = 0; ang < num_vec_angles; ang++)
              pc[ang] = _mm_add_pd(pc[ang], _mm_mul_pd(
                    _mm_set1_pd(vdelt), time_flux_in[ang]));
          }
          // Multiple by the precomputed denominator inverse
          __m128d *__restrict__ dinv = 
            get_sse_angle_ptr<3>(dinv_ptr, dinv_offsets, local_point);
          for (int ang = 0; ang < num_vec_angles; ang++)
            pc[ang] = _mm_mul_pd(pc[ang], dinv[ang]);
          if (Snap::flux_fixup) {
            // DO THE FIXUP
            unsigned old_negative_fluxes = 0;
            for (int ang = 0; ang < num_vec_angles; ang++)
              hv_x[ang] = _mm_set1_pd(1.0);
            for (int ang = 0; ang < num_vec_angles; ang++)
              hv_y[ang] = _mm_set1_pd(1.0);
            for (int ang = 0; ang < num_vec_angles; ang++)
              hv_z[ang] = _mm_set1_pd(1.0);
            for (int ang = 0; ang < num_vec_angles; ang++)
              hv_t[ang] = _mm_set1_pd(1.0);

            const double t_xs = *(t_xs_ptr + t_xs_offsets * local_point);
            while (true) {
              unsigned negative_fluxes = 0;
              // Figure out how many negative fluxes we have
              for (int ang = 0; ang < num_vec_angles; ang++) {
                fx_hv_x[ang] = _mm_sub_pd( _mm_mul_pd( _mm_set1_pd(2.0), pc[ang]), psii[ang]);
                __m128d ge = _mm_cmpge_pd(fx_hv_x[ang], _mm_set1_pd(0.0));
                // If not greater than or equal, set back to zero
                hv_x[ang] = _mm_and_pd(ge, hv_x[ang]);
                // Count how many negative fluxes we had
                __m128i negatives = _mm_andnot_si128(_mm_castpd_si128(ge), 
                                                     _mm_set_epi32(0, 1, 0, 1));
                negative_fluxes += _mm_extract_epi32(negatives, 0);
                negative_fluxes += _mm_extract_epi32(negatives, 2);
              }
              for (int ang = 0; ang < num_vec_angles; ang++) {
                fx_hv_y[ang] = _mm_sub_pd( _mm_mul_pd( _mm_set1_pd(2.0), pc[ang]), psij[ang]);
                __m128d ge = _mm_cmpge_pd(fx_hv_y[ang], _mm_set1_pd(0.0));
                // If not greater than or equal set back to zero
                hv_y[ang] = _mm_and_pd(ge, hv_y[ang]);
                // Count how many negative fluxes we had
                __m128i negatives = _mm_andnot_si128(_mm_castpd_si128(ge), 
                                                     _mm_set_epi32(0, 1, 0, 1));
                negative_fluxes += _mm_extract_epi32(negatives, 0);
                negative_fluxes += _mm_extract_epi32(negatives, 2);
              }
              for (int ang = 0; ang < num_vec_angles; ang++) {
                fx_hv_z[ang] = _mm_sub_pd( _mm_mul_pd( _mm_set1_pd(2.0), pc[ang]), psik[ang]);
                __m128d ge = _mm_cmpge_pd(fx_hv_z[ang], _mm_set1_pd(0.0));
                // If not greater than or equal set back to zero
                hv_z[ang] = _mm_and_pd(ge, hv_z[ang]);
                // Count how many negative fluxes we had
                __m128i negatives = _mm_andnot_si128(_mm_castpd_si128(ge), 
                                                     _mm_set_epi32(0, 1, 0, 1));
                negative_fluxes += _mm_extract_epi32(negatives, 0);
                negative_fluxes += _mm_extract_epi32(negatives, 2);
              }
              if (vdelt != 0.0) {
                for (int ang = 0; ang < num_vec_angles; ang++) {
                  fx_hv_t[ang] = _mm_sub_pd( _mm_mul_pd( _mm_set1_pd(2.0), pc[ang]), 
                                                        time_flux_in[ang]);
                  __m128d ge = _mm_cmpge_pd(fx_hv_t[ang], _mm_set1_pd(0.0));
                  // If not greater than or equal, set back to zero
                  hv_t[ang] = _mm_and_pd(ge, hv_t[ang]);
                  // Count how many negative fluxes we had
                  __m128i negatives = _mm_andnot_si128(_mm_castpd_si128(ge), 
                                                       _mm_set_epi32(0, 1, 0, 1));
                  negative_fluxes += _mm_extract_epi32(negatives, 0);
                  negative_fluxes += _mm_extract_epi32(negatives, 2);
                }
              }
              if (negative_fluxes == old_negative_fluxes)
                break;
              old_negative_fluxes = negative_fluxes;
              if (vdelt != 0.0) {
                for (int ang = 0; ang < num_vec_angles; ang++) {
                  __m128d sum = _mm_mul_pd(psii[ang], _mm_mul_pd(
                        _mm_set_pd(Snap::mu[2*ang+1], Snap::mu[2*ang]), 
                        _mm_mul_pd( _mm_set1_pd(Snap::hi), 
                          _mm_add_pd( _mm_set1_pd(1.0), hv_x[ang]))));
                  sum = _mm_add_pd(sum, _mm_mul_pd(psij[ang], _mm_mul_pd(
                          _mm_set_pd(Snap::eta[2*ang+1], Snap::eta[2*ang]),
                          _mm_mul_pd( _mm_set1_pd(Snap::hj),
                            _mm_add_pd( _mm_set1_pd(1.0), hv_y[ang])))));
                  sum = _mm_add_pd(sum, _mm_mul_pd(psik[ang], _mm_mul_pd(
                          _mm_set_pd(Snap::xi[2*ang+1], Snap::xi[2*ang]),
                          _mm_mul_pd( _mm_set1_pd(Snap::hk),
                            _mm_add_pd( _mm_set1_pd(1.0), hv_z[ang])))));
                  sum = _mm_add_pd(sum, _mm_mul_pd(time_flux_in[ang], 
                        _mm_mul_pd( _mm_set1_pd(vdelt), _mm_add_pd(
                            _mm_set1_pd(1.0), hv_t[ang]))));
                  pc[ang] = _mm_add_pd(psi[ang], _mm_mul_pd( _mm_set1_pd(0.5), sum));
                  __m128d den = _mm_add_pd(_mm_set1_pd(t_xs), 
                      _mm_add_pd( _mm_add_pd( _mm_add_pd(
                        _mm_mul_pd( _mm_mul_pd( _mm_set_pd(Snap::mu[2*ang+1], 
                              Snap::mu[2*ang]), _mm_set1_pd(Snap::hi)), hv_x[ang]),
                        _mm_mul_pd( _mm_mul_pd( _mm_set_pd(Snap::eta[2*ang+1],
                              Snap::eta[2*ang]), _mm_set1_pd(Snap::hj)), hv_y[ang])),
                        _mm_mul_pd( _mm_mul_pd( _mm_set_pd(Snap::xi[2*ang+1],
                              Snap::xi[2*ang]), _mm_set1_pd(Snap::hk)), hv_z[ang])),
                        _mm_mul_pd(_mm_set1_pd(vdelt), hv_t[ang])));
                  __m128d pc_gt = _mm_cmpgt_pd(pc[ang], _mm_set1_pd(0.0));
                  // Set the denominator back to zero if it is too small
                  den = _mm_and_pd(den, pc_gt);
                  __m128d den_ge = _mm_cmpge_pd(den, tolr);
                  pc[ang] = _mm_and_pd(den_ge, _mm_div_pd(pc[ang], den));
                }
              } else {
                for (int ang = 0; ang < num_vec_angles; ang++) {
                  __m128d sum = _mm_mul_pd(psii[ang], _mm_mul_pd(
                        _mm_set_pd(Snap::mu[2*ang+1], Snap::mu[2*ang]), 
                        _mm_mul_pd( _mm_set1_pd(Snap::hi), 
                          _mm_add_pd( _mm_set1_pd(1.0), hv_x[ang]))));
                  sum = _mm_add_pd(sum, _mm_mul_pd(psij[ang], _mm_mul_pd(
                          _mm_set_pd(Snap::eta[2*ang+1], Snap::eta[2*ang]),
                          _mm_mul_pd( _mm_set1_pd(Snap::hj),
                            _mm_add_pd( _mm_set1_pd(1.0), hv_y[ang])))));
                  sum = _mm_add_pd(sum, _mm_mul_pd(psik[ang], _mm_mul_pd(
                          _mm_set_pd(Snap::xi[2*ang+1], Snap::xi[2*ang]),
                          _mm_mul_pd( _mm_set1_pd(Snap::hk),
                            _mm_add_pd( _mm_set1_pd(1.0), hv_z[ang])))));
                  pc[ang] = _mm_add_pd(psi[ang], _mm_mul_pd( _mm_set1_pd(0.5), sum));
                  __m128d den = _mm_add_pd(_mm_set1_pd(t_xs), _mm_add_pd( _mm_add_pd( 
                        _mm_mul_pd( _mm_mul_pd( _mm_set_pd(Snap::mu[2*ang+1], 
                              Snap::mu[2*ang]), _mm_set1_pd(Snap::hi)), hv_x[ang]),
                        _mm_mul_pd( _mm_mul_pd( _mm_set_pd(Snap::eta[2*ang+1],
                              Snap::eta[2*ang]), _mm_set1_pd(Snap::hj)), hv_y[ang])),
                        _mm_mul_pd( _mm_mul_pd( _mm_set_pd(Snap::xi[2*ang+1],
                              Snap::xi[2*ang]), _mm_set1_pd(Snap::hk)), hv_z[ang])));
                  __m128d pc_gt = _mm_cmpgt_pd(pc[ang], _mm_set1_pd(0.0));
                  // Set the denominator back to zero if it is too small
                  den = _mm_and_pd(den, pc_gt);
                  __m128d den_ge = _mm_cmpge_pd(den, tolr);
                  pc[ang] = _mm_and_pd(den_ge, _mm_div_pd(pc[ang], den));
                }
              }
            }
            // Fixup done so compute the update values
            for (int ang = 0; ang < num_vec_angles; ang++)
              psii[ang] = _mm_mul_pd(fx_hv_x[ang], hv_x[ang]);
            for (int ang = 0; ang < num_vec_angles; ang++)
              psij[ang] = _mm_mul_pd(fx_hv_y[ang], hv_y[ang]);
            for (int ang = 0; ang < num_vec_angles; ang++)
              psik[ang] = _mm_mul_pd(fx_hv_z[ang], hv_z[ang]);
            if (vdelt != 0.0)
            {
              // Write out the outgoing temporal flux 
              __m128d *__restrict__ time_flux_out = 
                get_sse_angle_ptr<3>(time_flux_out_ptr, 
                                     time_flux_out_offsets, local_point);
              for (int ang = 0; ang < num_vec_angles; ang++)
                _mm_stream_pd((double*)(time_flux_out+ang), 
                    _mm_mul_pd(fx_hv_t[ang], hv_t[ang]));
            }
          } else {
            // NO FIXUP
            for (int ang = 0; ang < num_vec_angles; ang++)
              psii[ang] = _mm_sub_pd( _mm_mul_pd( _mm_set1_pd(2.0), pc[ang]), psii[ang]);
            for (int ang = 0; ang < num_vec_angles; ang++)
              psij[ang] = _mm_sub_pd( _mm_mul_pd( _mm_set1_pd(2.0), pc[ang]), psij[ang]);
            for (int ang = 0; ang < num_vec_angles; ang++)
              psik[ang] = _mm_sub_pd( _mm_mul_pd( _mm_set1_pd(2.0), pc[ang]), psik[ang]);
            if (vdelt != 0.0) 
            {
              // Write out the outgoing temporal flux 
              __m128d *__restrict__ time_flux_out = 
                get_sse_angle_ptr<3>(time_flux_out_ptr, 
                                     time_flux_out_offsets, local_point);
              for (int ang = 0; ang < num_vec_angles; ang++)
                _mm_stream_pd((double*)(time_flux_out+ang), 
                    _mm_sub_pd( _mm_mul_pd( _mm_set1_pd(2.0), pc[ang]), time_flux_in[ang]));
            }
          }
          // Write out the ghost regions
          if (x == (Snap::nx_per_chunk-1)) {
            // We write out on our own region
            Point<2> ghost_point = ghostx_point(local_point);
            __m128d *__restrict__ target = get_sse_angle_ptr<2>(ghostx_ptr, 
                                              ghostx_offsets, ghost_point);
            for (int ang = 0; ang < num_vec_angles; ang++)
              _mm_stream_pd((double*)(target+ang), psii[ang]);
          } 
          // Else nothing: psii just gets caried over to next iteration
          // Y ghost
          if (y == (Snap::ny_per_chunk-1)) {
            // Write out on our own region
            Point<2> ghost_point = ghosty_point(local_point);
            __m128d *__restrict__ target = get_sse_angle_ptr<2>(ghosty_ptr, 
                                                ghosty_offsets, ghost_point);
            for (int ang = 0; ang < num_vec_angles; ang++)
              _mm_stream_pd((double*)(target+ang), psij[ang]);
          } 
          // Else nothing: psij is already in place in the pencil
          // Z ghost
          if (z == (Snap::nz_per_chunk-1)) {
            Point<2> ghost_point = ghostz_point(local_point);
            __m128d *__restrict__ target = get_sse_angle_ptr<2>(ghostz_ptr, 
                                              ghostz_offsets, ghost_point);
            for (int ang = 0; ang < num_vec_angles; ang++)
              _mm_stream_pd((double*)(target+ang), psik[ang]);
          } 
          // Else nothing: psik is already in place in the plane

          // Finally we apply reductions to the flux moments
          __m128d vec_total = _mm_set1_pd(0.0);
          for (int ang = 0; ang < num_vec_angles; ang++) {
            psi[ang] = _mm_mul_pd(pc[ang], _mm_set_pd(Snap::w[2*ang+1], Snap::w[2*ang]));
            vec_total = _mm_add_pd(vec_total, psi[ang]);
          }
          double total = _mm_cvtsd_f64(_mm_hadd_pd(vec_total, vec_total));
#ifndef SNAP_USE_RELAXED_COHERENCE
          SumReduction::fold<true>(*(flux + flux_offsets * local_point), total);
#else
          SumReduction::apply<false>(*(flux + flux_offsets * local_point), total);
#endif
          if (Snap::num_moments > 1) {
            MomentTriple triple;
            for (int l = 1; l < Snap::num_moments; l++) {
              unsigned offset = l * Snap::num_angles + 
                args->corner * Snap::num_angles * Snap::num_moments;
              vec_total = _mm_set1_pd(0.0);
              for (int ang = 0; ang < num_vec_angles; ang++)
                vec_total = _mm_add_pd(vec_total, _mm_mul_pd(psi[ang],
                      _mm_set_pd(Snap::ec[offset+2*ang+1], Snap::ec[offset+2*ang])));
              triple[l-1] = _mm_cvtsd_f64(_mm_hadd_pd(vec_total, vec_total));
            }
#ifndef SNAP_USE_RELAXED_COHERENCE
            TripleReduction::fold<true>(*(fluxm + fluxm_offsets * local_point), triple);
#else
            TripleReduction::apply<false>(*(fluxm + fluxm_offsets * local_point), triple);
#endif
          }
        }
      }
    }
  }

  free(psi);
  free(pc);
  free(psii);
  free(hv_x);
  free(hv_y);
  free(hv_z);
  free(hv_t);
  free(fx_hv_x);
  free(fx_hv_y);
  free(fx_hv_z);
  free(fx_hv_t);
  free(yflux_pencil);
  free(zflux_plane);
#endif
}

#ifdef __AVX__
template<int DIM>
inline __m256d* get_avx_angle_ptr(void *ptr, const ByteOffset offsets[DIM],
                                  const Point<DIM> &point)
{
  return (__m256d*)(ptr + (offsets * point));
}

static inline void ignore_return(int arg) { }

inline __m256d* malloc_avx_aligned(size_t size)
{
  __m256d *result;
  ignore_return(posix_memalign((void**)&result, 32, size));
  return result;
}

//------------------------------------------------------------------------------
/*static*/ void MiniKBATask::avx_implementation(
  const Task *task, Context ctx, Runtime *runtime,
  const std::vector<MomentQuad*> &qtot_ptrs, const ByteOffset qtot_offsets[3],
  const std::vector<double*> &flux_ptrs, const ByteOffset flux_offsets[3],
  const std::vector<double*> &qim_ptrs, const ByteOffset qim_offsets[3],
  const std::vector<MomentTriple*> &fluxm_ptrs, const ByteOffset fluxm_offsets[3],
  const std::vector<double*> &dinv_ptrs, const ByteOffset dinv_offsets[3],
  const std::vector<double*> &time_flux_in_ptrs, const ByteOffset time_flux_in_offsets[3],
  const std::vector<double*> &time_flux_out_ptrs, const ByteOffset time_flux_out_offsets[3],
  const std::vector<double*> &t_xs_ptrs, const ByteOffset t_xs_offsets[3],
  const std::vector<double*> &ghost_z_ptrs, const ByteOffset ghostz_offsets[2],
  const std::vector<double*> &ghost_x_ptrs, const ByteOffset ghostx_offsets[2],
  const std::vector<double*> &ghost_y_ptrs, const ByteOffset ghosty_offsets[2],
  const std::vector<double*> &vdelt_ptrs, const ByteOffset vdelt_offsets[1])
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running AVX Mini-KBA Sweep");

  assert(task->arglen == sizeof(MiniKBAArgs));
  const MiniKBAArgs *args = reinterpret_cast<const MiniKBAArgs*>(task->args);
    
  // This implementation of the sweep assumes three dimensions
  assert(Snap::num_dims == 3);

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

  // Figure out the origin point based on which corner we are
  const bool stride_x_positive = ((args->corner & 0x1) != 0);
  const bool stride_y_positive = ((args->corner & 0x2) != 0);
  const bool stride_z_positive = ((args->corner & 0x4) != 0);
  // Convert to local coordinates
  const coord_t origin_ints[3] = { 
    (stride_x_positive ? subgrid_bounds.lo[0] : subgrid_bounds.hi[0]) - subgrid_bounds.lo[0],
    (stride_y_positive ? subgrid_bounds.lo[1] : subgrid_bounds.hi[1]) - subgrid_bounds.lo[1],
    (stride_z_positive ? subgrid_bounds.lo[2] : subgrid_bounds.hi[2]) - subgrid_bounds.lo[2]};
  const Point<3> origin(origin_ints);

  // Local arrays
  assert((Snap::num_angles % 4) == 0);
  const int num_vec_angles = Snap::num_angles/4;
  const size_t angle_buffer_size = num_vec_angles * sizeof(__m256d);
  __m256d *__restrict__ psi = malloc_avx_aligned(angle_buffer_size);
  __m256d *__restrict__ pc = malloc_avx_aligned(angle_buffer_size);
  __m256d *__restrict__ psii = malloc_avx_aligned(angle_buffer_size);
  __m256d *__restrict__ hv_x = malloc_avx_aligned(angle_buffer_size);
  __m256d *__restrict__ hv_y = malloc_avx_aligned(angle_buffer_size);
  __m256d *__restrict__ hv_z = malloc_avx_aligned(angle_buffer_size);
  __m256d *__restrict__ hv_t = malloc_avx_aligned(angle_buffer_size);
  __m256d *__restrict__ fx_hv_x = malloc_avx_aligned(angle_buffer_size);
  __m256d *__restrict__ fx_hv_y = malloc_avx_aligned(angle_buffer_size);
  __m256d *__restrict__ fx_hv_z = malloc_avx_aligned(angle_buffer_size);
  __m256d *__restrict__ fx_hv_t = malloc_avx_aligned(angle_buffer_size);

  const __m256d tolr = _mm256_set1_pd(1.0e-12);

  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1; 
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;
  // See note in the CPU implementation about why we do things this way
  __m256d *yflux_pencil = malloc_avx_aligned(x_range * angle_buffer_size);
  __m256d *zflux_plane  = malloc_avx_aligned(y_range * x_range * angle_buffer_size); 

  for (unsigned group = 0; group < flux_ptrs.size(); group++) {
    const MomentQuad *const qtot_ptr = qtot_ptrs[group];
    double *const flux = flux_ptrs[group];

    void *const qim_ptr = qim_ptrs[group];
    MomentTriple *const fluxm = fluxm_ptrs[group];
 
    // No types here since the size of these fields are dependent
    // on the number of angles
    void *const dinv_ptr = dinv_ptrs[group];
    void *const time_flux_in_ptr = time_flux_in_ptrs[group]; 
    void *const time_flux_out_ptr = time_flux_out_ptrs[group]; 
    const double *const t_xs_ptr = t_xs_ptrs[group];

    // Ghost regions
    void *const ghostx_ptr = ghost_x_ptrs[group];
    void *const ghosty_ptr = ghost_y_ptrs[group];
    void *const ghostz_ptr = ghost_z_ptrs[group];

    const double vdelt = *(vdelt_ptrs[group]);

    for (int z = 0; z < z_range; z++) {
      for (int y = 0; y < y_range; y++) {
        for (int x = 0; x < x_range; x++) {
          // Figure out the local point that we are working on    
          Point<3> local_point = origin;
          if (stride_x_positive)
            local_point.x[0] += x;
          else
            local_point.x[0] -= x;
          if (stride_y_positive)
            local_point.x[1] += y;
          else
            local_point.x[1] -= y;
          if (stride_z_positive)
            local_point.x[2] += z;
          else
            local_point.x[2] -= z;

          // Compute the angular source
          MomentQuad quad = *(qtot_ptr + qtot_offsets * local_point);
          for (int ang = 0; ang < num_vec_angles; ang++)
            psi[ang] = _mm256_set1_pd(quad[0]);
          if (Snap::num_moments > 1) {
            const int corner_offset = 
              args->corner * Snap::num_angles * Snap::num_moments;
            for (int l = 1; l < Snap::num_moments; l++) {
              const int moment_offset = corner_offset + l * Snap::num_angles;
              for (int ang = 0; ang < num_vec_angles; ang++) {
                psi[ang] = _mm256_add_pd(psi[ang], _mm256_mul_pd(
                      _mm256_set_pd(Snap::ec[moment_offset+4*ang+3],
                                    Snap::ec[moment_offset+4*ang+2],
                                    Snap::ec[moment_offset+4*ang+1],
                                    Snap::ec[moment_offset+4*ang]),
                      _mm256_set1_pd(quad[l])));
              }
            }
          }

          // If we're doing MMS, there is an additional term
          if (Snap::source_layout == Snap::MMS_SOURCE)
          {
            __m256d *__restrict__ qim = (__m256d*)(qim_ptr + qim_offsets * local_point);
            for (int ang = 0; ang < num_vec_angles; ang++)
              psi[ang] = _mm256_add_pd(psi[ang], qim[ang]);
          }

          // Compute the initial solution
          for (int ang = 0; ang < num_vec_angles; ang++)
            pc[ang] = psi[ang];
          // X ghost cells
          if (x == 0) {
            // Ghost cell array
            Point<2> ghost_point = ghostx_point(local_point);
            memcpy(psii, get_avx_angle_ptr<2>(ghostx_ptr, ghostx_offsets,
                                              ghost_point), angle_buffer_size);
          } // Else nothing: psii already contains next flux
          for (int ang = 0; ang < num_vec_angles; ang++)
            pc[ang] = _mm256_add_pd(pc[ang], _mm256_mul_pd( _mm256_mul_pd(psii[ang], 
                    _mm256_set_pd(Snap::mu[4*ang+3], Snap::mu[4*ang+2],
                                  Snap::mu[4*ang+1], Snap::mu[4*ang])), 
                    _mm256_set1_pd(Snap::hi)));
          // Y ghost cells
          __m256d *__restrict__ psij = yflux_pencil + x * num_vec_angles;
          if (y == 0) {
            // Ghost cell array
            Point<2> ghost_point = ghosty_point(local_point);
            memcpy(psij, get_avx_angle_ptr<2>(ghosty_ptr, ghosty_offsets, 
                                              ghost_point), angle_buffer_size);
          } // Else nothing: psij already points at the flux
          for (int ang = 0; ang < num_vec_angles; ang++)
            pc[ang] = _mm256_add_pd(pc[ang], _mm256_mul_pd( _mm256_mul_pd(psij[ang],
                    _mm256_set_pd(Snap::eta[4*ang+3], Snap::eta[4*ang+2],
                                  Snap::eta[4*ang+1], Snap::eta[4*ang])),
                    _mm256_set1_pd(Snap::hj)));
          // Z ghost cells
          __m256d *__restrict__ psik = zflux_plane + (y * x_range + x) * num_vec_angles;
          if (z == 0) {
            // Ghost cell array
            Point<2> ghost_point = ghostz_point(local_point);
            memcpy(psik, get_avx_angle_ptr<2>(ghostz_ptr, ghostz_offsets, 
                                              ghost_point), angle_buffer_size);
          } // Else nothing: psik already points at the flux
          for (int ang = 0; ang < num_vec_angles; ang++)
            pc[ang] = _mm256_add_pd(pc[ang], _mm256_mul_pd( _mm256_mul_pd(psik[ang],
                    _mm256_set_pd(Snap::xi[4*ang+3], Snap::xi[4*ang+2],
                                  Snap::xi[4*ang+1], Snap::xi[4*ang])),
                    _mm256_set1_pd(Snap::hk)));

          // See if we're doing anything time dependent
          __m256d *__restrict__ time_flux_in = get_avx_angle_ptr<3>(time_flux_in_ptr,
                                           time_flux_in_offsets, local_point);
          if (vdelt != 0.0) 
          {
            for (int ang = 0; ang < num_vec_angles; ang++)
              pc[ang] = _mm256_add_pd(pc[ang], _mm256_mul_pd(
                    _mm256_set1_pd(vdelt), time_flux_in[ang]));
          }
          // Multiple by the precomputed denominator inverse
          __m256d *__restrict__ dinv = 
            get_avx_angle_ptr<3>(dinv_ptr, dinv_offsets, local_point);
          for (int ang = 0; ang < num_vec_angles; ang++)
            pc[ang] = _mm256_mul_pd(pc[ang], dinv[ang]);

          if (Snap::flux_fixup) {
            // DO THE FIXUP
            unsigned old_negative_fluxes = 0;
            for (int ang = 0; ang < num_vec_angles; ang++)
              hv_x[ang] = _mm256_set1_pd(1.0);
            for (int ang = 0; ang < num_vec_angles; ang++)
              hv_y[ang] = _mm256_set1_pd(1.0);
            for (int ang = 0; ang < num_vec_angles; ang++)
              hv_z[ang] = _mm256_set1_pd(1.0);
            for (int ang = 0; ang < num_vec_angles; ang++)
              hv_t[ang] = _mm256_set1_pd(1.0);
            const double t_xs = *(t_xs_ptr + t_xs_offsets * local_point);
            while (true) {
              unsigned negative_fluxes = 0;
              // Figure out how many negative fluxes we have
              for (int ang = 0; ang < num_vec_angles; ang++) {
                fx_hv_x[ang] = _mm256_sub_pd( _mm256_mul_pd( 
                      _mm256_set1_pd(2.0), pc[ang]), psii[ang]);
                __m256d ge = _mm256_cmp_pd(fx_hv_x[ang], 
                    _mm256_set1_pd(0.0), _CMP_GE_OS);
                // If not greater than or equal, set back to zero
                hv_x[ang] = _mm256_and_pd(ge, hv_x[ang]);
                // Count how many negative fluxes we had
#ifdef __AVX2__
                __m256i negatives = _mm256_andnot_si256(_mm256_castpd_si256(ge),
                    _mm256_set_epi32(0, 1, 0, 1, 0, 1, 0, 1));
#else
                __m256i negatives = _mm256_castpd_si256(
                    _mm256_andnot_pd(ge, _mm256_castsi256_pd(
                        _mm256_set_epi32(0, 1, 0, 1, 0, 1, 0, 1))));
#endif
                negative_fluxes += _mm256_extract_epi32(negatives, 0);
                negative_fluxes += _mm256_extract_epi32(negatives, 2);
                negative_fluxes += _mm256_extract_epi32(negatives, 4);
                negative_fluxes += _mm256_extract_epi32(negatives, 6);
              }
              for (int ang = 0; ang < num_vec_angles; ang++) {
                fx_hv_y[ang] = _mm256_sub_pd( _mm256_mul_pd( 
                      _mm256_set1_pd(2.0), pc[ang]), psij[ang]);
                __m256d ge = _mm256_cmp_pd(fx_hv_y[ang], 
                    _mm256_set1_pd(0.0), _CMP_GE_OS);
                // If not greater than or equal set back to zero
                hv_y[ang] = _mm256_and_pd(ge, hv_y[ang]);
                // Count how many negative fluxes we had
#ifdef __AVX2__
                __m256i negatives = _mm256_andnot_si256(_mm256_castpd_si256(ge),
                    _mm256_set_epi32(0, 1, 0, 1, 0, 1, 0, 1));
#else
                __m256i negatives = _mm256_castpd_si256(
                    _mm256_andnot_pd(ge, _mm256_castsi256_pd(
                        _mm256_set_epi32(0, 1, 0, 1, 0, 1, 0, 1))));
#endif
                negative_fluxes += _mm256_extract_epi32(negatives, 0);
                negative_fluxes += _mm256_extract_epi32(negatives, 2);
                negative_fluxes += _mm256_extract_epi32(negatives, 4);
                negative_fluxes += _mm256_extract_epi32(negatives, 6);
              }
              for (int ang = 0; ang < num_vec_angles; ang++) {
                fx_hv_z[ang] = _mm256_sub_pd( _mm256_mul_pd( 
                      _mm256_set1_pd(2.0), pc[ang]), psik[ang]);
                __m256d ge = _mm256_cmp_pd(fx_hv_z[ang], 
                    _mm256_set1_pd(0.0), _CMP_GE_OS);
                // If not greater than or equal set back to zero
                hv_z[ang] = _mm256_and_pd(ge, hv_z[ang]);
                // Count how many negative fluxes we had
#ifdef __AVX2__
                __m256i negatives = _mm256_andnot_si256(_mm256_castpd_si256(ge),
                    _mm256_set_epi32(0, 1, 0, 1, 0, 1, 0, 1));
#else
                __m256i negatives = _mm256_castpd_si256(
                    _mm256_andnot_pd(ge, _mm256_castsi256_pd(
                        _mm256_set_epi32(0, 1, 0, 1, 0, 1, 0, 1))));
#endif
                negative_fluxes += _mm256_extract_epi32(negatives, 0);
                negative_fluxes += _mm256_extract_epi32(negatives, 2);
                negative_fluxes += _mm256_extract_epi32(negatives, 4);
                negative_fluxes += _mm256_extract_epi32(negatives, 6);
              }
              if (vdelt != 0.0) {
                for (int ang = 0; ang < num_vec_angles; ang++) {
                  fx_hv_t[ang] = _mm256_sub_pd( _mm256_mul_pd( 
                        _mm256_set1_pd(2.0), pc[ang]), time_flux_in[ang]);
                  __m256d ge = _mm256_cmp_pd(fx_hv_t[ang], 
                      _mm256_set1_pd(0.0), _CMP_GE_OS);
                  // If not greater than or equal, set back to zero
                  hv_t[ang] = _mm256_and_pd(ge, hv_t[ang]);
                  // Count how many negative fluxes we had
#ifdef __AVX2__
                __m256i negatives = _mm256_andnot_si256(_mm256_castpd_si256(ge),
                    _mm256_set_epi32(0, 1, 0, 1, 0, 1, 0, 1));
#else
                __m256i negatives = _mm256_castpd_si256(
                    _mm256_andnot_pd(ge, _mm256_castsi256_pd(
                        _mm256_set_epi32(0, 1, 0, 1, 0, 1, 0, 1))));
#endif
                  negative_fluxes += _mm256_extract_epi32(negatives, 0);
                  negative_fluxes += _mm256_extract_epi32(negatives, 2);
                  negative_fluxes += _mm256_extract_epi32(negatives, 4);
                  negative_fluxes += _mm256_extract_epi32(negatives, 6);
                }
              }
              if (negative_fluxes == old_negative_fluxes)
                break;
              old_negative_fluxes = negative_fluxes;
              if (vdelt != 0.0) {
                for (int ang = 0; ang < num_vec_angles; ang++) {
                  __m256d sum = _mm256_mul_pd(psii[ang], _mm256_mul_pd(
                        _mm256_set_pd(Snap::mu[4*ang+3], Snap::mu[4*ang+2],
                                      Snap::mu[4*ang+1], Snap::mu[4*ang]), 
                        _mm256_mul_pd( _mm256_set1_pd(Snap::hi), 
                          _mm256_add_pd( _mm256_set1_pd(1.0), hv_x[ang]))));
                  sum = _mm256_add_pd(sum, _mm256_mul_pd(psij[ang], _mm256_mul_pd(
                          _mm256_set_pd(Snap::eta[4*ang+3], Snap::eta[4*ang+2],
                                        Snap::eta[4*ang+1], Snap::eta[4*ang]),
                          _mm256_mul_pd( _mm256_set1_pd(Snap::hj),
                            _mm256_add_pd( _mm256_set1_pd(1.0), hv_y[ang])))));
                  sum = _mm256_add_pd(sum, _mm256_mul_pd(psik[ang], _mm256_mul_pd(
                          _mm256_set_pd(Snap::xi[4*ang+3], Snap::xi[4*ang+2],
                                        Snap::xi[4*ang+1], Snap::xi[4*ang]),
                          _mm256_mul_pd( _mm256_set1_pd(Snap::hk),
                            _mm256_add_pd( _mm256_set1_pd(1.0), hv_z[ang])))));
                  sum = _mm256_add_pd(sum, _mm256_mul_pd(time_flux_in[ang], 
                        _mm256_mul_pd( _mm256_set1_pd(vdelt), _mm256_add_pd(
                            _mm256_set1_pd(1.0), hv_t[ang]))));
                  pc[ang] = _mm256_add_pd(psi[ang], 
                      _mm256_mul_pd( _mm256_set1_pd(0.5), sum));
                  __m256d den = _mm256_add_pd(_mm256_set1_pd(t_xs), 
                      _mm256_add_pd( _mm256_add_pd( _mm256_add_pd(
                        _mm256_mul_pd( _mm256_mul_pd( _mm256_set_pd(
                              Snap::mu[4*ang+3], Snap::mu[4*ang+2],
                              Snap::mu[4*ang+1], Snap::mu[4*ang]), 
                            _mm256_set1_pd(Snap::hi)), hv_x[ang]),
                        _mm256_mul_pd( _mm256_mul_pd( _mm256_set_pd(
                              Snap::eta[4*ang+3], Snap::eta[4*ang+2],
                              Snap::eta[4*ang+1], Snap::eta[4*ang]), 
                            _mm256_set1_pd(Snap::hj)), hv_y[ang])),
                        _mm256_mul_pd( _mm256_mul_pd( _mm256_set_pd(
                              Snap::xi[4*ang+3], Snap::xi[4*ang+2],
                              Snap::xi[4*ang+1], Snap::xi[4*ang]), 
                            _mm256_set1_pd(Snap::hk)), hv_z[ang])),
                        _mm256_mul_pd(_mm256_set1_pd(vdelt), hv_t[ang])));
                  __m256d pc_gt = _mm256_cmp_pd(pc[ang], 
                      _mm256_set1_pd(0.0), _CMP_GT_OS);
                  // Set the denominator back to zero if it is too small
                  den = _mm256_and_pd(den, pc_gt);
                  __m256d den_ge = _mm256_cmp_pd(den, tolr, _CMP_GE_OS);
                  pc[ang] = _mm256_and_pd(den_ge, _mm256_div_pd(pc[ang], den));
                }
              } else {
                for (int ang = 0; ang < num_vec_angles; ang++) {
                  __m256d sum = _mm256_mul_pd(psii[ang], _mm256_mul_pd(
                        _mm256_set_pd(Snap::mu[4*ang+3], Snap::mu[4*ang+2],
                                      Snap::mu[4*ang+1], Snap::mu[4*ang]), 
                        _mm256_mul_pd( _mm256_set1_pd(Snap::hi), 
                          _mm256_add_pd( _mm256_set1_pd(1.0), hv_x[ang]))));
                  sum = _mm256_add_pd(sum, _mm256_mul_pd(psij[ang], _mm256_mul_pd(
                          _mm256_set_pd(Snap::eta[4*ang+3], Snap::eta[4*ang+2],
                                     Snap::eta[4*ang+1], Snap::eta[4*ang]),
                          _mm256_mul_pd( _mm256_set1_pd(Snap::hj),
                            _mm256_add_pd( _mm256_set1_pd(1.0), hv_y[ang])))));
                  sum = _mm256_add_pd(sum, _mm256_mul_pd(psik[ang], _mm256_mul_pd(
                          _mm256_set_pd(Snap::xi[4*ang+3], Snap::xi[4*ang+2],
                                        Snap::xi[4*ang+1], Snap::xi[4*ang]),
                          _mm256_mul_pd( _mm256_set1_pd(Snap::hk),
                            _mm256_add_pd( _mm256_set1_pd(1.0), hv_z[ang])))));
                  pc[ang] = _mm256_add_pd(psi[ang], 
                            _mm256_mul_pd( _mm256_set1_pd(0.5), sum));
                  __m256d den = _mm256_add_pd(_mm256_set1_pd(t_xs), 
                      _mm256_add_pd( _mm256_add_pd( 
                        _mm256_mul_pd( _mm256_mul_pd( _mm256_set_pd(
                              Snap::mu[4*ang+3], Snap::mu[4*ang+2], 
                              Snap::mu[4*ang+1], Snap::mu[4*ang]), 
                            _mm256_set1_pd(Snap::hi)), hv_x[ang]),
                        _mm256_mul_pd( _mm256_mul_pd( _mm256_set_pd(
                              Snap::eta[4*ang+3], Snap::eta[4*ang+2],
                              Snap::eta[4*ang+1], Snap::eta[4*ang]), 
                            _mm256_set1_pd(Snap::hj)), hv_y[ang])),
                        _mm256_mul_pd( _mm256_mul_pd( _mm256_set_pd(
                              Snap::xi[4*ang+3], Snap::xi[4*ang+2],
                              Snap::xi[4*ang+1], Snap::xi[4*ang]), 
                            _mm256_set1_pd(Snap::hk)), hv_z[ang])));
                  __m256d pc_gt = _mm256_cmp_pd(pc[ang], _mm256_set1_pd(0.0), _CMP_GT_OS);
                  // Set the denominator back to zero if it is too small
                  den = _mm256_and_pd(den, pc_gt);
                  __m256d den_ge = _mm256_cmp_pd(den, tolr, _CMP_GE_OS);
                  pc[ang] = _mm256_and_pd(den_ge, _mm256_div_pd(pc[ang], den));
                }
              }
            }
            // Fixup done so compute the update values
            for (int ang = 0; ang < num_vec_angles; ang++)
              psii[ang] = _mm256_mul_pd(fx_hv_x[ang], hv_x[ang]);
            for (int ang = 0; ang < num_vec_angles; ang++)
              psij[ang] = _mm256_mul_pd(fx_hv_y[ang], hv_y[ang]);
            for (int ang = 0; ang < num_vec_angles; ang++)
              psik[ang] = _mm256_mul_pd(fx_hv_z[ang], hv_z[ang]);
            if (vdelt != 0.0)
            {
              // Write out the outgoing temporal flux 
              __m256d *__restrict__ time_flux_out = 
                get_avx_angle_ptr<3>(time_flux_out_ptr, 
                                     time_flux_out_offsets, local_point);
              for (int ang = 0; ang < num_vec_angles; ang++)
                _mm256_stream_pd((double*)(time_flux_out+ang), 
                    _mm256_mul_pd(fx_hv_t[ang], hv_t[ang]));
            }
          } else {
            // NO FIXUP
            for (int ang = 0; ang < num_vec_angles; ang++)
              psii[ang] = _mm256_sub_pd( _mm256_mul_pd( 
                    _mm256_set1_pd(2.0), pc[ang]), psii[ang]);
            for (int ang = 0; ang < num_vec_angles; ang++)
              psij[ang] = _mm256_sub_pd( _mm256_mul_pd( 
                    _mm256_set1_pd(2.0), pc[ang]), psij[ang]);
            for (int ang = 0; ang < num_vec_angles; ang++)
              psik[ang] = _mm256_sub_pd( _mm256_mul_pd( 
                    _mm256_set1_pd(2.0), pc[ang]), psik[ang]);
            if (vdelt != 0.0) 
            {
              // Write out the outgoing temporal flux 
              __m256d *__restrict__ time_flux_out = 
                get_avx_angle_ptr<3>(time_flux_out_ptr, 
                                     time_flux_out_offsets, local_point);
              for (int ang = 0; ang < num_vec_angles; ang++)
                _mm256_stream_pd((double*)(time_flux_out+ang), 
                    _mm256_sub_pd( _mm256_mul_pd( _mm256_set1_pd(2.0), 
                        pc[ang]), time_flux_in[ang]));
            }
          }

          // Write out the ghost regions
          if (x == (Snap::nx_per_chunk-1)) {
            // We write out on our own region
            Point<2> ghost_point = ghostx_point(local_point);
            __m256d *__restrict__ target = get_avx_angle_ptr<2>(ghostx_ptr, 
                                              ghostx_offsets, ghost_point);
            for (int ang = 0; ang < num_vec_angles; ang++)
              _mm256_stream_pd((double*)(target+ang), psii[ang]);
          } 
          // Else nothing: psii just gets caried over to next iteration
          // Y ghost
          if (y == (Snap::ny_per_chunk-1)) {
            // Write out on our own region
            Point<2> ghost_point = ghosty_point(local_point);
            __m256d *__restrict__ target = get_avx_angle_ptr<2>(ghosty_ptr, 
                                              ghosty_offsets, ghost_point);
            for (int ang = 0; ang < num_vec_angles; ang++)
              _mm256_stream_pd((double*)(target+ang), psij[ang]);
          } 
          // Else nothing: psij is already in place in the pencil
          // Z ghost
          if (z == (Snap::nz_per_chunk-1)) {
            Point<2> ghost_point = ghostz_point(local_point);
            __m256d *__restrict__ target = get_avx_angle_ptr<2>(ghostz_ptr, 
                                              ghostz_offsets, ghost_point);
            for (int ang = 0; ang < num_vec_angles; ang++)
              _mm256_stream_pd((double*)(target+ang), psik[ang]);
          } 
          // Else nothing: psik is already in place in the plane

          // Finally we apply reductions to the flux moments
          __m256d vec_total = _mm256_set1_pd(0.0);
          for (int ang = 0; ang < num_vec_angles; ang++) {
            psi[ang] = _mm256_mul_pd(pc[ang], _mm256_set_pd(
                  Snap::w[4*ang+3], Snap::w[4*ang+2], Snap::w[4*ang+1], Snap::w[4*ang]));
            vec_total = _mm256_add_pd(vec_total, psi[ang]);
          }
          vec_total = _mm256_hadd_pd(vec_total, vec_total);
          double total = _mm_cvtsd_f64(_mm256_extractf128_pd(vec_total, 0)) + 
                         _mm_cvtsd_f64(_mm256_extractf128_pd(vec_total, 1));
#ifndef SNAP_USE_RELAXED_COHERENCE
          SumReduction::fold<true>(*(flux + flux_offsets * local_point), total);
#else
          SumReduction::apply<false>(*(flux + flux_offsets * local_point), total);
#endif
          if (Snap::num_moments > 1) {
            MomentTriple triple;
            for (int l = 1; l < Snap::num_moments; l++) {
              unsigned offset = l * Snap::num_angles + 
                args->corner * Snap::num_angles * Snap::num_moments;
              vec_total = _mm256_set1_pd(0.0);
              for (int ang = 0; ang < num_vec_angles; ang++)
                vec_total = _mm256_add_pd(vec_total, _mm256_mul_pd(psi[ang],
                      _mm256_set_pd(Snap::ec[offset+4*ang+3], Snap::ec[offset+4*ang+2],
                                    Snap::ec[offset+4*ang+1], Snap::ec[offset+4*ang])));
              vec_total = _mm256_hadd_pd(vec_total, vec_total);
              triple[l-1] = _mm_cvtsd_f64( _mm256_extractf128_pd(
                              _mm256_hadd_pd(vec_total, vec_total), 0));
            }
#ifndef SNAP_USE_RELAXED_COHERENCE
            TripleReduction::fold<true>(*(fluxm + fluxm_offsets * local_point), triple);
#else
            TripleReduction::apply<false>(*(fluxm + fluxm_offsets * local_point), triple);
#endif
          }
        }
      }
    }
  }

  free(psi);
  free(pc);
  free(psii);
  free(hv_x);
  free(hv_y);
  free(hv_z);
  free(hv_t);
  free(fx_hv_x);
  free(fx_hv_y);
  free(fx_hv_z);
  free(fx_hv_t);
  free(yflux_pencil);
  free(zflux_plane);
#endif
}
#endif // __AVX__

#ifdef USE_GPU_KERNELS
extern void run_gpu_sweep(const Point<3> origin, 
               const MomentQuad *qtot_ptr,
                     double     *flux_ptr,
                     MomentTriple *fluxm_ptr,
               const double     *dinv_ptr,
               const double     *time_flux_in_ptr,
                     double     *time_flux_out_ptr,
               const double     *t_xs_ptr,
                     double     *ghostx_ptr,
                     double     *ghosty_ptr,
                     double     *ghostz_ptr,
               const double     *qim_ptr,
               const ByteOffset qtot_offsets[3],
               const ByteOffset flux_offsets[3],
               const ByteOffset fluxm_offsets[3],
               const ByteOffset dinv_offsets[3],
               const ByteOffset time_flux_in_offsets[3],
               const ByteOffset time_flux_out_offsets[3],
               const ByteOffset t_xs_offsets[3],
               const ByteOffset ghostx_out_offsets[2],
               const ByteOffset ghosty_out_offsets[2],
               const ByteOffset ghostz_out_offsets[2],
               const ByteOffset qim_offsets[3],
               const int x_range, const int y_range, 
               const int z_range, const int corner,
               const bool stride_x_positive,
               const bool stride_y_positive,
               const bool stride_z_positive,
               const bool mms_source, 
               const int num_moments, 
               const double hi, const double hj,
               const double hk, const double vdelt,
               const int num_angles, const bool fixup);
#endif

//------------------------------------------------------------------------------
/*static*/ void MiniKBATask::gpu_implementation(
 const Task *task, Context ctx, Runtime *runtime,
 const std::vector<MomentQuad*> &qtot_ptrs, const ByteOffset qtot_offsets[3],
 const std::vector<double*> &flux_ptrs, const ByteOffset flux_offsets[3],
 const std::vector<double*> &qim_ptrs, const ByteOffset qim_offsets[3],
 const std::vector<MomentTriple*> &fluxm_ptrs, const ByteOffset fluxm_offsets[3],
 const std::vector<double*> &dinv_ptrs, const ByteOffset dinv_offsets[3],
 const std::vector<double*> &time_flux_in_ptrs, const ByteOffset time_flux_in_offsets[3],
 const std::vector<double*> &time_flux_out_ptrs, const ByteOffset time_flux_out_offsets[3],
 const std::vector<double*> &t_xs_ptrs, const ByteOffset t_xs_offsets[3],
 const std::vector<double*> &ghost_z_ptrs, const ByteOffset ghostz_offsets[2],
 const std::vector<double*> &ghost_x_ptrs, const ByteOffset ghostx_offsets[2],
 const std::vector<double*> &ghost_y_ptrs, const ByteOffset ghosty_offsets[2],
 const std::vector<double*> &vdelt_ptrs, const ByteOffset vdelt_offsets[1])
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
#ifdef USE_GPU_KERNELS
  log_snap.info("Running GPU Mini-KBA Sweep");

  assert(task->arglen == sizeof(MiniKBAArgs));
  const MiniKBAArgs *args = reinterpret_cast<const MiniKBAArgs*>(task->args);
    
  // This implementation of the sweep assumes three dimensions
  assert(Snap::num_dims == 3);

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

    // Figure out the origin point based on which corner we are
  const bool stride_x_positive = ((args->corner & 0x1) != 0);
  const bool stride_y_positive = ((args->corner & 0x2) != 0);
  const bool stride_z_positive = ((args->corner & 0x4) != 0);
  // Convert to local coordinates
  const coord_t origin_ints[3] = { 
    (stride_x_positive ? subgrid_bounds.lo[0] : subgrid_bounds.hi[0]) - subgrid_bounds.lo[0],
    (stride_y_positive ? subgrid_bounds.lo[1] : subgrid_bounds.hi[1]) - subgrid_bounds.lo[1],
    (stride_z_positive ? subgrid_bounds.lo[2] : subgrid_bounds.hi[2]) - subgrid_bounds.lo[2]};
  const Point<3> origin(origin_ints);

  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1; 
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;

  const bool mms_source = (Snap::source_layout == Snap::MMS_SOURCE);

  for (int group = 0; group < flux_ptrs.size(); group++) {
    const MomentQuad *const qtot_ptr = qtot_ptrs[group];
    double *const flux_ptr = flux_ptrs[group];
    MomentTriple *const fluxm_ptr = fluxm_ptrs[group];
    double *const qim_ptr = qim_ptrs[group];
    
    // No types here since the size of these fields are dependent
    // on the number of angles
    double *const dinv_ptr = dinv_ptrs[group]; 
    double *const time_flux_in_ptr = time_flux_in_ptrs[group]; 
    double *const time_flux_out_ptr = time_flux_out_ptrs[group]; 
    const double *const t_xs_ptr = t_xs_ptrs[group];

    // Ghost regions
    double *const ghostx_ptr = ghost_x_ptrs[group]; 
    double *const ghosty_ptr = ghost_y_ptrs[group]; 
    double *const ghostz_ptr = ghost_z_ptrs[group];

    const double vdelt = *(vdelt_ptrs[group]);

    run_gpu_sweep(origin, qtot_ptr, flux_ptr, fluxm_ptr, dinv_ptr, 
        time_flux_in_ptr, time_flux_out_ptr, t_xs_ptr, ghostx_ptr,
        ghosty_ptr, ghostz_ptr, qim_ptr, qtot_offsets, flux_offsets,
        fluxm_offsets, dinv_offsets, time_flux_in_offsets, 
        time_flux_out_offsets, t_xs_offsets, ghostx_offsets,
        ghosty_offsets, ghostz_offsets, qim_offsets,
        x_range, y_range, z_range, args->corner, stride_x_positive,
        stride_y_positive, stride_z_positive, mms_source, 
        Snap::num_moments, Snap::hi, Snap::hj, Snap::hk, vdelt,
        Snap::num_angles, Snap::flux_fixup);
  }
#else
  assert(false);
#endif
#endif
}


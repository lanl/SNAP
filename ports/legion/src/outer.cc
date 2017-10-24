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

#include <cmath>

#include "snap.h"
#include "outer.h"
#include "legion_stl.h"

extern LegionRuntime::Logger::Category log_snap;

using namespace LegionRuntime::Accessor;
using namespace Legion::STL;

//------------------------------------------------------------------------------
CalcOuterSource::CalcOuterSource(const Snap &snap, const Predicate &pred,
                                 const SnapArray &qi, const SnapArray &slgg,
                                 const SnapArray &mat, const SnapArray &q2rgp0, 
                                 const SnapArray &q2grpm, 
                                 const SnapArray &flux0, const SnapArray &fluxm)
  : SnapTask<CalcOuterSource, Snap::CALC_OUTER_SOURCE_TASK_ID>(
      snap, snap.get_launch_bounds(), pred)
//------------------------------------------------------------------------------
{
  qi.add_projection_requirement(READ_ONLY, *this); // qi0
  flux0.add_projection_requirement(READ_ONLY, *this); // flux0
  slgg.add_region_requirement(READ_ONLY, *this); // sxs_g
  mat.add_projection_requirement(READ_ONLY, *this); // map
  q2rgp0.add_projection_requirement(WRITE_DISCARD, *this); // qo0
  // Only have to initialize this if there are multiple moments
  if (Snap::num_moments > 1) {
    fluxm.add_projection_requirement(READ_ONLY, *this); // fluxm 
    q2grpm.add_projection_requirement(WRITE_DISCARD, *this); // qom
  } else {
    fluxm.add_projection_requirement(NO_ACCESS, *this); // fluxm 
    q2grpm.add_projection_requirement(NO_ACCESS, *this); // qom
  }
}

//------------------------------------------------------------------------------
/*static*/ void CalcOuterSource::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need x86 CPU
  execution_constraints.add_constraint(ISAConstraint(X86_ISA));
  // Need L1 cache at least 32 KB
  execution_constraints.add_constraint(
      ResourceConstraint(L1_CACHE_SIZE, GE_EK/*>=*/, 32768/*32 KB*/));
  // Need L1 cache with at least 8 way set associativity
  execution_constraints.add_constraint(
      ResourceConstraint(L1_CACHE_ASSOCIATIVITY, GE_EK/*>=*/, 8));
  TaskLayoutConstraintSet layout_constraints;
  // All regions need to be SOA
  for (unsigned idx = 0; idx < 7; idx++)
    layout_constraints.add_layout_constraint(idx/*index*/,
                                             Snap::get_soa_layout()); 
#if defined(BOUNDS_CHECKS) || defined(PRIVILEGE_CHECKS)
  register_cpu_variant<cpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
#else
  register_cpu_variant<
    raw_rect_task_wrapper<double,3,double,3,MomentQuad,2,int,3,double,3,
                       MomentTriple,3,MomentTriple,3,fast_implementation> >(
          execution_constraints, layout_constraints, true/*leaf*/);
#endif
}

//------------------------------------------------------------------------------
/*static*/ void CalcOuterSource::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need a CUDA GPU with at least sm30
  execution_constraints.add_constraint(ISAConstraint(CUDA_ISA | SM_30_ISA));
  // Need at least 48 KB of shared memory
  execution_constraints.add_constraint(
      ResourceConstraint(SHARED_MEMORY_SIZE, GE_EK/*>=*/, 49152/*48KB*/));
  // Need at least 64K registers
  execution_constraints.add_constraint(
      ResourceConstraint(REGISTER_FILE_SIZE, GE_EK/*>=*/, 65536/*registers*/));
  // Need at least two CTAs per SM for performance
  execution_constraints.add_constraint(LaunchConstraint(CTAS_PER_SM, 2));
  TaskLayoutConstraintSet layout_constraints;
  // All regions need to be SOA
  for (unsigned idx = 0; idx < 7; idx++)
    layout_constraints.add_layout_constraint(idx/*index*/, 
                                             Snap::get_soa_layout());
  register_gpu_variant<
    raw_rect_task_wrapper<double,3,double,3,MomentQuad,2,int,3,double,3,
                       MomentTriple,3,MomentTriple,3,gpu_implementation> >(
            execution_constraints, layout_constraints, true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void CalcOuterSource::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running Calc Outer Source");

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();
  const bool multi_moment = (Snap::num_moments > 1);
  const int num_groups = task->regions[0].privilege_fields.size();
  assert(num_groups == int(task->regions[1].privilege_fields.size()));
  assert(num_groups == int(task->regions[4].privilege_fields.size()));
  // Make the accessors for all the groups up front
  std::vector<RegionAccessor<AccessorType::Generic,double> > fa_qi0(num_groups);
  std::vector<RegionAccessor<AccessorType::Generic,double> > fa_flux0(num_groups);
  std::vector<RegionAccessor<AccessorType::Generic,double> > fa_qo0(num_groups);
  std::vector<RegionAccessor<AccessorType::Generic,MomentTriple> > 
    fa_fluxm(multi_moment ? num_groups : 0);
  // Field spaces are all the same so this is safe
  int g = 0;
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++, g++)
  {
    fa_qi0[g] = regions[0].get_field_accessor(*it).typeify<double>();
    fa_flux0[g] = regions[1].get_field_accessor(*it).typeify<double>();
    fa_qo0[g] = regions[4].get_field_accessor(*it).typeify<double>();
    if (multi_moment)
      fa_fluxm[g] = regions[5].get_field_accessor(*it).typeify<MomentTriple>();
  }
  RegionAccessor<AccessorType::Generic,int> fa_mat = 
    regions[3].get_field_accessor(Snap::FID_SINGLE).typeify<int>();

  // Do pairwise group intersections
  g = 0;
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++, g++)
  {
    RegionAccessor<AccessorType::Generic,MomentQuad> fa_slgg = 
      regions[2].get_field_accessor(*it).typeify<MomentQuad>();
    for (GenericPointInRectIterator<3> itr(subgrid_bounds); itr; itr++)
    {
      DomainPoint dp = DomainPoint::from_point<3>(itr.p);
      double qo0 = fa_qi0[g].read(dp);
      const int mat = fa_mat.read(dp);
      for (int g2 = 0; g2 < num_groups; g2++)
      {
        if (g == g2)
          continue;
        const int pvals[2] = { mat, g2 }; 
        DomainPoint xsp = DomainPoint::from_point<2>(Point<2>(pvals));
        const MomentQuad cs = fa_slgg.read(xsp);
        qo0 += cs[0] * fa_flux0[g2].read(dp);
      }
      fa_qo0[g].write(dp, qo0);
    }
    if (multi_moment)
    {
      RegionAccessor<AccessorType::Generic,MomentTriple> fa_qom = 
        regions[6].get_field_accessor(*it).typeify<MomentTriple>();
      for (GenericPointInRectIterator<3> itr(subgrid_bounds); itr; itr++)
      {
        DomainPoint dp = DomainPoint::from_point<3>(itr.p);
        const int mat = fa_mat.read(dp);
        MomentTriple qom;
        for (int g2 = 0; g2 < num_groups; g2++)
        {
          if (g == g2)
            continue;
          int moment = 0;
          MomentTriple csm;
          for (int l = 1; l < Snap::num_moments; l++)
          {
            const int pvals[2] = { mat, g2 };
            DomainPoint xsp = DomainPoint::from_point<2>(Point<2>(pvals));
            const MomentQuad scat = fa_slgg.read(xsp);
            for (int i = 0; i < Snap::lma[l]; i++)
              csm[moment+i] = scat[l];
            moment += Snap::lma[l];
          }
          MomentTriple fluxm = fa_fluxm[g2].read(dp); 
          for (int i = 0; i < (Snap::num_moments-1); i++)
            qom[i] += csm[i] * fluxm[i];
        }
        fa_qom.write(dp, qom);  
      }
    }
  }
#endif
}

static int gcd(int a, int b)
{
  if (a < b) return gcd(a, b-a);
  if (a > b) return gcd(a-b, b);
  return a;
}

//------------------------------------------------------------------------------
/*static*/ void CalcOuterSource::fast_implementation(
    const Task *task, Context ctx, Runtime *runtime,
    const std::vector<double*> &qi0_ptrs, const ByteOffset qi0_offsets[3],
    const std::vector<double*> &flux0_ptrs, const ByteOffset flux0_offsets[3],
    const std::vector<MomentQuad*> &slgg_ptrs, const ByteOffset slgg_offsets[2],
    const std::vector<int*> &mat_ptrs, const ByteOffset mat_offsets[3],
    const std::vector<double*> &qo0_ptrs, const ByteOffset qo0_offsets[3],
    const std::vector<MomentTriple*> &fluxm_ptrs, const ByteOffset fluxm_offsets[3],
    const std::vector<MomentTriple*> &qom_ptrs, const ByteOffset qom_offsets[3])
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running Fast Calc Outer Source");

  // Note: there's no real need for vectors here, it's all about blocking
  // for caches into order to get good memory locality

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();
  const bool multi_moment = (Snap::num_moments > 1);
  const int num_groups = task->regions[0].privilege_fields.size();

  const int *const mat_ptr = mat_ptrs[0]; 

  // We'll block the innermost dimension to get some cache locality
  // This assumes a worst case 128 energy groups and a 32 KB L1 cache
  const int max_x = subgrid_bounds.hi[0] - subgrid_bounds.lo[0] + 1;
  const int max_y = subgrid_bounds.hi[1] - subgrid_bounds.lo[1] + 1;
  const int max_z = subgrid_bounds.hi[2] - subgrid_bounds.lo[2] + 1;
  const int strip_size = gcd(max_x, 32);
  double *flux_strip = (double*)malloc(num_groups*strip_size*sizeof(double));

  for (int z = 0; z < max_z; z++) {
    for (int y = 0; y < max_y; y++) {
      for (int x = 0; x < max_x; x+=strip_size) {
        // Read in the flux strip first
        const ByteOffset flux_offset = 
          x * flux0_offsets[0] + y * flux0_offsets[1] + z * flux0_offsets[2];
        for (int g = 0; g < num_groups; g++) {
          double *flux_ptr = flux0_ptrs[g] + flux_offset;
          for (int i = 0; i < strip_size; i++) 
            flux_strip[g * strip_size + i] = *(flux_ptr + i * flux0_offsets[0]);
        }
        const ByteOffset qi0_offset = 
          x * qi0_offsets[0] + y * qi0_offsets[1] + z * qi0_offsets[2];
        const ByteOffset mat_offset = 
          x * mat_offsets[0] + y * mat_offsets[1] + z * mat_offsets[2];
        const ByteOffset qo0_offset = 
          x * qo0_offsets[0] + y * qo0_offsets[1] + z * qo0_offsets[2];
        // We've loaded all the strips, now do the math
        for (int g1 = 0; g1 < num_groups; g1++) {
          for (int i = 0; i < strip_size; i++) {
            double qo0 = *(qi0_ptrs[g1] + qi0_offset + i * qi0_offsets[0]);
            // Have to look up the the two materials separately 
            const int mat = *(mat_ptr + mat_offset + i * mat_offsets[0]);
            for (int g2 = 0; g2 < num_groups; g2++) {
              if (g1 == g2)
                continue;
#ifdef LEGION_ISSUE_214_FIX
              MomentQuad cs = *(slgg_ptrs[g1] + 
                mat * slgg_offsets[0] + g2 * slgg_offsets[1]);
#else
              MomentQuad cs = *(slgg_ptrs[g1] + 
                (mat-1) * slgg_offsets[0] + g2 * slgg_offsets[1]);
#endif
              qo0 += cs[0] * flux_strip[g2 * strip_size + i];
            }
            *(qo0_ptrs[g1] + qo0_offset + i * qo0_offsets[0]) = qo0;
          }
        }
      }
    }
  }
  free(flux_strip);
  // Handle multi-moment
  if (multi_moment) {
    MomentTriple *fluxm_strip = 
      (MomentTriple*)malloc(num_groups*strip_size*sizeof(MomentTriple));
    for (int z = 0; z < max_z; z++) {
      for (int y = 0; y < max_y; y++) {
        for (int x = 0; x < max_x; x+=strip_size) {
          const ByteOffset fluxm_offset = 
            x * fluxm_offsets[0] + y * fluxm_offsets[1] + z * fluxm_offsets[2];
          // Read in the fluxm strip first
          for (int g = 0; g < num_groups; g++) {
            MomentTriple *fluxm_ptr = fluxm_ptrs[g] + fluxm_offset;
            for (int i = 0; i < strip_size; i++)
              fluxm_strip[g * strip_size+ i] = *(fluxm_ptr + i * fluxm_offsets[0]);
          }
          const ByteOffset mat_offset = 
            x * mat_offsets[0] + y * mat_offsets[1] + z * mat_offsets[2];
          const ByteOffset qom_offset = 
            x * qom_offsets[0] + y * qom_offsets[1] + z * qom_offsets[2];
          // We've loaded all the strips, now do the math
          for (int g1 = 0; g1 < num_groups; g1++) {
            for (int i = 0; i < strip_size; i++) {
              const int mat = *(mat_ptr + mat_offset + i * mat_offsets[0]);
              MomentTriple qom;
              for (int g2 = 0; g2 < num_groups; g2++) {
                if (g1 == g2)
                  continue;
                int moment = 0;
                MomentTriple csm;
#ifdef LEGION_ISSUE_214_FIX
                MomentQuad scat = *(slgg_ptrs[g1] + 
                      mat * slgg_offsets[0] + g2 * slgg_offsets[1]);
#else
                MomentQuad scat = *(slgg_ptrs[g1] + 
                      (mat-1) * slgg_offsets[0] + g2 * slgg_offsets[1]);
#endif
                for (int l = 1; l < Snap::num_moments; l++) {
                  for (int j = 0; j < Snap::lma[l]; j++)
                    csm[moment+j] = scat[l];
                  moment += Snap::lma[l];
                }
                MomentTriple fluxm = fluxm_strip[g2 * strip_size + i]; 
                for (int l = 0; l < (Snap::num_moments-1); l++)
                  qom[l] += csm[l] * fluxm[l];
              }
              *(qom_ptrs[g1] + qom_offset + i * qom_offsets[0]) = qom;
            }
          }
        }
      }
    }
    free(fluxm_strip);
  }
#endif
}

#ifdef USE_GPU_KERNELS
  extern void run_flux0_outer_source(Rect<3> subgrid_bounds,
                            const std::vector<double*> &qi0_ptrs,
                            const std::vector<double*> &flux0_ptrs,
                            const std::vector<MomentQuad*> &slgg_ptrs,
                            const std::vector<double*> &qo0_ptrs, 
                            const int *mat_ptr, 
                            const ByteOffset qi0_offsets[3], 
                            const ByteOffset flux0_offsets[3],
                            const ByteOffset slgg_offsets[2], 
                            const ByteOffset qo0_offsets[3],
                            const ByteOffset mat_offsets[3], const int num_groups);
  extern void run_fluxm_outer_source(Rect<3> subgrid_bounds,
                            const std::vector<MomentTriple*> &fluxm_ptrs,
                            const std::vector<MomentQuad*> &slgg_ptrs,
                            const std::vector<MomentTriple*> &qom_ptrs, 
                            const int *mat_ptr, 
                            const ByteOffset fluxm_offsets[3], 
                            const ByteOffset slgg_offsets[2],
                            const ByteOffset mat_offsets[3], 
                            const ByteOffset qom_offsets[3],
                            const int num_groups, const int num_moments, const int lma[4]);
#endif

//------------------------------------------------------------------------------
/*static*/ void CalcOuterSource::gpu_implementation(
    const Task *task, Context ctx, Runtime *runtime,
    const std::vector<double*> &qi0_ptrs, const ByteOffset qi0_offsets[3],
    const std::vector<double*> &flux0_ptrs, const ByteOffset flux0_offsets[3],
    const std::vector<MomentQuad*> &slgg_ptrs, const ByteOffset slgg_offsets[2],
    const std::vector<int*> &mat_ptrs, const ByteOffset mat_offsets[3],
    const std::vector<double*> &qo0_ptrs, const ByteOffset qo0_offsets[3],
    const std::vector<MomentTriple*> &fluxm_ptrs, const ByteOffset fluxm_offsets[3],
    const std::vector<MomentTriple*> &qom_ptrs, const ByteOffset qom_offsets[3])
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running GPU Calc Outer Source");
#ifdef USE_GPU_KERNELS
  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();
  const bool multi_moment = (Snap::num_moments > 1);
  const int num_groups = task->regions[0].privilege_fields.size();

  const int *const mat_ptr = mat_ptrs[0];

  run_flux0_outer_source(subgrid_bounds, qi0_ptrs, flux0_ptrs, slgg_ptrs, 
                         qo0_ptrs, mat_ptr, qi0_offsets, flux0_offsets,
                         slgg_offsets, qo0_offsets, mat_offsets, num_groups);

  if (multi_moment) {
    run_fluxm_outer_source(subgrid_bounds, fluxm_ptrs, slgg_ptrs, qom_ptrs,
                           mat_ptr, fluxm_offsets, slgg_offsets, mat_offsets,
                           qom_offsets, num_groups, Snap::num_moments, Snap::lma);
  }
#else
  assert(false);
#endif
#endif
}

//------------------------------------------------------------------------------
TestOuterConvergence::TestOuterConvergence(const Snap &snap, 
                                           const Predicate &pred,
                                           const SnapArray &flux0,
                                           const SnapArray &flux0po,
                                           const Future &inner_converged,
                                           const Future &true_future,
                                           int group_start, int group_stop)
  : SnapTask<TestOuterConvergence, Snap::TEST_OUTER_CONVERGENCE_TASK_ID>(
      snap, snap.get_launch_bounds(), pred)
//------------------------------------------------------------------------------
{
  if (group_start == group_stop) {
    // Special case for a single field
    const Snap::SnapFieldID group_field = SNAP_ENERGY_GROUP_FIELD(group_start);
    flux0.add_projection_requirement(READ_ONLY, *this, group_field);
    flux0po.add_projection_requirement(READ_ONLY, *this, group_field);
  } else {
    // General case for arbitrary set of fields
    std::vector<Snap::SnapFieldID> group_fields((group_stop - group_start) + 1);
    for (int group = group_start; group <= group_stop; group++)
      group_fields[group-group_start] = SNAP_ENERGY_GROUP_FIELD(group);
    flux0.add_projection_requirement(READ_ONLY, *this, group_fields);
    flux0po.add_projection_requirement(READ_ONLY, *this, group_fields);
  }
  add_future(inner_converged);
  predicate_false_future = true_future;
}

//------------------------------------------------------------------------------
/*static*/ void TestOuterConvergence::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need x86 CPU
  execution_constraints.add_constraint(ISAConstraint(X86_ISA));
  TaskLayoutConstraintSet layout_constraints;
  // All regions need to be SOA
  for (unsigned idx = 0; idx < 2; idx++)
    layout_constraints.add_layout_constraint(idx/*index*/,
                                             Snap::get_soa_layout());
#if defined(BOUNDS_CHECKS) || defined(PRIVILEGE_CHECKS)
  register_cpu_variant<bool, cpu_implementation>(execution_constraints,
                                                 layout_constraints,
                                                 true/*leaf*/);
#else
  register_cpu_variant<bool,
    raw_rect_task_wrapper<bool, double, 3, double, 3, fast_implementation> >(
        execution_constraints, layout_constraints, true/*leaf*/);
#endif
}

//------------------------------------------------------------------------------
/*static*/ void TestOuterConvergence::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need a CUDA GPU with at least sm30
  execution_constraints.add_constraint(ISAConstraint(CUDA_ISA | SM_30_ISA));
  // Need at least 128B of shared memory
  execution_constraints.add_constraint(
      ResourceConstraint(SHARED_MEMORY_SIZE, GE_EK/*>=*/, 128/*B*/));
  TaskLayoutConstraintSet layout_constraints;
  // All regions need to be SOA
  for (unsigned idx = 0; idx < 2; idx++)
    layout_constraints.add_layout_constraint(idx/*index*/, 
                                             Snap::get_soa_layout());
  register_gpu_variant<bool,
    raw_rect_task_wrapper<bool, double, 3, double, 3, gpu_implementation> >(
        execution_constraints, layout_constraints, true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ bool TestOuterConvergence::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running Test Outer Convergence");

  // If the inner loop didn't converge, then we can't either
  assert(!task->futures.empty());
  bool inner_converged = task->futures[0].get_result<bool>();
  if (!inner_converged)
    return false;
  // Get the index space domain for iteration
  assert(task->regions[0].region.get_index_space() == 
         task->regions[1].region.get_index_space());
  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();
  const double tolr = 1.0e-12;
  const double epsi = 100.0 * Snap::convergence_eps;
  // Iterate over all the energy groups
  assert(task->regions[0].privilege_fields.size() == 
         task->regions[1].privilege_fields.size());
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++)
  {
    RegionAccessor<AccessorType::Generic,double> fa_flux0 = 
      regions[0].get_field_accessor(*it).typeify<double>();
    RegionAccessor<AccessorType::Generic,double> fa_flux0po = 
      regions[1].get_field_accessor(*it).typeify<double>();
    for (GenericPointInRectIterator<3> itr(subgrid_bounds); itr; itr++)
    {
      DomainPoint dp = DomainPoint::from_point<3>(itr.p);
      double flux0po = fa_flux0po.read(dp);
      double df = 1.0;
      if (fabs(flux0po) < tolr) {
        flux0po = 1.0;
        df = 0.0;
      }
      double flux0 = fa_flux0.read(dp);
      df = fabs( (flux0 / flux0po) - df );
      // Skip anything less than -INF
      if (df < -INFINITY)
        continue;
      if (df > epsi)
        return false;
    }
  }
  return true;
#else
  return false;
#endif
}

//------------------------------------------------------------------------------
/*static*/ bool TestOuterConvergence::fast_implementation(
  const Task *task, Context ctx, Runtime *runtime,
  const std::vector<double*> &flux0_ptrs, const ByteOffset flux0_offsets[3],
  const std::vector<double*> &flux0po_ptrs, const ByteOffset flux0po_offsets[3])
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running Fast Test Outer Convergence");

  // If the inner loop didn't converge, then we can't either
  assert(!task->futures.empty());
  bool inner_converged = task->futures[0].get_result<bool>();
  if (!inner_converged)
    return false;

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();
  const double tolr = 1.0e-12;
  const double epsi = 100.0 * Snap::convergence_eps;
  const int max_x = subgrid_bounds.hi[0] - subgrid_bounds.lo[0] + 1;
  const int max_y = subgrid_bounds.hi[1] - subgrid_bounds.lo[1] + 1;
  const int max_z = subgrid_bounds.hi[2] - subgrid_bounds.lo[2] + 1;

  for (unsigned group = 0; group < flux0_ptrs.size(); group++)
  {
    const double *const flux0_ptr = flux0_ptrs[group];
    const double *const flux0po_ptr = flux0po_ptrs[group];
    for (int z = 0; z < max_z; z++) {
      for (int y = 0; y < max_y; y++) {
        for (int x = 0; x < max_x; x++) {
          double flux0po = *(flux0po_ptr + x * flux0po_offsets[0] +
              y * flux0po_offsets[1] + z * flux0po_offsets[2]);
          double df = 1.0;
          if (fabs(flux0po) < tolr) {
            flux0po = 1.0;
            df = 0.0;
          }
          double flux0 = *(flux0_ptr + x * flux0_offsets[0] + 
              y * flux0_offsets[1] + z * flux0_offsets[2]);
          df = fabs( (flux0 / flux0po) - df );
          // Skip anything less than -INF
          if (df < -INFINITY)
            continue;
          if (df > epsi)
            return false;
        }
      }
    }
  }
  return true;
#else
  return false;
#endif
}

#ifdef USE_GPU_KERNELS
extern bool run_outer_convergence(Rect<3> subgrid_bounds,
                                  const std::vector<double*> flux0_ptrs,
                                  const std::vector<double*> flux0po_ptrs,
                                  const ByteOffset flux0_offsets[3], 
                                  const ByteOffset flux0po_offsets[3],
                                  const double epsi);
#endif

//------------------------------------------------------------------------------
/*static*/ bool TestOuterConvergence::gpu_implementation(
  const Task *task, Context ctx, Runtime *runtime,
  const std::vector<double*> &flux0_ptrs, const ByteOffset flux0_offsets[3],
  const std::vector<double*> &flux0po_ptrs, const ByteOffset flux0po_offsets[3])
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running GPU Test Outer Convergence");
#ifdef USE_GPU_KERNELS
  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();
  const double epsi = 100.0 * Snap::convergence_eps;
  return run_outer_convergence(subgrid_bounds, flux0_ptrs, flux0po_ptrs,
                               flux0_offsets, flux0po_offsets, epsi);
#else
  assert(false);
#endif
#endif
  return false;
}


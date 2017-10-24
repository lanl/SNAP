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

extern LegionRuntime::Logger::Category log_snap;

using namespace LegionRuntime::Accessor;

//------------------------------------------------------------------------------
InitMaterial::InitMaterial(const Snap &snap, const SnapArray &mat)
  : SnapTask<InitMaterial,Snap::INIT_MATERIAL_TASK_ID>(
      snap, snap.get_launch_bounds(), Predicate::TRUE_PRED)
//------------------------------------------------------------------------------
{
  mat.add_projection_requirement(READ_WRITE, *this);
}

//------------------------------------------------------------------------------
/*static*/ void InitMaterial::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need x86 CPU
  execution_constraints.add_constraint(ISAConstraint(X86_ISA));
  TaskLayoutConstraintSet layout_constraints;
  layout_constraints.add_layout_constraint(0/*idx*/,
                                           Snap::get_soa_layout());
  register_cpu_variant<cpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void InitMaterial::cpu_implementation(const Task *task,
    const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running Init Material");

  int i1 = 1, i2 = 1, j1 = 1, j2 = 1, k1 = 1, k2 = 1;
  switch (Snap::material_layout)
  {
    case Snap::CENTER_LAYOUT:
      {
        const int nx_gl = Snap::nx;
        i1 = nx_gl / 4 + 1;
        i2 = 3 * nx_gl / 4;
        if (Snap::num_dims > 1) {
          const int ny_gl = Snap::ny;
          j1 = ny_gl/ 4 + 1;
          j2 = 3 * ny_gl / 4;
          if (Snap::num_dims > 2) {
            const int nz_gl = Snap::nz;
            k1 = nz_gl / 4 + 1;
            k2 = 3 * nz_gl / 4;
          }
        }
        break;
      }
    case Snap::CORNER_LAYOUT:
      {
        const int nx_gl = Snap::nx;
        i2 = nx_gl / 2;
        if (Snap::num_dims > 1) {
          const int ny_gl = Snap::ny;
          j2 = ny_gl / 2;
          if (Snap::num_dims > 2) {
            const int nz_gl = Snap::nz;
            k2 = nz_gl / 2;
          }
        }
        break;
      }
    default:
      assert(false);
  }
  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();
  RegionAccessor<AccessorType::Generic,int> fa_mat = 
    regions[0].get_field_accessor(Snap::FID_SINGLE).typeify<int>();
  Rect<3> mat_bounds;
  mat_bounds.lo.x[0] = i1-1;
  mat_bounds.lo.x[1] = j1-1;
  mat_bounds.lo.x[2] = k1-1;
  mat_bounds.hi.x[0] = i2-1;
  mat_bounds.hi.x[1] = j2-1;
  mat_bounds.hi.x[2] = k2-1;
  Rect<3> local_bounds = subgrid_bounds.intersection(mat_bounds);
  if (local_bounds.volume() == 0)
    return;
  for (GenericPointInRectIterator<3> itr(local_bounds); itr; itr++) {
    DomainPoint dp = DomainPoint::from_point<3>(itr.p);
    fa_mat.write(dp, 2);
  }
#endif
}

//------------------------------------------------------------------------------
InitSource::InitSource(const Snap &snap, const SnapArray &qi)
  : SnapTask<InitSource, Snap::INIT_SOURCE_TASK_ID>(
      snap, snap.get_launch_bounds(), Predicate::TRUE_PRED)
//------------------------------------------------------------------------------
{
  qi.add_projection_requirement(READ_WRITE, *this);
}

//------------------------------------------------------------------------------
/*static*/ void InitSource::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need x86 CPU
  execution_constraints.add_constraint(ISAConstraint(X86_ISA));
  TaskLayoutConstraintSet layout_constraints;
  layout_constraints.add_layout_constraint(0/*index*/,
                                           Snap::get_soa_layout());
  register_cpu_variant<cpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void InitSource::cpu_implementation(const Task *task,
    const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running Init Source");

  const int nx_gl = Snap::nx;
  const int ny_gl = Snap::ny;
  const int nz_gl = Snap::nz;

  int i1 = 1, i2 = nx_gl, j1 = 1, j2 = ny_gl, k1 = 1, k2 = nz_gl;

  switch (Snap::source_layout)
  {
    case Snap::EVERYWHERE_SOURCE:
      break;
    case Snap::CENTER_SOURCE:
      {
        i1 = nx_gl / 4 + 1;
        i2 = 3 * nx_gl / 4;
        if (Snap::num_dims > 1) {
          j1 = ny_gl / 4 + 1;
          j2 = 3 * ny_gl / 4;
          if (Snap::num_dims > 2) { 
            k1 = nz_gl / 4 + 1;
            k2 = 3 * nz_gl / 4;
          }
        }
        break;
      }
    case Snap::CORNER_SOURCE:
      {
        i2 = nx_gl / 2;
        if (Snap::num_dims > 1) {
          j2 = ny_gl / 2;
          if (Snap::num_dims > 2)
            k2 = nz_gl / 2;
        }
        break;
      }
    default: // nothing else should be called
      assert(false);
  }
  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();
  Rect<3> source_bounds;
  source_bounds.lo.x[0] = i1-1;
  source_bounds.lo.x[1] = j1-1;
  source_bounds.lo.x[2] = k1-1;
  source_bounds.hi.x[0] = i2-1;
  source_bounds.hi.x[1] = j2-1;
  source_bounds.hi.x[2] = k2-1;
  Rect<3> local_bounds = subgrid_bounds.intersection(source_bounds);
  if (local_bounds.volume() == 0)
    return;
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++)
  {
    RegionAccessor<AccessorType::Generic,double> fa_qi = 
      regions[0].get_field_accessor(*it).typeify<double>();
    for (GenericPointInRectIterator<3> itr(local_bounds); itr; itr++) {
      DomainPoint dp = DomainPoint::from_point<3>(itr.p);
      fa_qi.write(dp, 1.0);
    }
  }
#endif
}

//------------------------------------------------------------------------------
InitGPUSweep::InitGPUSweep(const Snap &snap, const Rect<3> &launch)
  : SnapTask<InitGPUSweep, Snap::INIT_GPU_SWEEP_TASK_ID>(
      snap, launch, Predicate::TRUE_PRED)
//------------------------------------------------------------------------------
{
}

//------------------------------------------------------------------------------
/*static*/ void InitGPUSweep::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need a CUDA GPU with at least sm30
  execution_constraints.add_constraint(ISAConstraint(CUDA_ISA));
  TaskLayoutConstraintSet layout_constraints;
  register_gpu_variant<gpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
}

#ifdef USE_GPU_KERNELS
extern void initialize_gpu_context(const double *ec_h, const double *mu_h,
                                   const double *eta_h, const double *xi_h,
                                   const double *w_h, const int num_angles,
                                   const int num_moments, const int num_octants,
                                   const int nx_per_chunk, const int ny_per_chunk);
#endif

//------------------------------------------------------------------------------
/*static*/ void InitGPUSweep::gpu_implementation(const Task *task,
    const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime) 
//------------------------------------------------------------------------------
{
  log_snap.info("Running Init GPU Sweep");
#ifdef USE_GPU_KERNELS
  initialize_gpu_context(Snap::ec, Snap::mu, Snap::eta, Snap::xi, Snap::w,
                         Snap::num_angles, Snap::num_moments, Snap::num_octants,
                         Snap::nx_per_chunk, Snap::ny_per_chunk);
#else
  assert(false); 
#endif
}


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

#ifndef __SNAP_H__
#define __SNAP_H__

#include "legion.h"
#include "default_mapper.h"
#include "snap_types.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#ifndef SNAP_MAX_ENERGY_GROUPS
#define SNAP_MAX_ENERGY_GROUPS            1024
#endif

#ifndef PI
#define PI (3.14159265358979)
#endif

using namespace Legion;
using namespace Legion::Mapping;
using namespace LegionRuntime::Arrays;

extern LegionRuntime::Logger::Category log_snap;

class SnapArray;

class Snap {
public:
  enum SnapTaskID {
    SNAP_TOP_LEVEL_TASK_ID,
    INIT_MATERIAL_TASK_ID,
    INIT_SOURCE_TASK_ID,
    INIT_GPU_SWEEP_TASK_ID,
    CALC_OUTER_SOURCE_TASK_ID,
    TEST_OUTER_CONVERGENCE_TASK_ID,
    CALC_INNER_SOURCE_TASK_ID,
    TEST_INNER_CONVERGENCE_TASK_ID,
    MINI_KBA_TASK_ID,
    EXPAND_CROSS_SECTION_TASK_ID,
    EXPAND_SCATTERING_CROSS_SECTION_TASK_ID,
    CALCULATE_GEOMETRY_PARAM_TASK_ID,
    MMS_INIT_FLUX_TASK_ID,
    MMS_INIT_SOURCE_TASK_ID,
    MMS_INIT_TIME_DEPENDENT_TASK_ID,
    MMS_SCALE_TASK_ID,
    MMS_COMPARE_TASK_ID,
    BIND_INNER_CONVERGENCE_TASK_ID,
    BIND_OUTER_CONVERGENCE_TASK_ID,
    SUMMARY_TASK_ID,
    LAST_TASK_ID, // must be last
  };
#define SNAP_TASK_NAMES                 \
    "Top_Level_Task",                   \
    "Initialize_Material",              \
    "Initialize_Source",                \
    "Initialize_GPU Sweep",             \
    "Calc_Outer_Source",                \
    "Test_Outer_Convergence",           \
    "Calc_Inner_Source",                \
    "Test_Inner_Convergence",           \
    "Mini_KBA",                         \
    "Expand_Cross_Section",             \
    "Expand_Scattering_Cross_Section",  \
    "Calcuate_Geometry Param",          \
    "MMS_Init_Flux",                    \
    "MMS_Init_Source",                  \
    "MMS_Init_Time Dependent",          \
    "MMS_Scale",                        \
    "MMS_Compare",                      \
    "Bind_Inner_Convergence",           \
    "Bind_Outer_Convergence",           \
    "Summary"
  static const char* task_names[LAST_TASK_ID];
  enum MaterialLayout {
    HOMOGENEOUS_LAYOUT = 0,
    CENTER_LAYOUT = 1,
    CORNER_LAYOUT = 2,
  };
  enum SourceLayout {
    EVERYWHERE_SOURCE = 0,
    CENTER_SOURCE = 1,
    CORNER_SOURCE = 2,
    MMS_SOURCE = 3,
  };
  enum SnapTunable {
    OUTER_RUNAHEAD_TUNABLE = DefaultMapper::DEFAULT_TUNABLE_LAST,
    INNER_RUNAHEAD_TUNABLE = DefaultMapper::DEFAULT_TUNABLE_LAST+1,
    SWEEP_ENERGY_CHUNKS_TUNABLE = DefaultMapper::DEFAULT_TUNABLE_LAST+2,
  };
  enum SnapReductionID {
    NO_REDUCTION_ID = 0,
    AND_REDUCTION_ID = 1,
    SUM_REDUCTION_ID = 2,
    TRIPLE_REDUCTION_ID = 3,
    MMS_REDUCTION_ID = 4,
  };
  enum SnapFieldID {
    FID_SINGLE = 0, // For field spaces with just one field
    // Fields for energy groups
    FID_GROUP_0 = FID_SINGLE,
    // ...
    FID_GROUP_MAX = FID_GROUP_0 + SNAP_MAX_ENERGY_GROUPS,
    FID_FLUX_START = FID_GROUP_MAX,
    FID_FLUX_MAX = FID_FLUX_START + 8/*corners*/*SNAP_MAX_ENERGY_GROUPS,
  };
#define SNAP_ENERGY_GROUP_FIELD(group)    \
  ((Snap::SnapFieldID)(Snap::FID_GROUP_0 + (group)))
#define SNAP_FLUX_GROUP_FIELD(group, corner)          \
  ((Snap::SnapFieldID)(Snap::FID_FLUX_START + (group * 8) + corner))
  enum SnapPartitionID {
    DISJOINT_PARTITION = 0,
  };
  enum SnapProjectionID {
    SWEEP_PROJECTION = 1,
    XY_PROJECTION = 2,
    YZ_PROJECTION = 3,
    XZ_PROJECTION = 4,
  };
public:
  Snap(Context c, Runtime *rt)
    : ctx(c), runtime(rt) { }
public:
  inline const Rect<3>& get_simulation_bounds(void) const 
    { return simulation_bounds; }
  inline const Rect<3>& get_launch_bounds(void) const
    { return launch_bounds; }
public:
  void setup(void);
  void transport_solve(void);
protected:
  void initialize_scattering(const SnapArray &sigt, const SnapArray &siga,
                             const SnapArray &sigs, const SnapArray &slgg) const;
  void initialize_velocity(const SnapArray &vel, const SnapArray &vdelt) const;
  void save_fluxes(const Predicate &pred,
                   const SnapArray &src, const SnapArray &dst) const;
  void perform_sweeps(const Predicate &pred, const SnapArray &flux,
                      const SnapArray &fluxm, const SnapArray &qtot, 
                      const SnapArray &vdelt, const SnapArray &dinv, 
                      const SnapArray &t_xs, SnapArray *time_flux_in[8], 
                      SnapArray *time_flux_out[8], SnapArray *qim[8],
                      const SnapArray &flux_xy, const SnapArray &flux_yz,
                      const SnapArray &flux_xz, int energy_group_chunks) const;
  Predicate test_inner_convergence(const Predicate &pred, const SnapArray &flux0,
                      const SnapArray &flux0pi, const Future &pred_false_result,
                      int energy_group_chunks) const;
  Predicate test_outer_convergence(const Predicate &pred, const SnapArray &flux0,
                      const SnapArray &flux0po, const Future &inner_converged,
                      const Future &pred_false_result,
                      int energy_group_chunks) const;
private:
  const Context ctx;
  Runtime *const runtime;
private:
  // Simulation bounds
  Rect<3> simulation_bounds;
  Rect<3> launch_bounds;
private:
  IndexSpace simulation_is;
  IndexPartition spatial_ip;
  IndexSpace material_is;
  IndexSpace slgg_is;
  IndexSpace point_is;
  IndexSpace xy_flux_is;
  IndexPartition xy_flux_ip;
  IndexSpace yz_flux_is;
  IndexPartition yz_flux_ip;
  IndexSpace xz_flux_is;
  IndexPartition xz_flux_ip;
private:
  FieldSpace group_fs;
  FieldSpace flux_fs;
  FieldSpace moment_fs;
  FieldSpace flux_moment_fs;
  FieldSpace mat_fs;
  FieldSpace angle_fs;
private:
  std::vector<Domain> wavefront_domains[8];
public:
  static void snap_top_level_task(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx, Runtime *runtime); 
public:
  static void parse_arguments(int argc, char **argv);
  static void compute_wavefronts(void);
  static void compute_derived_globals(void);
  static void report_arguments(void);
  static void perform_registrations(void);
  static void mapper_registration(Machine machine, Runtime *runtime,
                                  const std::set<Processor> &local_procs);
  static LayoutConstraintID get_soa_layout(void);
  static LayoutConstraintID get_reduction_layout(void);
public:
  // Configuration parameters read from input file
  static int num_dims; // originally ndimen 1-3
  static int nx_chunks; // originally ichunk 1 <= # <= nx
  static int ny_chunks; // originally npey 1 <= # <= ny
  static int nz_chunks; // originally npez 1 <= # <= nz
  static int nx; // 4 <= #
  static double lx; // 0.0 < #
  static int ny; // 4 <= #
  static double ly; // 0.0 < #
  static int nz; // 4 <= #
  static double lz; // 0.0 < #
  static int num_moments; // originally nmom 1 <= # <= 4
  static int num_angles; // originally nang 1 <= #
  static int num_groups; // originally ng 1 <= #
  static double convergence_eps; // originally epsi 0.0 < # < 1e-2
  static int max_inner_iters; // originally iitm 1 <= # 
  static int max_outer_iters; // originally oitm 1 <= #
  static bool time_dependent; // originally timedep
  static double total_sim_time; // originally tf 0.0 <= # if time dependent
  static int num_steps; // originally nsteps 1 <= #
  static MaterialLayout material_layout; // originally mat_opt
  static SourceLayout source_layout; // originally src_opt
  static bool dump_scatter; // originally scatp
  static bool dump_iteration; // originally it_dep
  static int dump_flux; // originally fluxp 0,1,2
  static bool flux_fixup; // originally fixup
  static bool dump_solution; // originally soloutp
  static int dump_kplane; // originally kplane 0,1,2
  static int dump_population;  // originally popout
  static bool minikba_sweep; // originally swp_typ
  static bool single_angle_copy; // originally angcpy
public: // derived
  static int num_corners; // orignally ncor
  static int nx_per_chunk;
  static int ny_per_chunk;
  static int nz_per_chunk;
  // Indexed by wavefront number and the point number
  static std::vector<std::vector<DomainPoint> > wavefront_map[8];
public:
  static double dt; 
  static int cmom;
  static int num_octants;
  static double hi, hj, hk;
  static double *mu; // num angles
  static double *w; // num angles
  static double *wmu; // num angles
  static double *eta; // num angles
  static double *weta; // num angles
  static double *xi; // num angles
  static double *wxi; // num angles
  static double *ec; // num angles x num moments x num_octants
  static double *dinv; // num_angles x nx x ny x nz x 
  static int lma[4];
public:
  // Snap mapper derived from the default mapper
  class SnapMapper : public Legion::Mapping::DefaultMapper {
  public:
    SnapMapper(MapperRuntime *rt, Machine machine, Processor local,
               const char *mapper_name);
  public:
    virtual void select_tunable_value(const MapperContext ctx,
                                      const Task& task,
                                      const SelectTunableInput& input,
                                            SelectTunableOutput& output);
    virtual void speculate(const MapperContext ctx,
                           const Copy &copy,
                                 SpeculativeOutput &output);
    virtual void map_copy(const MapperContext ctx,
                          const Copy &copy,
                          const MapCopyInput &input,
                                MapCopyOutput &output);
    virtual void select_task_options(const MapperContext ctx,
                                     const Task& task,
                                           TaskOptions& options);
    virtual void slice_task(const MapperContext ctx,
                            const Task &task,
                            const SliceTaskInput &input,
                                  SliceTaskOutput &output);
    virtual void speculate(const MapperContext ctx,
                           const Task &task,
                                 SpeculativeOutput &output);
    virtual void map_task(const MapperContext ctx,
                          const Task &task,
                          const MapTaskInput &input,
                                MapTaskOutput &output);
  protected:
    void update_variants(const MapperContext ctx);
    void map_snap_array(const MapperContext ctx, 
                        LogicalRegion region, Memory target,
                        std::vector<PhysicalInstance> &instances);
#ifdef LOCAL_MAP_TASKS
  protected:
    Memory get_associated_sysmem(Processor proc);
    Memory get_associated_framebuffer(Processor proc);
    Memory get_associated_zerocopy(Processor proc);
    void get_associated_procs(Processor proc, std::vector<Processor> &procs);
#endif
  protected:
    bool has_variants;
    std::map<SnapTaskID,VariantID> cpu_variants;
    std::map<SnapTaskID,VariantID> gpu_variants;
  protected:
    Memory local_sysmem, local_zerocopy, local_framebuffer;
    std::map<std::pair<LogicalRegion,Memory>,PhysicalInstance> local_instances;
    // Copy instances always go in the system memory
    std::map<LogicalRegion,PhysicalInstance> copy_instances;
#ifdef LOCAL_MAP_TASKS
    std::map<Processor,Memory> associated_sysmems;
    std::map<Processor,Memory> associated_framebuffers;
    std::map<Processor,Memory> associated_zerocopy;
    std::map<Processor,std::vector<Processor> > associated_procs;
#endif
  protected:
    std::map<Point<3>,Processor,Point<3>::STLComparator> global_cpu_mapping;
    std::map<Point<3>,Processor,Point<3>::STLComparator> global_gpu_mapping;
  };
};

template<typename T, Snap::SnapTaskID TASK_ID> 
class SnapTask : public IndexLauncher {
public:
  SnapTask(const Snap &snap, const Rect<3> &launch_domain, const Predicate &pred)
    : IndexLauncher(TASK_ID, Domain::from_rect<3>(launch_domain), 
                    TaskArgument(), ArgumentMap(), pred) { }
public:
  void dispatch(Context ctx, Runtime *runtime, bool block = false)
  { 
    log_snap.info("Dispatching Task %s (ID %d)", 
        Snap::task_names[TASK_ID], TASK_ID);
    if (block) {
      FutureMap fm = runtime->execute_index_space(ctx, *this);
      fm.wait_all_results(true/*silence warnings*/);
    } else
      runtime->execute_index_space(ctx, *this);
  }
  template<typename OP>
  Future dispatch(Context ctx, Runtime *runtime, bool block = false)
  {
    log_snap.info("Dispatching Task %s (ID %d) with Reduction %d", 
                  Snap::task_names[TASK_ID], TASK_ID, OP::REDOP);
    if (block) {
      Future f = runtime->execute_index_space(ctx, *this, OP::REDOP);
      f.get_void_result(true/*silence warnings*/);
      return f;
    } else
      return runtime->execute_index_space(ctx, *this, OP::REDOP);
  }
public:
  static void preregister_all_variants(void)
  {
    T::preregister_cpu_variants();
    T::preregister_gpu_variants();
  }
  static void register_task_name(Runtime *runtime)
  {
    runtime->attach_name(TASK_ID, Snap::task_names[TASK_ID]);
  }
public:
  template<void (*TASK_PTR)(const Task*,
      const std::vector<PhysicalRegion>&, Context, Runtime*)>
  static void snap_task_wrapper(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
  {
    log_snap.info("Running Task %s (UID %lld) on Processor " IDFMT "",
        task->get_task_name(), task->get_unique_id(), 
        runtime->get_executing_processor(ctx).id);
    (*TASK_PTR)(task, regions, ctx, runtime);
  }
  template<typename RET_T, RET_T (*TASK_PTR)(const Task*,
      const std::vector<PhysicalRegion>&, Context, Runtime*)>
  static RET_T snap_task_wrapper(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
  {
    log_snap.info("Running Task %s (UID %lld) on Processor " IDFMT "",
        task->get_task_name(), task->get_unique_id(), 
        runtime->get_executing_processor(ctx).id);
    RET_T result = (*TASK_PTR)(task, regions, ctx, runtime);
    return result;
  }
protected:
  // For registering CPU variants
  template<void (*TASK_PTR)(const Task*,
      const std::vector<PhysicalRegion>&, Context, Runtime*)>
  static void register_cpu_variant(const ExecutionConstraintSet &execution_constraints,
                                   const TaskLayoutConstraintSet &layout_constraints,
                                   bool leaf = false, bool inner = false)
  {
    char variant_name[128];
    strcpy(variant_name, "CPU ");
    strncat(variant_name, Snap::task_names[TASK_ID], 123);
    TaskVariantRegistrar registrar(TASK_ID, true/*global*/,
        NULL/*generator*/, variant_name);
    registrar.execution_constraints = execution_constraints;
    registrar.layout_constraints = layout_constraints;
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.leaf_variant = leaf;
    registrar.inner_variant = inner;
    Runtime::preregister_task_variant<
      SnapTask<T,TASK_ID>::template snap_task_wrapper<TASK_PTR> >(
          registrar, Snap::task_names[TASK_ID]);
  }
  template<typename RET_T, RET_T (*TASK_PTR)(const Task*,
      const std::vector<PhysicalRegion>&, Context, Runtime*)>
  static void register_cpu_variant(const ExecutionConstraintSet &execution_constraints,
                                   const TaskLayoutConstraintSet &layout_constraints,
                                   bool leaf = false, bool inner = false)
  {
    char variant_name[128];
    strcpy(variant_name, "CPU ");
    strncat(variant_name, Snap::task_names[TASK_ID], 123);
    TaskVariantRegistrar registrar(TASK_ID, true/*global*/,
        NULL/*generator*/, variant_name);
    registrar.execution_constraints = execution_constraints;
    registrar.layout_constraints = layout_constraints;
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.leaf_variant = leaf;
    registrar.inner_variant = inner;
    Runtime::preregister_task_variant<RET_T,
      SnapTask<T,TASK_ID>::template snap_task_wrapper<RET_T,TASK_PTR> >(
                                         registrar, Snap::task_names[TASK_ID]);
  }
protected:
  // For registering GPU variants
  template<void (*TASK_PTR)(const Task*,
      const std::vector<PhysicalRegion>&, Context, Runtime*)>
  static void register_gpu_variant(const ExecutionConstraintSet &execution_constraints,
                                   const TaskLayoutConstraintSet &layout_constraints,
                                   bool leaf = false, bool inner = false)
  {
    char variant_name[128];
    strcpy(variant_name, "GPU ");
    strncat(variant_name, Snap::task_names[TASK_ID], 123);
    TaskVariantRegistrar registrar(TASK_ID, true/*global*/,
        NULL/*generator*/, variant_name);
    registrar.execution_constraints = execution_constraints;
    registrar.layout_constraints = layout_constraints;
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.leaf_variant = leaf;
    registrar.inner_variant = inner;
    Runtime::preregister_task_variant<
      SnapTask<T,TASK_ID>::template snap_task_wrapper<TASK_PTR> >(
          registrar, Snap::task_names[TASK_ID]);
  }
  template<typename RET_T, RET_T (*TASK_PTR)(const Task*,
      const std::vector<PhysicalRegion>&, Context, Runtime*)>
  static void register_gpu_variant(const ExecutionConstraintSet &execution_constraints,
                                   const TaskLayoutConstraintSet &layout_constraints,
                                   bool leaf = false, bool inner = false)
  {
    char variant_name[128];
    strcpy(variant_name, "GPU ");
    strncat(variant_name, Snap::task_names[TASK_ID], 123);
    TaskVariantRegistrar registrar(TASK_ID, true/*global*/,
        NULL/*generator*/, variant_name);
    registrar.execution_constraints = execution_constraints;
    registrar.layout_constraints = layout_constraints;
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.leaf_variant = leaf;
    registrar.inner_variant = inner;
    Runtime::preregister_task_variant<RET_T,
      SnapTask<T,TASK_ID>::template snap_task_wrapper<RET_T,TASK_PTR> >(
                                         registrar, Snap::task_names[TASK_ID]);
  }
};

class SnapArray {
public:
  SnapArray(IndexSpace is, IndexPartition ip, FieldSpace fs, 
            Context ctx, Runtime *runtime, const char *name);
  ~SnapArray(void);
private:
  SnapArray(const SnapArray &rhs);
  SnapArray& operator=(const SnapArray &rhs);
public:
  inline LogicalRegion get_region(void) const { return lr; }
  inline LogicalPartition get_partition(void) const { return lp; }
  inline const std::set<FieldID>& get_all_fields(void) const 
    { return all_fields; }
  LogicalRegion get_subregion(const DomainPoint &color) const;
public:
  void initialize(Predicate pred = Predicate::TRUE_PRED) const; 
  template<typename T>
  void initialize(T value, Predicate pred = Predicate::TRUE_PRED) const;
  PhysicalRegion map(void) const;
  void unmap(const PhysicalRegion &region) const;
public:
  template<typename T>
  inline void add_projection_requirement(PrivilegeMode priv,
                                         T &launcher) const
  {
    launcher.add_region_requirement(RegionRequirement(lp, 0/*proj id*/,
                                                      priv, EXCLUSIVE, lr));
    launcher.region_requirements.back().privilege_fields = all_fields;
  }
  template<typename T>
  inline void add_projection_requirement(PrivilegeMode priv, T &launcher,
                        Snap::SnapFieldID field, ProjectionID proj_id = 0) const
  {
    launcher.add_region_requirement(RegionRequirement(lp, proj_id, priv,
                                                      EXCLUSIVE, lr));
    launcher.region_requirements.back().privilege_fields.insert(field);
  }
  template<typename T>
  inline void add_projection_requirement(PrivilegeMode priv, T &launcher,
   const std::vector<Snap::SnapFieldID> &fields, ProjectionID proj_id = 0) const
  {
    launcher.add_region_requirement(RegionRequirement(lp, proj_id, priv,
                                                      EXCLUSIVE, lr));
    launcher.region_requirements.back().privilege_fields.insert(
                                        fields.begin(), fields.end());
  }
  template<typename T>
  inline void add_projection_requirement(T &launcher, Snap::SnapReductionID reduction,
      Snap::SnapFieldID field, ProjectionID proj_id = 0) const
  {
    launcher.add_region_requirement(RegionRequirement(lp, proj_id, reduction,
                                                      EXCLUSIVE, lr));
    launcher.region_requirements.back().privilege_fields.insert(field);
  }
  template<typename T>
  inline void add_projection_requirement(T &launcher, Snap::SnapReductionID reduction,
      const std::vector<Snap::SnapFieldID> &fields, ProjectionID proj_id = 0) const
  {
    launcher.add_region_requirement(RegionRequirement(lp, proj_id, reduction,
                                                      EXCLUSIVE, lr));
    launcher.region_requirements.back().privilege_fields.insert(
                                        fields.begin(), fields.end());
  }
  template<typename T>
  inline void add_region_requirement(PrivilegeMode priv,
                                     T &launcher) const
  {
    launcher.add_region_requirement(RegionRequirement(lr, priv, EXCLUSIVE, lr));
    launcher.region_requirements.back().privilege_fields = all_fields;
  }
  template<typename T>
  inline void add_region_requirement(PrivilegeMode priv, T &launcher,
                                     Snap::SnapFieldID field) const
  {
    launcher.add_region_requirement(RegionRequirement(lr, priv, EXCLUSIVE, lr));
    launcher.region_requirements.back().privilege_fields.insert(field);
  }
  template<typename T>
  inline void add_region_requirement(PrivilegeMode priv, T &launcher,
                  const std::vector<Snap::SnapFieldID> &fields) const
  {
    launcher.add_region_requirement(RegionRequirement(lr, priv, EXCLUSIVE, lr));
    launcher.region_requirements.back().privilege_fields.insert(
                                                  fields.begin(), fields.end());
  }
protected:
  const Context ctx;
  Runtime *const runtime;
protected:
  LogicalRegion lr;
  LogicalPartition lp;
  std::set<FieldID> all_fields;
  Domain color_space;
  mutable std::map<DomainPoint,LogicalRegion> subregions;
  void *fill_buffer;
  size_t field_size;
};

class SnapSweepProjectionFunctor : public ProjectionFunctor {
public:
  SnapSweepProjectionFunctor(void);
public:
  virtual LogicalRegion project(const Mappable *mappable, unsigned index,
                                LogicalRegion upper_bound,
                                const DomainPoint &point);
  virtual LogicalRegion project(const Mappable *mappable, unsigned index,
                                LogicalPartition upper_bound,
                                const DomainPoint &point);
  virtual unsigned get_depth(void) const { return 0; }
};

class FluxProjectionFunctor : public ProjectionFunctor {
public:
  FluxProjectionFunctor(Snap::SnapProjectionID kind);
public:
  virtual LogicalRegion project(const Mappable *mappable, unsigned index,
                                LogicalRegion upper_bound,
                                const DomainPoint &point);
  virtual LogicalRegion project(const Mappable *mappable, unsigned index,
                                LogicalPartition upper_bound,
                                const DomainPoint &point);
  virtual unsigned get_depth(void) const { return 0; }
public:
  const Snap::SnapProjectionID projection_kind;
};

class AndReduction {
public:
  static const Snap::SnapReductionID REDOP = Snap::AND_REDUCTION_ID;
public:
  typedef bool LHS;
  typedef bool RHS;
  static const bool identity = true;
public:
  template<bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);
  template<bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
};

class SumReduction {
public:
  static const Snap::SnapReductionID REDOP = Snap::SUM_REDUCTION_ID;
public:
  typedef double LHS;
  typedef double RHS;
  static const double identity;
public:
  template<bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);
  template<bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
};

class TripleReduction {
public:
  static const Snap::SnapReductionID REDOP = Snap::TRIPLE_REDUCTION_ID;
public:
  typedef MomentTriple LHS;
  typedef MomentTriple RHS;
  static const MomentTriple identity;
public:
  template<bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);
  template<bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
};

template<int DIM>
inline bool offsets_match(const LegionRuntime::Accessor::ByteOffset x[DIM], 
                          const LegionRuntime::Accessor::ByteOffset y[DIM])
{
  for (int i = 0; i < DIM; i++) {
    if (x[i].offset != y[i].offset)
      return false;
  }
  return true;
}

#endif // __SNAP_H__


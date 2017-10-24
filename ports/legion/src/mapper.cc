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

//------------------------------------------------------------------------------
Snap::SnapMapper::SnapMapper(MapperRuntime *rt, Machine machine, 
                             Processor local, const char *mapper_name)
  : DefaultMapper(rt, machine, local, mapper_name), has_variants(false)
//------------------------------------------------------------------------------
{
  // Get our local memories
  {
    Machine::MemoryQuery sysmem_query(machine);
    sysmem_query.local_address_space();
    sysmem_query.only_kind(Memory::SYSTEM_MEM);
    local_sysmem = sysmem_query.first();
    assert(local_sysmem.exists());
  }
  if (!local_gpus.empty()) {
    Machine::MemoryQuery zc_query(machine);
    zc_query.local_address_space();
    zc_query.only_kind(Memory::Z_COPY_MEM);
    local_zerocopy = zc_query.first();
    assert(local_zerocopy.exists());
  } else {
    local_zerocopy = Memory::NO_MEMORY;
  }
  if (local_kind == Processor::TOC_PROC) {
    Machine::MemoryQuery fb_query(machine);
    fb_query.local_address_space();
    fb_query.only_kind(Memory::GPU_FB_MEM);
    fb_query.best_affinity_to(local_proc);
    local_framebuffer = fb_query.first();
    assert(local_framebuffer.exists());
  } else {
    local_framebuffer = Memory::NO_MEMORY;
  }
  // Compute the local CPU and GPU mappings
  // TODO: make these topology aware
  const int upper_bounds[3] = { nx_chunks, ny_chunks, nz_chunks };
  const Rect<3> bounds(Point<3>::ZEROES(), Point<3>(upper_bounds));
  {
    // Round robin these across nodes, not individual processors so we
    // evenly distribute blocks of cells across the machine, then let 
    // individual nodes use field parallelism across processors
    std::map<AddressSpace,Processor> node_cpus;
    for (std::vector<Processor>::const_iterator it = remote_cpus.begin();
          it != remote_cpus.end(); it++)
    {
      const AddressSpace space = it->address_space();
      if (node_cpus.find(space) == node_cpus.end())
        node_cpus[space] = *it;
    }
    for (GenericPointInRectIterator<3> pir(bounds); pir; pir++) {
      const int index = 
        (pir.p.x[2] * ny_chunks + pir.p.x[1]) * nx_chunks + pir.p.x[0];
      std::map<AddressSpace,Processor>::const_iterator finder = 
        node_cpus.find(index % node_cpus.size());
      assert(finder != node_cpus.end());
      global_cpu_mapping[pir.p] = finder->second;
    }
  }
  {
    // Round robin these across all GPUs
    Machine::ProcessorQuery all_procs(machine);
    all_procs.only_kind(Processor::TOC_PROC);
    if (all_procs.count() > 0) {
      std::vector<Processor> all_gpus(all_procs.count());
      unsigned idx = 0;
      for (Machine::ProcessorQuery::iterator it = all_procs.begin();
            it != all_procs.end(); it++, idx++)
        all_gpus[idx] = *it;
      for (GenericPointInRectIterator<3> pir(bounds); pir; pir++) {
        const int index = 
          (pir.p.x[2] * ny_chunks + pir.p.x[1]) * nx_chunks + pir.p.x[0];
        global_gpu_mapping[pir.p] = all_gpus[index % all_gpus.size()];
      }
    }
  }
}

//------------------------------------------------------------------------------
void Snap::SnapMapper::select_tunable_value(const MapperContext ctx,
                                            const Task& task,
                                            const SelectTunableInput& input,
                                                  SelectTunableOutput& output)
//------------------------------------------------------------------------------
{
  switch (input.tunable_id)
  {
    case OUTER_RUNAHEAD_TUNABLE:
      {
        // Just need to unroll this enough to avoid blocking
        // We still might see some bubbles across time steps
        // but that should be relatively minor
        runtime->pack_tunable<unsigned>(2, output);
        break;
      }
    case INNER_RUNAHEAD_TUNABLE:
      {
        // Always fully unroll the inner loop so we can see
        // to the next outer loop
        runtime->pack_tunable<unsigned>(Snap::max_inner_iters, output);
        break;
      }
    case SWEEP_ENERGY_CHUNKS_TUNABLE:
      {
        // 8 directions * number of energy fields should be larger
        // then the number of processors in a node since we use field 
        // parallelism to keep all the processors in a node busy 
        if (local_gpus.empty()) {
          // Mapping to CPUs only
          const int num_cpus = local_cpus.size();
          int result = 8/*directions*/ * Snap::num_groups / num_cpus;
          // Clamp it at the number of groups if necessary
          if (result > Snap::num_groups)
            result = Snap::num_groups;
          runtime->pack_tunable<int>(result, output);
        } else {
          // Mapping to GPUs
          const int num_gpus = local_gpus.size();
          int result = 8/*directions*/ * Snap::num_groups / num_gpus;
          // Clamp it at the number of groups if necessary
          if (result > Snap::num_groups)
            result = Snap::num_groups;
          runtime->pack_tunable<int>(result, output);
        }
        break;
      }
    default:
      // Fall back to the default mapper
      DefaultMapper::select_tunable_value(ctx, task, input, output);
  }
}

//------------------------------------------------------------------------------
void Snap::SnapMapper::speculate(const MapperContext ctx,
                                 const Copy &copy,
                                       SpeculativeOutput &output)
//------------------------------------------------------------------------------
{
#ifdef ENABLE_SPECULATION
  output.speculate = true;
  output.speculative_value = true; // not converged
  output.speculate_mapping_only = true;
#else
  output.speculate = false;
#endif
}

//------------------------------------------------------------------------------
void Snap::SnapMapper::map_copy(const MapperContext ctx,
                                const Copy &copy,
                                const MapCopyInput &input,
                                      MapCopyOutput &output)
//------------------------------------------------------------------------------
{
  assert(!copy.is_index_space);
  // See if we already know where the copy is going
  for (unsigned idx = 0; idx < copy.src_requirements.size(); idx++)
  {
    // Source first
    const LogicalRegion &src_region = copy.src_requirements[idx].region;
    std::map<LogicalRegion,PhysicalInstance>::const_iterator finder = 
      copy_instances.find(src_region);
    if (finder == copy_instances.end()) {
      // Didn't find it, so we have to make it
      // First figure out which memory it is going into
      assert(copy.index_point.get_dim() == 3);
      Point<3> point = copy.index_point.get_point<3>();
      assert(global_cpu_mapping.find(point) != global_cpu_mapping.end());
      Processor cpu_proc = global_cpu_mapping[point];
      // Find the target memory with affinity to the proper node
      Machine::MemoryQuery target_query(machine);
      target_query.has_affinity_to(cpu_proc);
      target_query.only_kind(Memory::SYSTEM_MEM);
      Memory target = target_query.first();
      assert(target.exists());
      // Now we can make the instance
      std::vector<LogicalRegion> regions(1, src_region);  
      LayoutConstraintSet layout_constraints;
      // No specialization
      layout_constraints.add_constraint(SpecializedConstraint());
      // SOA-Fortran dimension ordering
      std::vector<DimensionKind> dimension_ordering(4);
      dimension_ordering[0] = DIM_X;
      dimension_ordering[1] = DIM_Y;
      dimension_ordering[2] = DIM_Z;
      dimension_ordering[3] = DIM_F;
      layout_constraints.add_constraint(OrderingConstraint(dimension_ordering, 
                                                           false/*contiguous*/));
      // Constrained for the target memory kind
      layout_constraints.add_constraint(MemoryConstraint(target.kind()));
      // Have all the field for the instance available
      std::vector<FieldID> all_fields;
      runtime->get_field_space_fields(ctx, src_region.get_field_space(), all_fields);
      layout_constraints.add_constraint(FieldConstraint(all_fields, false/*contiguous*/,
                                                        false/*inorder*/));
      PhysicalInstance result; bool created;
      if (!runtime->find_or_create_physical_instance(ctx, target, layout_constraints,
            regions, result, created, true/*acquire*/, GC_NEVER_PRIORITY)) {
        log_snap.error("ERROR: SNAP mapper failed to allocate instance for copy");
        assert(false);
      }
      output.src_instances[idx].push_back(result);
      // Save it for the next time
      assert((copy_instances.find(src_region) == copy_instances.end()) ||
              (copy_instances[src_region] == result));;
      copy_instances[src_region] = result;
    } else {
      // Found it, add it to the set
      output.src_instances[idx].push_back(finder->second);
    }
    const LogicalRegion &dst_region = copy.dst_requirements[idx].region;
    finder = copy_instances.find(dst_region);
    if (finder == copy_instances.end()) {
      // Didn't find it so we have to make it
      // First figure out which memory it is going into
      assert(copy.index_point.get_dim() == 3);
      Point<3> point = copy.index_point.get_point<3>();
      assert(global_cpu_mapping.find(point) != global_cpu_mapping.end());
      Processor cpu_proc = global_cpu_mapping[point];
      // Find the target memory with affinity to the proper node
      Machine::MemoryQuery target_query(machine);
      target_query.has_affinity_to(cpu_proc);
      target_query.only_kind(Memory::SYSTEM_MEM);
      Memory target = target_query.first();
      assert(target.exists());
      // Now we can make the instance
      std::vector<LogicalRegion> regions(1, dst_region);  
      LayoutConstraintSet layout_constraints;
      // No specialization
      layout_constraints.add_constraint(SpecializedConstraint());
      // SOA-Fortran dimension ordering
      std::vector<DimensionKind> dimension_ordering(4);
      dimension_ordering[0] = DIM_X;
      dimension_ordering[1] = DIM_Y;
      dimension_ordering[2] = DIM_Z;
      dimension_ordering[3] = DIM_F;
      layout_constraints.add_constraint(OrderingConstraint(dimension_ordering, 
                                                           false/*contiguous*/));
      // Constrained for the target memory kind
      layout_constraints.add_constraint(MemoryConstraint(target.kind()));
      // Have all the field for the instance available
      std::vector<FieldID> all_fields;
      runtime->get_field_space_fields(ctx, dst_region.get_field_space(), all_fields);
      layout_constraints.add_constraint(FieldConstraint(all_fields, false/*contiguous*/,
                                                        false/*inorder*/));
      PhysicalInstance result; bool created;
      if (!runtime->find_or_create_physical_instance(ctx, target, layout_constraints,
            regions, result, created, true/*acquire*/, GC_NEVER_PRIORITY)) {
        log_snap.error("ERROR: SNAP mapper failed to allocate instance for copy");
        assert(false);
      }
      output.dst_instances[idx].push_back(result);
      // Save it for the next time
      assert((copy_instances.find(dst_region) == copy_instances.end()) ||
              (copy_instances[dst_region] == result));
      copy_instances[dst_region] = result;
    } else {
      // Found it, add it to the set
      output.dst_instances[idx].push_back(finder->second);
    }
  }
  runtime->acquire_instances(ctx, output.src_instances);
  runtime->acquire_instances(ctx, output.dst_instances);
}

//------------------------------------------------------------------------------
void Snap::SnapMapper::select_task_options(const MapperContext ctx,
                                           const Task& task,
                                                 TaskOptions& options)
//------------------------------------------------------------------------------
{
  options.initial_proc = default_policy_select_initial_processor(ctx, task);
  options.inline_task = false;
  options.stealable = false;
#ifdef LOCAL_MAP_TASKS
  options.map_locally = true;
#else
  options.map_locally = false;
#endif
}


//------------------------------------------------------------------------------
void Snap::SnapMapper::slice_task(const MapperContext ctx,
                                  const Task &task, 
                                  const SliceTaskInput &input,
                                        SliceTaskOutput &output)
//------------------------------------------------------------------------------
{
  if (!has_variants)
    update_variants(ctx);
  // Sweep tasks compute their target processors differently than
  // all the other data parallel tasks
  if (task.task_id == MINI_KBA_TASK_ID) {
    // Figure out 3-D point from corner, and wavefront
    const MiniKBATask::MiniKBAArgs *args = 
      (const MiniKBATask::MiniKBAArgs*)task.args;  
    const std::vector<DomainPoint> &wavefront_points = 
      wavefront_map[args->corner][args->wavefront];
    const bool use_gpu = !local_gpus.empty() &&
      (gpu_variants.find(MINI_KBA_TASK_ID) != gpu_variants.end());
    Rect<1> all_points = input.domain.get_rect<1>();
    for (GenericPointInRectIterator<1> pir(all_points); pir; pir++) {
      // Get the physical space point
      Point<3> point = wavefront_points[pir.p[0]].get_point<3>();
      TaskSlice slice;
      slice.domain = Domain::from_rect<1>(Rect<1>(pir.p, pir.p));
      slice.proc = use_gpu ? global_gpu_mapping[point] : global_cpu_mapping[point];
      slice.recurse = false;
      slice.stealable = false;
      output.slices.push_back(slice);
    }
  } else if (task.task_id == INIT_GPU_SWEEP_TASK_ID) {
    // This one needs the default mapper implementation
    DefaultMapper::slice_task(ctx, task, input, output);
  } else {
    // Iterate over the points and assign them to the best target processors
    Rect<3> all_points = input.domain.get_rect<3>();
    // We still keep convergence tests on the CPU if we're doing reductions
#ifndef SNAP_USE_RELAXED_COHERENCE
    const bool use_gpu = !local_gpus.empty() &&
      (gpu_variants.find((SnapTaskID)task.task_id) != gpu_variants.end()) &&
      (task.task_id != TEST_OUTER_CONVERGENCE_TASK_ID) && 
      (task.task_id != TEST_INNER_CONVERGENCE_TASK_ID);;
#else
    const bool use_gpu = !local_gpus.empty() &&
      (gpu_variants.find((SnapTaskID)task.task_id) != gpu_variants.end());
#endif
    for (GenericPointInRectIterator<3> pir(all_points); pir; pir++) {
      TaskSlice slice;
      slice.domain = Domain::from_rect<3>(Rect<3>(pir.p, pir.p));
      slice.proc = use_gpu ? global_gpu_mapping[pir.p] : global_cpu_mapping[pir.p];
      slice.recurse = false;
      slice.stealable = false;
      output.slices.push_back(slice);
    }
  }
}

//------------------------------------------------------------------------------
void Snap::SnapMapper::speculate(const MapperContext ctx,
                                 const Task &task,
                                       SpeculativeOutput &output)
//------------------------------------------------------------------------------
{
#ifdef DISABLE_SPECULATION
  output.speculate = false;
#else
  output.speculate = true;
  output.speculative_value = true; // not converged
  output.speculate_mapping_only = true;
#endif
}

//------------------------------------------------------------------------------
void Snap::SnapMapper::map_task(const MapperContext ctx,
                                const Task &task,
                                const MapTaskInput &input,
                                      MapTaskOutput &output)
//------------------------------------------------------------------------------
{
  if (!has_variants)
    update_variants(ctx);
  // Assume we are mapping on the target processor
#ifndef LOCAL_MAP_TASKS
  assert(task.target_proc == local_proc);
#endif
  output.chosen_instances.resize(task.regions.size());
  switch (task.task_id)
  {
    // These tasks go on the GPU if possible, otherwise CPU
    case INIT_MATERIAL_TASK_ID:
    case INIT_SOURCE_TASK_ID:
    case CALC_OUTER_SOURCE_TASK_ID:
    case CALC_INNER_SOURCE_TASK_ID:
    case EXPAND_CROSS_SECTION_TASK_ID:
    case EXPAND_SCATTERING_CROSS_SECTION_TASK_ID:
    case CALCULATE_GEOMETRY_PARAM_TASK_ID:
    case MMS_SCALE_TASK_ID:
      {
        Memory target_mem;
        std::map<SnapTaskID,VariantID>::const_iterator finder = 
          gpu_variants.find((SnapTaskID)task.task_id);
        if (finder != gpu_variants.end() && 
            (local_kind == Processor::TOC_PROC)) {
          output.chosen_variant = finder->second; 
#ifdef LOCAL_MAP_TASKS
          output.target_procs.push_back(task.target_proc);
          target_mem = get_associated_framebuffer(task.target_proc);
#else
          output.target_procs.push_back(local_proc);
          target_mem = local_framebuffer;
#endif
        } else {
          output.chosen_variant = cpu_variants[(SnapTaskID)task.task_id];
#ifdef LOCAL_MAP_TASKS
          get_associated_procs(task.target_proc, output.target_procs);
          target_mem = get_associated_sysmem(task.target_proc);
#else
          output.target_procs = local_cpus;
          target_mem = local_sysmem;
#endif
        }
        for (unsigned idx = 0; idx < task.regions.size(); idx++) { 
          if (task.regions[idx].privilege == NO_ACCESS)
            continue;
          map_snap_array(ctx, task.regions[idx].region, target_mem,
                         output.chosen_instances[idx]);
        }
        break;
      }
    // Convergence tests go on CPU always since reductions are coming
    // from zero-copy memory anyway and we want to avoid using
    // too much duplicate memory for storing flux0po
    case TEST_OUTER_CONVERGENCE_TASK_ID:
    case TEST_INNER_CONVERGENCE_TASK_ID:
      {
        output.chosen_variant = cpu_variants[(SnapTaskID)task.task_id];
#ifdef LOCAL_MAP_TASKS
        get_associated_procs(task.target_proc, output.target_procs);
        Memory target_mem = get_associated_sysmem(task.target_proc);
#else
        output.target_procs = local_cpus;
        Memory target_mem = local_sysmem;   
#endif
        for (unsigned idx = 0; idx < task.regions.size(); idx++)
          map_snap_array(ctx, task.regions[idx].region, target_mem, 
                         output.chosen_instances[idx]);
        break;
      }
    case BIND_INNER_CONVERGENCE_TASK_ID:
    case BIND_OUTER_CONVERGENCE_TASK_ID:
      {
        // These tasks have no region requirements so they 
        // can go wherever on the cpus
        output.chosen_variant = cpu_variants[(SnapTaskID)task.task_id];
#ifdef LOCAL_MAP_TASKS
        get_associated_procs(task.target_proc, output.target_procs);
#else
        output.target_procs = local_cpus;
#endif
        break;
      }
    case MINI_KBA_TASK_ID:
      {
        // Mini KBA is special
        Memory target_mem, reduction_mem, vdelt_mem;
        std::map<SnapTaskID,VariantID>::const_iterator finder = 
          gpu_variants.find((SnapTaskID)task.task_id);
        if (finder != gpu_variants.end() && 
            (local_kind == Processor::TOC_PROC)) {
          output.chosen_variant = finder->second; 
#ifdef LOCAL_MAP_TASKS
          output.target_procs.push_back(task.target_proc);
          target_mem = get_associated_framebuffer(task.target_proc);
          reduction_mem = get_associated_zerocopy(task.target_proc);
          vdelt_mem = get_associated_zerocopy(task.target_proc);
#else
          output.target_procs.push_back(local_proc);
          target_mem = local_framebuffer;
          reduction_mem = local_zerocopy;
          vdelt_mem = local_zerocopy;
#endif
        } else {
          output.chosen_variant = cpu_variants[(SnapTaskID)task.task_id];
#ifdef LOCAL_MAP_TASKS
          get_associated_procs(task.target_proc, output.target_procs);
          target_mem = get_associated_sysmem(task.target_proc);
          reduction_mem = get_associated_sysmem(task.target_proc);
          vdelt_mem = get_associated_sysmem(task.target_proc);
#else
          output.target_procs = local_cpus;
          target_mem = local_sysmem;
          reduction_mem = local_sysmem;
          vdelt_mem = local_sysmem;
#endif
        }
        // qtot is normal
        map_snap_array(ctx, task.regions[0].region, target_mem,
                       output.chosen_instances[0]);
#ifndef SNAP_USE_RELAXED_COHERENCE
        // have to make reductions for flux, use default mapper implementation
        std::set<FieldID> dummy_fields;
        TaskLayoutConstraintSet dummy_constraints;
        default_create_custom_instances(ctx, task.target_proc,
            reduction_mem, task.regions[1], 1/*index*/, dummy_fields,
            dummy_constraints, false/*need check*/, output.chosen_instances[1]);
#else
        map_snap_array(ctx, task.regions[1].region, target_mem,
                       output.chosen_instances[1]);
#endif
        if (task.regions[2].privilege != NO_ACCESS) {
          // qim is normal
          map_snap_array(ctx, task.regions[2].region, target_mem,
                         output.chosen_instances[2]);
#ifndef SNAP_USE_RELAXED_COHERENCE
          // Need reductions for fluxm
          default_create_custom_instances(ctx, task.target_proc,
            reduction_mem, task.regions[3], 3/*index*/, dummy_fields,
            dummy_constraints, false/*need check*/, output.chosen_instances[3]); 
#else
          map_snap_array(ctx, task.regions[3].region, target_mem,
                         output.chosen_instances[3]);
#endif
        }
        // Remaining arrays that are not vdelt are normal
        const unsigned last_idx = task.regions.size() - 1;
        for (unsigned idx = 4; idx < last_idx; idx++) {
          map_snap_array(ctx, task.regions[idx].region, target_mem, 
                         output.chosen_instances[idx]);
        }
        // Put vdelt in a special memory since it is read locally
        map_snap_array(ctx, task.regions[last_idx].region, vdelt_mem,
                       output.chosen_instances[last_idx]);
        break;
      }
    default:
      {
        DefaultMapper::map_task(ctx, task, input, output);
        return;
      }
  }
  runtime->acquire_instances(ctx, output.chosen_instances);
}

//------------------------------------------------------------------------------
void Snap::SnapMapper::update_variants(const MapperContext ctx)
//------------------------------------------------------------------------------
{
  // Find our CPU and GPU variants
  for (unsigned idx = 0; idx < LAST_TASK_ID; idx++) {
    SnapTaskID tid = (SnapTaskID)(SNAP_TOP_LEVEL_TASK_ID + idx);
    std::vector<VariantID> variants;
    runtime->find_valid_variants(ctx, tid, variants, Processor::LOC_PROC);
    if (!variants.empty())
      cpu_variants[tid] = variants[0];
    variants.clear();
    runtime->find_valid_variants(ctx, tid, variants, Processor::TOC_PROC);
    if (!variants.empty())
      gpu_variants[tid] = variants[0];
  }
}

//------------------------------------------------------------------------------
void Snap::SnapMapper::map_snap_array(const MapperContext ctx, 
  LogicalRegion region, Memory target, std::vector<PhysicalInstance> &instances)
//------------------------------------------------------------------------------
{
  const std::pair<LogicalRegion,Memory> key(region, target);
  std::map<std::pair<LogicalRegion,Memory>,PhysicalInstance>::const_iterator
    finder = local_instances.find(key);
  if (finder != local_instances.end()) {
    instances.push_back(finder->second);
    return;
  }
  // First time through, then we make an instance
  std::vector<LogicalRegion> regions(1, region);  
  LayoutConstraintSet layout_constraints;
  // No specialization
  layout_constraints.add_constraint(SpecializedConstraint());
  // SOA-Fortran dimension ordering
  std::vector<DimensionKind> dimension_ordering(4);
  dimension_ordering[0] = DIM_X;
  dimension_ordering[1] = DIM_Y;
  dimension_ordering[2] = DIM_Z;
  dimension_ordering[3] = DIM_F;
  layout_constraints.add_constraint(OrderingConstraint(dimension_ordering, 
                                                       false/*contiguous*/));
  // Constrained for the target memory kind
  layout_constraints.add_constraint(MemoryConstraint(target.kind()));
  // Have all the field for the instance available
  std::vector<FieldID> all_fields;
  runtime->get_field_space_fields(ctx, region.get_field_space(), all_fields);
  layout_constraints.add_constraint(FieldConstraint(all_fields, false/*contiguous*/,
                                                    false/*inorder*/));

  PhysicalInstance result; bool created;
  if (!runtime->find_or_create_physical_instance(ctx, target, layout_constraints,
        regions, result, created, true/*acquire*/, GC_NEVER_PRIORITY)) {
    log_snap.error("ERROR: SNAP mapper failed to allocate instance");
    assert(false);
  }
  instances.push_back(result);
  // Save the result for future use
  local_instances[key] = result;
}

#ifdef LOCAL_MAP_TASKS
//------------------------------------------------------------------------------
Memory Snap::SnapMapper::get_associated_sysmem(Processor proc)
//------------------------------------------------------------------------------
{
  std::map<Processor,Memory>::const_iterator finder = 
    associated_sysmems.find(proc);
  if (finder != associated_sysmems.end())
    return finder->second;
  Machine::MemoryQuery sysmem_query(machine);
  sysmem_query.same_address_space_as(proc);
  sysmem_query.only_kind(Memory::SYSTEM_MEM);
  Memory result = sysmem_query.first();
  assert(result.exists());
  associated_sysmems[proc] = result;
  return result;
}

//------------------------------------------------------------------------------
Memory Snap::SnapMapper::get_associated_framebuffer(Processor proc)
//------------------------------------------------------------------------------
{
  std::map<Processor,Memory>::const_iterator finder = 
    associated_framebuffers.find(proc);
  if (finder != associated_framebuffers.end())
    return finder->second;
  Machine::MemoryQuery fbmem_query(machine);
  fbmem_query.same_address_space_as(proc);
  fbmem_query.only_kind(Memory::GPU_FB_MEM);
  Memory result = fbmem_query.first();
  assert(result.exists());
  associated_framebuffers[proc] = result;
  return result;
}

//------------------------------------------------------------------------------
Memory Snap::SnapMapper::get_associated_zerocopy(Processor proc)
//------------------------------------------------------------------------------
{
  std::map<Processor,Memory>::const_iterator finder = 
    associated_zerocopy.find(proc);
  if (finder != associated_zerocopy.end())
    return finder->second;
  Machine::MemoryQuery zcmem_query(machine);
  zcmem_query.same_address_space_as(proc);
  zcmem_query.only_kind(Memory::Z_COPY_MEM);
  Memory result = zcmem_query.first();
  assert(result.exists());
  associated_zerocopy[proc] = result;
  return result;
}

//------------------------------------------------------------------------------
void Snap::SnapMapper::get_associated_procs(Processor proc,
                                            std::vector<Processor> &procs)
//------------------------------------------------------------------------------
{
  std::map<Processor,std::vector<Processor> >::const_iterator finder = 
    associated_procs.find(proc);
  if (finder != associated_procs.end()) {
    procs = finder->second;
    return;
  }
  Machine::ProcessorQuery proc_query(machine);
  proc_query.same_address_space_as(proc);
  proc_query.only_kind(proc.kind());
  for (Machine::ProcessorQuery::iterator it = proc_query.begin();
        it != proc_query.end(); it++)
    procs.push_back(*it);
  associated_procs[proc] = procs;
}
#endif

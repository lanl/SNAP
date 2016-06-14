/*
 * SweepMapper.cc
 *
 *  Created on: Aug 10, 2015
 *      Author: payne
 */

#include "SweepMapper.h"
#include <LegionHelper.h>
using namespace Dragon;

void SweepMapper::slice_domain(const Task *task, const Domain &domain,
						  std::vector<DomainSplit> &slices)
{
    assert(domain.get_dim() == 1);
    Arrays::Rect<1> rect = domain.get_rect<1>();
    unsigned num_elmts = rect.volume();
    printf("ng = %i\n",ng);
    assert(num_elmts == ng);



    if (map_to_gpus && (task->task_id != 0) && (task->target_proc.kind() == TOC_PROC))
     {
    	SweepMapper::decompose_index_space(domain, all_gpus_v, 1/*splitting factor*/, slices,dogrp_inner);
     }
     else
     {
    	 SweepMapper::decompose_index_space(domain, all_cpus_v, 1/*splitting factor*/, slices,dogrp_inner);
     }

}

// Break an IndexSpace of tasks into IndexSplits
void SweepMapper::decompose_index_space(const Domain &domain,
						const std::vector<Processor> &targets,
						unsigned splitting_factor,
						std::vector<Mapper::DomainSplit> &slices,
						bool* _dogrp)
{
  	static unsigned i_proc = 0;

      // Only handle these two cases right now
      assert((domain.get_dim() == 0) || (domain.get_dim() == 1));
      if (domain.get_dim() == 0)
      {
        assert(false);
      }
      else
      {
        // Only works for one dimensional rectangles right now
        assert(domain.get_dim() == 1);
        Arrays::Rect<1> rect = domain.get_rect<1>();
        unsigned num_elmts = rect.volume();
        unsigned num_chunks = targets.size()*splitting_factor;
        if (num_chunks > num_elmts)
          num_chunks = num_elmts;
        // Number of elements per chunk rounded up
        // which works because we know that rectangles are contiguous
        unsigned lower_bound = num_elmts/num_chunks;
        unsigned upper_bound = lower_bound+1;
        unsigned number_small = num_chunks - (num_elmts % num_chunks);
        unsigned index = 0;
        unsigned n_left = 0;
        for (unsigned idx = 0; idx < num_chunks; idx++)
        {
          unsigned elmts = (idx < number_small) ? lower_bound : upper_bound;
          Arrays::Point<1> lo(index);
          Arrays::Point<1> hi(index+elmts-1);
          index += elmts;
          Arrays::Rect<1> chunk(rect.lo+lo,rect.lo+hi);

          if(_dogrp[idx])
          {

          i_proc++;

          i_proc = i_proc%targets.size();
          unsigned proc_idx = i_proc;
//          printf("mapping chunk %i to proc %u\n",idx,proc_idx);
          slices.push_back(DomainSplit(
                Domain::from_rect<1>(chunk), targets[proc_idx], false, false));
          n_left++;
          }
          else
              slices.push_back(DomainSplit(
                    Domain::from_rect<1>(chunk), targets[0], false, false));

        }

        printf("n_left = %i\n",n_left);
        assert(n_left > 0);
      }
}

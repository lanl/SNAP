/*
 * SweepMapper.h
 *
 *  Created on: Aug 10, 2015
 *      Author: payne
 */

#ifndef SWEEPMAPPER_H_
#define SWEEPMAPPER_H_

#include <BetterMapper.h>
#include <LegionMatrix.h>
#include <LRWrapper.h>


using namespace LegionRuntime;
using namespace HighLevel;
using namespace Dragon;

class SweepMapper : public BetterMapper
{
public:

	bool* dogrp_inner;
	int ng;

	SweepMapper(Machine machine, HighLevelRuntime *rt, Processor local)
	: BetterMapper(machine,rt,local){}

	void update_stuff(bool* _dogrp_inner,int _ng)
	{
		dogrp_inner = _dogrp_inner;
		ng = _ng;
	}


	virtual void slice_domain(const Task *task, const Domain &domain,
                              std::vector<DomainSplit> &slices);

    // Break an IndexSpace of tasks into IndexSplits
    static void decompose_index_space(const Domain &domain,
                            const std::vector<Processor> &targets,
                            unsigned splitting_factor,
                            std::vector<Mapper::DomainSplit> &slice,
                            bool* dogrp);
};

#endif /* SWEEPMAPPER_H_ */

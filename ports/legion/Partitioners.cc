/*
 * GlobalParticleList.cpp
 *
 *  Created on: Oct 9, 2014
 *      Author: payne
 */


#include "Partitioners.h"
#include <unistd.h>



int nTotal_from_region(Context ctx, HighLevelRuntime *runtime,LogicalRegion lr_in)
{
	Domain dom = runtime->get_index_space_domain(ctx,lr_in.get_index_space());
	return dom.get_volume();
//	return lr_in.get_index_space().get_valid_mask().get_num_elmts();

}

void PVecItem::finalize(Context ctx,HighLevelRuntime* rt)
{
	for(Domain::DomainPointIterator p(lDom);p;p++)
	{
		points.push_back(p.p);

		subregions[p.p.point_data[0]] = rt->get_logical_subregion_by_color(ctx,lPart,p.p.point_data[0]);
	}
}

SimpleSubPart::SimpleSubPart(Context ctx, HighLevelRuntime *runtime,LogicalRegion lr_in,int _nSub) : PVecItem(2), partID(2)
{
	nTotal = nTotal_from_region(ctx,runtime,lr_in);
	nSub = _nSub;
	printf("Ntotal = %i\n",nTotal);

//	DomainColoring coloring;
//
//	const int lower_bound = nTotal/nSub;
//	const int upper_bound = lower_bound+1;
//	const int number_small = nSub - (nTotal % nSub);
//	int index = 0;
//	for (int i = 0; i< nSub; i++)
//	{
//		int num_elmts = i < number_small ? lower_bound : upper_bound;
//		assert((index+num_elmts) <= nTotal);
//		Rect<1> subrect(Point<1>(index),Point<1>(index+num_elmts-1));
//		subgridBnds.push_back(subrect);
//		Domain temp = Domain::from_rect<1>(subrect);
////		temp.get_index_space(true);
//		coloring[i] = temp;		index += num_elmts;
//	}

	Coloring coloring;


//    IndexIterator itr(lr_in.get_index_space());

	const int lower_bound = nTotal/nSub;
	const int upper_bound = lower_bound+1;
	const int number_small = nSub - (nTotal % nSub);
	int index = 0;
	for (int n = 0; n< nSub; n++)
	{
		int num_elmts = n < number_small ? lower_bound : upper_bound;
		assert((index+num_elmts) <= nTotal);

		coloring[n].ranges.insert(std::pair<ptr_t,ptr_t>(index,index+num_elmts-1));
		index+=num_elmts;
//		for(int i=0;i<num_elmts;i++)
//		{
//	        assert(itr.has_next());
//			coloring[n].points.insert(itr.next());
//			index++;
//		}
	}

	Rect<1> elem_rect(Point<1>(0),Point<1>(nSub-1));
	lDom = Domain::from_rect<1>(elem_rect);
//	lDom.get_index_space(true);

//	IndexPartition iPart = runtime->create_index_partition(ctx,lr_in.get_index_space(),lDom,coloring,true);
	IndexPartition iPart = runtime->create_index_partition(ctx,lr_in.get_index_space(),coloring,true);


	parent_lr = lr_in;

	lPart = runtime->get_logical_partition(ctx,lr_in,iPart);

}

SinglePart::SinglePart(Context ctx,HighLevelRuntime *runtime,LogicalRegion lr_in) : PVecItem(1),  partID(2)
{
//	int nTotal = nTotal_from_region(ctx,runtime,lr_in);
//	int nSub = 1;
//	DomainColoring coloring;
//
//	const int lower_bound = nTotal/nSub;
//	const int upper_bound = lower_bound+1;
//	const int number_small = nSub - (nTotal % nSub);
//	int index = 0;
//	for (int i = 0; i< nSub; i++)
//	{
//		int num_elmts = i < number_small ? lower_bound : upper_bound;
//		assert((index+num_elmts) <= nTotal);
//		Rect<1> subrect(Point<1>(index),Point<1>(index+num_elmts-1));
//		subgridBnds.push_back(subrect);
//		Domain temp = Domain::from_rect<1>(subrect);
////		temp.get_index_space(true);
//		coloring[i] = temp;
//		index += num_elmts;
//	}
//
//	Rect<1> elem_rect(Point<1>(0),Point<1>(nSub-1));
//	lDom = Domain::from_rect<1>(elem_rect);
//	lDom.get_index_space(true);
//
//
//	IndexPartition iPart = runtime->create_index_partition(ctx,lr_in.get_index_space(),lDom,coloring,true);
//
//	lPart = runtime->get_logical_partition(ctx,lr_in,iPart);
//
//
	int nTotal = nTotal_from_region(ctx,runtime,lr_in);
	int nSub = 1;

//	DomainColoring coloring;
//
//	const int lower_bound = nTotal/nSub;
//	const int upper_bound = lower_bound+1;
//	const int number_small = nSub - (nTotal % nSub);
//	int index = 0;
//	for (int i = 0; i< nSub; i++)
//	{
//		int num_elmts = i < number_small ? lower_bound : upper_bound;
//		assert((index+num_elmts) <= nTotal);
//		Rect<1> subrect(Point<1>(index),Point<1>(index+num_elmts-1));
//		subgridBnds.push_back(subrect);
//		Domain temp = Domain::from_rect<1>(subrect);
////		temp.get_index_space(true);
//		coloring[i] = temp;		index += num_elmts;
//	}

	Coloring coloring;


//    IndexIterator itr(lr_in.get_index_space());

	const int lower_bound = nTotal/nSub;
	const int upper_bound = lower_bound+1;
	const int number_small = nSub - (nTotal % nSub);
	int index = 0;
	for (int n = 0; n< nSub; n++)
	{
		int num_elmts = n < number_small ? lower_bound : upper_bound;
		assert((index+num_elmts) <= nTotal);

		coloring[n].ranges.insert(std::pair<ptr_t,ptr_t>(index,index+num_elmts-1));
		index+=num_elmts;

//		for(int i=0;i<num_elmts;i++)
//		{
//	        assert(itr.has_next());
//			coloring[n].points.insert(itr.next());
//			index++;
//		}
	}

	Rect<1> elem_rect(Point<1>(0),Point<1>(nSub-1));
	lDom = Domain::from_rect<1>(elem_rect);
//	lDom.get_index_space(true);

//	IndexPartition iPart = runtime->create_index_partition(ctx,lr_in.get_index_space(),lDom,coloring,true);
	IndexPartition iPart = runtime->create_index_partition(ctx,lr_in.get_index_space(),coloring,true);


	parent_lr = lr_in;

	lPart = runtime->get_logical_partition(ctx,lr_in,iPart);
}


CopiedPart::CopiedPart(Context ctx, HighLevelRuntime *runtime,LogicalRegion lr_in,int nSub) : PVecItem(3), partID(3)
{
	int nTotal = nTotal_from_region(ctx,runtime,lr_in);

//	DomainColoring coloring;
//
//	const int lower_bound = nTotal/nSub;
//	const int upper_bound = lower_bound+1;
//	const int number_small = nSub - (nTotal % nSub);
//	int index = 0;
//	for (int i = 0; i< nSub; i++)
//	{
//		int num_elmts = i < number_small ? lower_bound : upper_bound;
//		assert((index+num_elmts) <= nTotal);
//		Rect<1> subrect(Point<1>(index),Point<1>(index+num_elmts-1));
//		subgridBnds.push_back(subrect);
//		Domain temp = Domain::from_rect<1>(subrect);
////		temp.get_index_space(true);
//		coloring[i] = temp;		index += num_elmts;
//	}

	Coloring coloring;

	for (int n = 0; n< nSub; n++)
	{
			coloring[n].ranges.insert(std::pair<ptr_t,ptr_t>(0,nTotal-1));
	}

	Rect<1> elem_rect(Point<1>(0),Point<1>(nSub-1));
	lDom = Domain::from_rect<1>(elem_rect);
//	lDom.get_index_space(true);

//	IndexPartition iPart = runtime->create_index_partition(ctx,lr_in.get_index_space(),lDom,coloring,true);
	IndexPartition iPart = runtime->create_index_partition(ctx,lr_in.get_index_space(),coloring,true);


	parent_lr = lr_in;

	lPart = runtime->get_logical_partition(ctx,lr_in,iPart);

}

ColoredPart::ColoredPart(Context ctx, HighLevelRuntime *runtime,LogicalRegion lr_in,Coloring _coloring) : partID(4)
{
	IndexPartition iPart = runtime->create_index_partition(ctx,lr_in.get_index_space(),_coloring,true);

	Rect<1> elem_rect(Point<1>(0),Point<1>((*_coloring.rbegin()).first));
	lDom = runtime->get_index_partition_color_space(ctx,iPart);
	parent_lr = lr_in;
	lPart = runtime->get_logical_partition(ctx,parent_lr,iPart);
}

SetVals::SetVals(Context ctx, HighLevelRuntime *runtime,
		LogicalRegion data,int _val, std::vector<FieldID> fids)
 : TaskLauncher(SetVals::TASK_ID(),TaskArgument(this,sizeof(SetVals)))
{

	val = _val;
	LogicalRegion lr_parent;
	if(runtime->has_parent_logical_partition(ctx,data))
		lr_parent = runtime->get_parent_logical_region(ctx,runtime->get_parent_logical_partition(ctx,data));
	else
		lr_parent = data;
	RegionRequirement rr_in(data,READ_WRITE,EXCLUSIVE,lr_parent);

	rr_in.add_fields(fids);

	add_region_requirement(rr_in);



}


int SetVals::cpu_base_impl(const Task *task,Context ctx,
            const std::vector<PhysicalRegion> &regions, HighLevelRuntime *runtime)
{
	Domain dom = runtime->get_index_space_domain(ctx,
	  task->regions[0].region.get_index_space());
	Rect<1> rect = dom.get_rect<1>();
	std::set<FieldID>::iterator dst_fid;
	for(dst_fid=task->regions[0].privilege_fields.begin();
			dst_fid!=task->regions[0].privilege_fields.end();dst_fid++)
	{
		RegionAccessor<AccessorType::Generic, float> dst_acc =
			regions[0].get_field_accessor(*dst_fid).typeify<float>();
		for(GenericPointInRectIterator<1> pir(rect); pir; pir++)
		{
			float cur_val = dst_acc.read(DomainPoint::from_point<1>(pir.p));
			printf("Array_val[%i] = %f\n",pir.p[0],cur_val);
//			dst_acc.write(DomainPoint::from_point<1>(pir.p),val);
		}


	}
	return 0;
}





































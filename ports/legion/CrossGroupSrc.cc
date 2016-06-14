/*
 * CrossGroupSrc.cc
 *
 *  Created on: Jun 16, 2015
 *      Author: payne
 */

#include "CrossGroupSrc.h"
#include "InGroupSource.h"
#include "Sweep.h"
#include <math.h>
#include <LegionHelper.h>
#include <MustEpochKernelLauncher.h>
#include "SweepMapper.h"

#include <utmpx.h>


SingleKernelArgs CrossGroupSrc::genArgs(Context ctx, HighLevelRuntime* runtime,
               LRWrapper _lw_allArrays,
			   LRWrapper _lw_chunkArrays,
			   LRWrapper _lw_slgg,
			   LRWrapper _lw_diag,
			   LRWrapper _lw_lines)
{
	using namespace FIDs;
	lw_allArrays = _lw_allArrays;

	SingleKernelArgs args;
	args.add_arg(_lw_allArrays,0,READ_ONLY,EXCLUSIVE);
	args.add_arg(_lw_allArrays,1,READ_ONLY,EXCLUSIVE);
	args.add_arg(_lw_allArrays,2,READ_ONLY,EXCLUSIVE);
	args.add_arg(_lw_allArrays,3,READ_ONLY,EXCLUSIVE);


	// We need to add region requirements for all of the LRWrappers, so generate those too
	std::vector<FieldPrivlages> allArray_fids;
	allArray_fids.push_back(FieldPrivlages(0,READ_ONLY));
	allArray_fids.push_back(FieldPrivlages(1,READ_ONLY));
	allArray_fids.push_back(FieldPrivlages(2,READ_ONLY));
	allArray_fids.push_back(FieldPrivlages(3,READ_ONLY));

	std::vector<FieldPrivlages> chunkArrays_fids;
	chunkArrays_fids.push_back(FieldPrivlages(Array::angFluxes,READ_WRITE));
	chunkArrays_fids.push_back(FieldPrivlages(Array::scalFluxes,READ_WRITE));
	chunkArrays_fids.push_back(FieldPrivlages(Array::savedFlux,READ_WRITE));
	chunkArrays_fids.push_back(FieldPrivlages(Array::sigmaT,READ_ONLY));
	chunkArrays_fids.push_back(FieldPrivlages(Array::sigmaS,READ_ONLY));
	chunkArrays_fids.push_back(FieldPrivlages(Array::mat,READ_ONLY));
	chunkArrays_fids.push_back(FieldPrivlages(Array::bci,READ_WRITE));
	chunkArrays_fids.push_back(FieldPrivlages(Array::bcj,READ_WRITE));
	chunkArrays_fids.push_back(FieldPrivlages(Array::bck,READ_WRITE));

	typedef std::pair<FieldID,FieldID> FieldPair;
	std::map<FieldPair,PrivilegeMode> allArray_child;
	std::map<FieldPair,PrivilegeMode> chunkArray_child;


	allArray_child = propagate_nested_regions_privs(ctx,runtime,_lw_allArrays,allArray_fids);
	chunkArray_child = propagate_nested_regions_privs(ctx,runtime,_lw_chunkArrays,chunkArrays_fids);



	std::map<Color,std::vector<RegionRequirement>> rreqs;
	std::vector<RegionRequirement> rreqs_chunk;

	rreqs = distribute_nested_regions(ctx,runtime,_lw_allArrays,allArray_child);
	rreqs_chunk = propagate_nested_regions(ctx,runtime,_lw_chunkArrays,chunkArray_child);

	args.add_nested(rreqs[0]);
	args.add_nested(rreqs_chunk);



	return args;

}

int CrossGroupSrc::any_left(Context ctx,HighLevelRuntime* rt,LRWrapper _dogrp,LRWrapper _df)
{
//	rt->disable_profiling();

	int res = 0;

	RegionRequirement rr(_df.lr,READ_ONLY,EXCLUSIVE,_df.lr);
	rr.add_field(0);

	RegionRequirement rr2(_dogrp.lr,READ_WRITE,EXCLUSIVE,_dogrp.lr);
	rr2.add_field(0);


	PhysicalRegion pr1 = rt->map_region(ctx,rr);
	PhysicalRegion pr2 = rt->map_region(ctx,rr2);
	{
	LegionMatrix<double> df(_df,pr1,0);
	LegionMatrix<bool> dogrp(_dogrp,pr2,0);
	double tolr = -1.0;

	for(int ig=0;ig<params.ng;ig++)
		dogrp(ig) = std::min(dogrp(ig).cast(),(rand()%16 > 16));


			for(int i=0;i<params.nCx*params.nCy*params.nCz;i++)
				for(int ig=0;ig<params.ng;ig++)
				{

//						if(df(ig,i) <= tolr || dogrp(ig).cast() == 0)
//							dogrp(ig) = 0;
				}

			for(int ig=0;ig<params.ng;ig++)
				if(dogrp(ig).cast())
					res+=1;


	}
	rt->unmap_region(ctx,pr1);
	rt->unmap_region(ctx,pr2);

//	rt->enable_profiling();

	printf("There are %%%i groups left\n",res);
	return res;

}

timespec diff(timespec start, timespec end)
{
	timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0) {
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
}

__host__
void CrossGroupSrc::evaluate(const Task* task, Context ctx, HighLevelRuntime* rt,
              LegionMatrix<LRWrapper> lr_chunkArrays,
			  LegionMatrix<LRWrapper> lr_slgg,
			  LegionMatrix<LRWrapper> lr_diag, LegionMatrix<LRWrapper> lr_lines)
{
	rt->disable_profiling();

	srand(5928347);
	LRWrapper dogrp_o;
	dogrp_o.create(ctx,rt,"dogrp_o",{params.ng},0,bool());

	LRWrapper dogrp_i;
	dogrp_i.create(ctx,rt,"dogrp_i",{params.ng},0,bool());

	LRWrapper df_o;
	df_o.create(ctx,rt,"df_o",{params.ng,params.nCx*params.nCy*params.nCz},0,double());
	LRWrapper df_i;
	df_i.create(ctx,rt,"df_i",{params.ng,params.nCx*params.nCy*params.nCz},0,double());

	LegionMatrix<LRWrapper> lr_scalFlux(ctx,lr_chunkArrays(0),FIDs::Array::scalFluxes);
	LegionMatrix<LRWrapper> lr_angFlux(ctx,lr_chunkArrays(0),FIDs::Array::angFluxes);
	LegionMatrix<LRWrapper> lr_sigmaT(ctx,lr_chunkArrays(0),FIDs::Array::sigmaT);
	LegionMatrix<LRWrapper> lr_bci(ctx,lr_chunkArrays(0),FIDs::Array::bci);
	LegionMatrix<LRWrapper> lr_bcj(ctx,lr_chunkArrays(0),FIDs::Array::bcj);
	LegionMatrix<LRWrapper> lr_bck(ctx,lr_chunkArrays(0),FIDs::Array::bck);



	LegionHelper helper(ctx,rt);
//	std::set<Processor> local_procs =  dynamic_cast<BetterMapper*>(rt->get_mapper(ctx,1))->all_cpus;
	bool dgrp_tmp[params.ng];
	bool dgrp_tmp_inner[params.ng];

//	for (std::set<Processor>::const_iterator it = local_procs.begin();
//		it != local_procs.end(); it++)
//	{
//	  dynamic_cast<SweepMapper*>(rt->get_mapper(ctx,1,*it))->update_stuff(dgrp_tmp_inner,params.ng);
//	}





	{
	memset(dgrp_tmp,1,params.ng*sizeof(bool));
	memset(dgrp_tmp_inner,1,params.ng*sizeof(bool));




	helper.set(dogrp_o.lr,0,dgrp_tmp,params.ng);
	helper.set(dogrp_i.lr,0,dgrp_tmp,params.ng);


	}

	OuterSrcScat scat_o(params);
	auto src_scat_op = genMustEpochKernel(scat_o,ctx,rt,lr_chunkArrays(0).cast(),lr_slgg(0).cast(),dogrp_o);

	SaveFluxes save_outer(params);
	auto save_fluxes_o = genMustEpochKernel(save_outer,ctx,rt,lr_chunkArrays(0).cast(),dogrp_o,FIDs::SavedFlux::fluxpo);
	auto save_fluxes_i = genMustEpochKernel(save_outer,ctx,rt,lr_chunkArrays(0).cast(),dogrp_i,FIDs::SavedFlux::fluxpi);

	OuterConv check_conv(params);
	auto check_conv_o = genMustEpochKernel(check_conv,ctx,rt,lr_chunkArrays(0).cast(),dogrp_o,df_o,FIDs::SavedFlux::fluxpo);
	auto check_conv_i = genMustEpochKernel(check_conv,ctx,rt,lr_chunkArrays(0).cast(),dogrp_i,df_i,FIDs::SavedFlux::fluxpi);


	InnerSrcScat scat_i(params);
	auto inner_src_scat_op = genMustEpochKernel(scat_i,ctx,rt,lr_chunkArrays(0).cast(),dogrp_i);

	// Set up the partitions for the lines and diags
	LPWrapper lp_diag;
	lp_diag.singlePart(ctx,rt,lr_diag(0).cast());
	LPWrapper lp_lines;
	lp_lines.singlePart(ctx,rt,lr_lines(0).cast());


	std::vector<MEKLInterface*> sweep_ops;

	sweep_ops = genSweeps(ctx,rt,params,lr_scalFlux,lr_sigmaT,lr_angFlux,lr_bci,lr_bcj,lr_bck,dogrp_i,lp_diag,lp_lines);
//	rt->unmap_all_regions(ctx);

	rt->enable_profiling();

	int nprocs = params.nCx*params.nCy*params.nCz;
	timespec time1, time2;


	int n_left = params.ng;
	int i_iter = 0;
	while(n_left > 0)
	{
		printf("There are %%%i groups left to converge\n",n_left);



	//	LegionMatrix<LRWrapper> angFluxes(targs,lr_chunkArrays(0).cast(),FIDs::Array::angFluxes);
	//	LegionMatrix<LRWrapper> scalFluxes(targs,lr_chunkArrays(0).cast(),FIDs::Array::scalFluxes);
	//	LegionMatrix<LRWrapper> savedFluxes(targs,lr_chunkArrays(0).cast(),FIDs::Array::savedFlux);
	//	LegionMatrix<LRWrapper> sigmaT(targs,lr_chunkArrays(0).cast(),FIDs::Array::sigmaT);
	//	LegionMatrix<LRWrapper> sigmaS(targs,lr_chunkArrays(0).cast(),FIDs::Array::sigmaS);
	//	LegionMatrix<LRWrapper> mat(targs,lr_chunkArrays(0).cast(),FIDs::Array::mat);
	//	LegionMatrix<LRWrapper> bci(targs,lr_chunkArrays(0).cast(),FIDs::Array::bci);
	//	LegionMatrix<LRWrapper> bcj(targs,lr_chunkArrays(0).cast(),FIDs::Array::bcj);
	//	LegionMatrix<LRWrapper> bck(targs,lr_chunkArrays(0).cast(),FIDs::Array::bck);
	//
	//	// Calculate the cross - group source terms
	//	for(int k=0;k<params.nCz;k++)
	//		for(int j=0;j<params.nCy;j++)
	//			for(int i=0;i<params.nCx;i++)
	//			{
	//				OuterSrcScat scat(params);
	//				auto src_scat_op = genFieldKernel(scat,ctx,rt,lr_slgg(0).cast(),
	//				                                  scalFluxes(i,j,k).cast(),
	//				                                  sigmaT(i,j,k).cast(),
	//				                                  mat(i,j,k).cast(),
	//				                                  dogrp,i+params.nCx*(j+params.nCy*k));
	//
	//				src_scat_op.execute(ctx,rt);
	//
	//			}



		src_scat_op.execute(ctx,rt);
		printf("finished sending off the src sct op\n");



		// Zero out the inner iterations and set inner_done to false

		// Save the flux from the previous iteration
		printf("executing save fluxes\n");
		save_fluxes_o.execute(ctx,rt);


		// Converge the inner - group sources
		helper.copy<bool>(dogrp_i.lr,0,dogrp_o.lr,0,params.ng);
		int ninner_left = n_left;
		int iter_inner = 0;

		while(ninner_left > 0)
		{
//			helper.set(dgrp_tmp_inner,dogrp_i.lr,0,params.ng);
		// Compute the inner - group source
			inner_src_scat_op.execute(ctx,rt);

		// Save the previous iteration flux
			save_fluxes_i.execute(ctx,rt);

		clock_gettime(CLOCK_REALTIME, &time1);
			FutureMap fm ;
		// Sweep
		int isweep = 0;
		for(auto sweep : sweep_ops)
		{

			printf("Executing sweep %i\n",isweep);
			fm = rt->execute_must_epoch(ctx,*sweep);

//			if(((isweep+1)%nprocs) == 0)
//				fm.wait_all_results();
//			sweep->execute_tasks(ctx,rt,dogrp_i);

			isweep++;
		}
//		fm.wait_all_results();

		clock_gettime(CLOCK_REALTIME, &time2);

			// Check for Convergence
			check_conv_i.execute(ctx,rt);

			ninner_left = any_left(ctx,rt,dogrp_i,df_i);
			iter_inner++;

		}


		// Check the cross group convergence
		printf("executing convergence check\n");
		check_conv_o.execute(ctx,rt);




		n_left = any_left(ctx,rt,dogrp_o,df_o);
		i_iter++;

	}


	std::cout<<diff(time1,time2).tv_sec+diff(time1,time2).tv_nsec/1.0e9<<std::endl;


}

void CrossGroupSrc::register_cpu()
{
	SingleKernelLancher<CrossGroupSrc>::register_cpu();
	MustEpochKernelLauncher<OuterSrcScat>::register_cpu();
	MustEpochKernelLauncher<OuterConv>::register_cpu();
	MustEpochKernelLauncher<SaveFluxes>::register_cpu();

}



EpochKernelArgs OuterSrcScat::genArgs(Context ctx, HighLevelRuntime* runtime,
               LRWrapper _lw_chunkArrays,
               LRWrapper _lw_slgg,
               LRWrapper _lw_dogrp)
{
	using namespace FIDs;
	EpochKernelArgs args;
	printf("Generating Args for OuterSrcScat\n");
	lw_slgg = _lw_slgg;


	LPWrapper* lp_chunkArrays = new LPWrapper();
	LPWrapper* lp_chunkArrays2 = new LPWrapper();

	LPWrapper* lp_dogrp = new LPWrapper();
	LPWrapper* lp_slgg = new LPWrapper();

	lp_chunkArrays->simpleSubPart(ctx,runtime,_lw_chunkArrays,params.nCx*params.nCy*params.nCz);

	lp_dogrp->singlePart(ctx,runtime,_lw_dogrp);
	lp_slgg->singlePart(ctx,runtime,_lw_slgg);

	using namespace FIDs;

	args.add_arg(*lp_dogrp,0,READ_ONLY,EXCLUSIVE);
	args.add_arg(*lp_chunkArrays,Array::sigmaT,READ_ONLY,EXCLUSIVE);
	args.add_arg(*lp_chunkArrays,Array::scalFluxes,READ_ONLY,EXCLUSIVE);

	args.add_arg(*lp_slgg,0,READ_ONLY,EXCLUSIVE);


	args.add_arg(*lp_chunkArrays,Array::mat,READ_ONLY,EXCLUSIVE);





	typedef std::pair<FieldID,FieldID> FieldPair;
	std::map<FieldPair,PrivilegeMode> chunkArray_child;

	chunkArray_child[FieldPair(Array::scalFluxes,ScalFlux::q2grp)] = READ_WRITE;
	chunkArray_child[FieldPair(Array::scalFluxes,ScalFlux::flux)] = READ_ONLY;
	chunkArray_child[FieldPair(Array::sigmaT,SigmaT::qi)] = READ_ONLY;
	chunkArray_child[FieldPair(Array::mat,Mat::mat)] = READ_ONLY;



	std::map<Color,std::vector<RegionRequirement>> rreqs;

	rreqs = distribute_nested_regions(ctx,runtime,_lw_chunkArrays,chunkArray_child);

	args.nested_reqs = rreqs;



	return args;

}

__host__
void OuterSrcScat::evaluate(const Task* task, Context ctx, HighLevelRuntime* rt,
					LegionMatrix<bool> _dgrp,
					LegionMatrix<LRWrapper> _qi,
					LegionMatrix<LRWrapper> _scalFlux,
					LegionMatrix<double> slgg,
					LegionMatrix<LRWrapper> _mat)
{
	int iDomain = task->index_point.point_data[0];

	for(int i_grp=0;i_grp<params.ng;i_grp++)
	if(_dgrp(i_grp))
	{
//		  printf("CPU: %d\n", sched_getcpu());

//	printf("In OuterSrcScat dom %i grp %i\n",iDomain,i_grp);
	typedef LegionMatrix<double> lMatrix_d;
	typedef LegionMatrix<int> lMatrix_i;

	lMatrix_d qi(ctx,_qi(iDomain).cast(),FIDs::SigmaT::qi);
	lMatrix_i mat(ctx,_mat(iDomain).cast(),FIDs::Mat::mat);
	lMatrix_d flux(ctx,_scalFlux(iDomain).cast(),FIDs::ScalFlux::flux);
	lMatrix_d q2grp(ctx,_scalFlux(iDomain).cast(),FIDs::ScalFlux::q2grp);
	int nx_l(params.nx_l),ny_l(params.ny_l),nz_l(params.nz_l);
	int cmom = params.cmom;
	int nmom = params.nmom;

	for(int k=0;k<nz_l;k++)
		for(int j=0;j<ny_l;j++)
			for(int i=0;i<nx_l;i++)
			{
				q2grp(0,i,j,k,i_grp) = qi(i,j,k,i_grp).cast();

				for(int l=1;l<cmom;l++)
				{
					q2grp(l,i,j,k,i_grp) = 0.0;
				}
			}

	for(int grp2=0;grp2<params.ng;grp2++)
	for(int k=0;k<nz_l;k++)
		for(int j=0;j<ny_l;j++)
			for(int i=0;i<nx_l;i++)
			{
					int nt = 0;
				for(int l=0;l<nmom;l++)
				{
					for(int cm=0;cm<(2*l+1);cm++)
					{
						q2grp(nt+cm,i,j,k,i_grp) = q2grp(nt+cm,i,j,k,i_grp) + slgg(mat(i,j,k),l,grp2,i_grp).cast()*flux(nt+cm,i,j,k,grp2).cast();
					}
					nt += 2*l+1;
				}
			}
	}

}

//__host__
//void OuterSrcScat::evaluate(int i_grp,lrAccessor<bool> _dgrp,
//              lrAccessor<double> _qi,
//              lrAccessor<double> _slgg,
//              lrAccessor<int> _mat,
//              lrAccessor<double> _flux,
//              lrAccessor<double> _q2grp)
//{
//
//	int iDomain = targs.task->index_point.point_data[0];
//
//	typedef LegionMatrix<double> lMatrix_d;
//	typedef LegionMatrix<int> lMatrix_i;
//
//	lMatrix_d qi(lw_src,_qi);
//	lMatrix_d slgg(lw_slgg,_slgg);
//	lMatrix_i mat(lw_mat,_mat);
//	lMatrix_d flux(lw_scFlux,_flux);
//	lMatrix_d q2grp(lw_scFlux,_q2grp);
//	int nx_l(params.nx_l),ny_l(params.ny_l),nz_l(params.nz_l);
//	int cmom = params.cmom;
//	int nmom = params.nmom;
//
//	for(int k=0;k<nz_l;k++)
//		for(int j=0;j<ny_l;j++)
//			for(int i=0;i<nx_l;i++)
//			{
//				q2grp(0,i,j,k,i_grp) = qi(i,j,k,i_grp).cast();
//
//				for(int l=1;l<cmom;l++)
//				{
//					q2grp(l,i,j,k,i_grp) = 0.0;
//				}
//			}
//
//	for(int grp2=0;grp2<params.ng;grp2++)
//	for(int k=0;k<nz_l;k++)
//		for(int j=0;j<ny_l;j++)
//			for(int i=0;i<nx_l;i++)
//			{
//					int nt = 0;
//				for(int l=0;l<nmom;l++)
//				{
//					for(int cm=0;cm<(2*l+1);cm++)
//					{
//						q2grp(nt+cm,i,j,k,i_grp) = q2grp(nt+cm,i,j,k,i_grp) + slgg(mat(i,j,k),l,grp2,i_grp).cast()*flux(nt+cm,i,j,k,grp2).cast();
//					}
//					nt += 2*l+1;
//				}
//			}
//
//}

EpochKernelArgs OuterConv::genArgs(Context ctx, HighLevelRuntime* runtime,
                          LRWrapper chunkArrays,LRWrapper _dogrp,LRWrapper _df,
                          FieldID _save_fid)
{

	EpochKernelArgs args;
	printf("Generating args for Couter Convergence Test\n");
	lw_df = _df;
	save_fid = _save_fid;

	LPWrapper* lp_chunkArrays = new LPWrapper();
	LPWrapper* lp_df = new LPWrapper();
	LPWrapper* lp_dogrp = new LPWrapper();


	lp_chunkArrays->slicedPart(ctx,runtime,chunkArrays,"%1","%1","%1");
	lp_df->simpleSubPart(ctx,runtime,_df,params.nCx*params.nCy*params.nCz);
	lp_dogrp->singlePart(ctx,runtime,_dogrp);

	using namespace FIDs;

	// Add the evaluate function arguments
	args.add_arg(*lp_chunkArrays,Array::scalFluxes,READ_ONLY,EXCLUSIVE);
	args.add_arg(*lp_chunkArrays,Array::savedFlux,READ_ONLY,EXCLUSIVE);
	args.add_arg(*lp_dogrp,0,READ_ONLY,EXCLUSIVE);
	args.add_arg(*lp_df,0,READ_WRITE,EXCLUSIVE);



	typedef std::pair<FieldID,FieldID> FieldPair;
	std::map<FieldPair,PrivilegeMode> child_privs;
	// Propogate and distrubute requirements for nested regions
	child_privs[FieldPair(Array::scalFluxes,ScalFlux::flux)] = READ_ONLY;
	child_privs[FieldPair(Array::savedFlux,save_fid)] = READ_ONLY;


	std::map<Color,std::vector<RegionRequirement>> rreqs;
	rreqs = distribute_nested_regions(ctx,runtime,chunkArrays,child_privs);


	args.nested_reqs = rreqs;
	return args;
}


__host__
void OuterConv::evaluate(const Task* task, Context ctx, HighLevelRuntime* rt,
              LegionMatrix<LRWrapper> _flux,
              LegionMatrix<LRWrapper> _fluxpo,
              LegionMatrix<bool> _dogrp,
              LegionMatrix<double> df)
{

	int i_dom = task->index_point.point_data[0];
//	printf("Inside outer convergence check %i\n",i_dom);

	double tolr = 1.0e-2;
	typedef LegionMatrix<double> lMatrix_d;
	lMatrix_d flux(ctx,_flux(i_dom).cast(),FIDs::ScalFlux::flux);
	lMatrix_d fluxpo(ctx,_fluxpo(i_dom).cast(),save_fid);

	for(int ig=0;ig<params.ng;ig++)
		df(ig,i_dom) = 0.0;

	for(int ig=0;ig<params.ng;ig++)
		if(_dogrp(ig).cast())
		for(int k=0;k<params.nz_l;k++)
			for(int j=0;j<params.ny_l;j++)
				for(int i=0;i<params.nx_l;i++)
				{
					if(fabs(fluxpo(i,j,k,ig)) > tolr)
						df(ig,i_dom) = fmax(fabs(flux(0,i,j,k,ig).cast()/fluxpo(i,j,k,ig) - 1.0),df(ig,i_dom));
					else
						df(ig,i_dom) = fmax(fabs(flux(0,i,j,k,ig).cast() - fluxpo(i,j,k,ig)),df(ig,i_dom));

//					printf("chunk %i df(%i,%i,%i,%i) = %e\n",i_dom,i,j,k,ig,df(ig,i_dom).cast());
				}

}


EpochKernelArgs SaveFluxes::genArgs(Context ctx, HighLevelRuntime* runtime,
               LRWrapper chunkArrays,LRWrapper _dogrp,FieldID _save_fid)
{
	printf("Generating args for Saving Fluxes\n");

	EpochKernelArgs args;

	save_fid = _save_fid;

//	SimpleSubPart lp_chunkArrays(ctx,runtime,chunkArrays.lr,params.nCx*params.nCy*params.nCz);
//	CopiedPart lp_dogrp(ctx,runtime,_dogrp.lr,params.nCx*params.nCy*params.nCz);

	LPWrapper* lp_chunkArrays = new LPWrapper();
	lp_chunkArrays->simpleSubPart(ctx,runtime,chunkArrays,params.nCx*params.nCy*params.nCz);
	LPWrapper* lp_dogrp  = new LPWrapper();
	lp_dogrp->singlePart(ctx,runtime,_dogrp);

	using namespace FIDs;
	std::vector<FieldPrivlages> parent_privs;
	parent_privs.push_back(FieldPrivlages(Array::scalFluxes,READ_ONLY));
	parent_privs.push_back(FieldPrivlages(Array::savedFlux,READ_ONLY));


	args.add_arg(*lp_chunkArrays,Array::scalFluxes,READ_ONLY,EXCLUSIVE);
	args.add_arg(*lp_chunkArrays,Array::savedFlux,READ_ONLY,EXCLUSIVE);
	args.add_arg(*lp_dogrp,0,READ_ONLY,EXCLUSIVE);



	typedef std::pair<FieldID,FieldID> FieldPair;
	std::map<FieldPair,PrivilegeMode> child_privs;

	child_privs[FieldPair(Array::scalFluxes,ScalFlux::flux)] = READ_ONLY;
	child_privs[FieldPair(Array::savedFlux,save_fid)] = READ_WRITE;




	std::map<Color,std::vector<RegionRequirement>> rreqs;
	rreqs = distribute_nested_regions(ctx,runtime,chunkArrays,child_privs);
	args.nested_reqs = rreqs;
	return args;
}


__host__
void SaveFluxes::evaluate(const Task* task, Context ctx, HighLevelRuntime* rt,
              LegionMatrix<LRWrapper> _flux,
              LegionMatrix<LRWrapper> _fluxpo,
              LegionMatrix<bool> _dogrp)
{
	int i_dom = task->index_point.point_data[0];
//	printf("Inside saving fluxes %i\n",i_dom);

	LegionMatrix<double> flux(ctx,_flux(i_dom).cast(),FIDs::ScalFlux::flux);
	LegionMatrix<double> fluxpo(ctx,_fluxpo(i_dom).cast(),save_fid);
	for(int ig=0;ig<params.ng;ig++)
	for(int k=0;k<params.nz_l;k++)
		for(int j=0;j<params.ny_l;j++)
			for(int i=0;i<params.nx_l;i++)
			{
//				printf("saving flux(%i,%i,%i,%i)\n",i,j,k,ig);
				fluxpo(i,j,k,ig) = flux(0,i,j,k,ig).cast();
			}
}








































/*
 * InGroupSource.cc
 *
 *  Created on: Jun 16, 2015
 *      Author: payne
 */

#include "InGroupSource.h"
#include "SlappyParams.h"



EpochKernelArgs InnerSrcScat::genArgs(Context ctx, HighLevelRuntime* runtime,
               LRWrapper _lw_chunkArrays,
               LRWrapper _lw_dogrp)
{
	using namespace FIDs;

	printf("Generating Args for OuterSrcScat\n");

	EpochKernelArgs args;


	// Create the parent partitions
	lp_chunkArrays.slicedPart(ctx,runtime,_lw_chunkArrays,"%1","%1","%1");
	lp_dogrp.singlePart(ctx,runtime,_lw_dogrp);

	// Add arguments for parent partition region requirement generation
	args.add_arg(lp_dogrp,0,READ_ONLY,EXCLUSIVE);
	args.add_arg(lp_chunkArrays,Array::sigmaS,READ_ONLY,EXCLUSIVE);
	args.add_arg(lp_chunkArrays,Array::scalFluxes,READ_ONLY,EXCLUSIVE);


	// Propogate child regions
	typedef std::pair<FieldID,FieldID> FieldPair;
	std::map<FieldPair,PrivilegeMode> chunkArray_child;

	chunkArray_child[FieldPair(Array::scalFluxes,ScalFlux::qtot)] = READ_WRITE;
	chunkArray_child[FieldPair(Array::scalFluxes,ScalFlux::flux)] = READ_ONLY;
	chunkArray_child[FieldPair(Array::scalFluxes,ScalFlux::q2grp)] = READ_ONLY;

	chunkArray_child[FieldPair(Array::sigmaS,SigmaS::s_xs)] = READ_ONLY;

	std::map<Color,std::vector<RegionRequirement>> rreqs;
	rreqs = distribute_nested_regions(ctx,runtime,_lw_chunkArrays,chunkArray_child);

	args.nested_reqs = rreqs;


	return args;
}



__host__
void InnerSrcScat::evaluate(const Task* task, Context ctx, HighLevelRuntime* rt,
                            LegionMatrix<bool> _dgrp,
                            LegionMatrix<LRWrapper> _s_xs,
                            LegionMatrix<LRWrapper> _scalFlux)
{
	int iDomain = task->index_point.point_data[0];
	for(int i_grp=0;i_grp<params.ng;i_grp++)
	if(_dgrp(i_grp))
	{
//		printf("In OuterSrcScat dom %i grp %i\n",iDomain,i_grp);
		typedef LegionMatrix<double> lMatrix_d;
		lMatrix_d flux(ctx,_scalFlux(iDomain).cast(),FIDs::ScalFlux::flux);
		lMatrix_d q2grp(ctx,_scalFlux(iDomain).cast(),FIDs::ScalFlux::q2grp);
		lMatrix_d qtot(ctx,_scalFlux(iDomain).cast(),FIDs::ScalFlux::qtot);
		lMatrix_d s_xs(ctx,_s_xs(iDomain).cast(),FIDs::SigmaS::s_xs);

		// import the parameters
		int nx_l(params.nx_l),ny_l(params.ny_l),nz_l(params.nz_l);
		int cmom = params.cmom;
		int nmom = params.nmom;

		for(int k=0;k<nz_l;k++)
			for(int j=0;j<ny_l;j++)
				for(int i=0;i<nx_l;i++)
				{
						int nt = 0;
					for(int l=0;l<nmom;l++)
					{
						for(int cm=0;cm<(2*l+1);cm++)
						{
							qtot(nt+cm,i,j,k,i_grp) = q2grp(nt+cm,i,j,k,i_grp)
													+ flux(nt+cm,i,j,k,i_grp).cast()*s_xs(l,i,j,k,i_grp).cast();
						}
						nt += 2*l+1;
					}
				}
		}


}


void InnerSrcScat::register_cpu()
{
	MustEpochKernelLauncher<InnerSrcScat>::register_cpu();
}


































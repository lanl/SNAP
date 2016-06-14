/*
 * Initializer.cpp
 *
 *  Created on: Jun 21, 2015
 *      Author: payne
 */

#include "Initializer.h"


EpochKernelArgs Initializer::genArgs(Context ctx, HighLevelRuntime* runtime,
                            LRWrapper chunkArrays,LRWrapper lw_slgg)
{
	int nCx(params.nCx),nCy(params.nCy),nCz(params.nCz);
	// Create the partitioning
	LPWrapper lp_chunkArrays;
//	LPWrapper lp_chunkArrays2;

	lp_chunkArrays.simpleSubPart(ctx,runtime,chunkArrays,nCx*nCy*nCz);
//	lp_chunkArrays2.slicedPart(ctx,runtime,chunkArrays,"%1",":",":");

	LPWrapper lp_slgg;
	lp_slgg.singlePart(ctx,runtime,lw_slgg);

	using namespace FIDs;
	std::vector<FieldPrivlages> parent_privs;
	parent_privs.push_back(FieldPrivlages(Array::scalFluxes,READ_WRITE));
	parent_privs.push_back(FieldPrivlages(Array::sigmaT,READ_WRITE));
	parent_privs.push_back(FieldPrivlages(Array::sigmaS,READ_WRITE));
	parent_privs.push_back(FieldPrivlages(Array::mat,READ_WRITE));



	EpochKernelArgs args;

	args.add_arg(lp_chunkArrays,Array::scalFluxes,READ_ONLY,EXCLUSIVE);
	args.add_arg(lp_chunkArrays,Array::sigmaT,READ_ONLY,EXCLUSIVE);
	args.add_arg(lp_chunkArrays,Array::sigmaS,READ_ONLY,EXCLUSIVE);
	args.add_arg(lp_chunkArrays,Array::mat,READ_ONLY,EXCLUSIVE);
	args.add_arg(lp_slgg,0,READ_ONLY,EXCLUSIVE);



	typedef std::pair<FieldID,FieldID> FieldPair;
	std::map<FieldPair,PrivilegeMode> child_privs;
	child_privs = propagate_nested_regions_privs(ctx,runtime,chunkArrays,parent_privs);


	std::map<Color,std::vector<RegionRequirement>> rreqs;
	rreqs = distribute_nested_regions(ctx,runtime,chunkArrays,child_privs);

	args.nested_reqs.insert(rreqs.begin(),rreqs.end());

//	for(auto& reqv : rreqs)
//	{
//		int ireg = 2;
//		for(auto& req : reqv.second)
//		{
//			const char* lr_name;
//			runtime->retrieve_name(req.parent,lr_name);
//			if(ireg == 2)
//			printf("Initializer Adding extra region requirement %i for color %i with lr %s\n",ireg,reqv.first,lr_name);
//			ireg++;
//		}
//
//	}


	return args;



}

__host__
void Initializer::evaluate(const Task* task,Context ctx,HighLevelRuntime* rt,
			  LegionMatrix<LRWrapper> lw_scalFluxes,
			  LegionMatrix<LRWrapper> lw_sigmaT,
			  LegionMatrix<LRWrapper> lw_sigmaS,
			  LegionMatrix<LRWrapper> lw_mat,
			  LegionMatrix<double> slgg)
{

	int nx=params.nx;
	int ny=params.ny;
	int nz = params.nz;
	int nx_l(params.nx_l),ny_l(params.ny_l),nz_l(params.nz_l);
	int nCx(params.nCx), nCy(params.nCy), nCz(params.nCz);
	int ng(params.ng),nang(params.nang),nmom(params.nmom);
	int noct(params.noct);
	int nmat(params.nmat);
	int cmom(params.cmom);

	int i_global = task->index_point.point_data[0];

	printf("In the initializer %i\n",i_global);


	typedef LegionMatrix<double> lMatrix_d;
	typedef LegionMatrix<int> lMatrix_i;

	lMatrix_d qi(ctx,lw_sigmaT(i_global).cast(),FIDs::SigmaT::qi);
	lMatrix_d t_xs(ctx,lw_sigmaT(i_global).cast(),FIDs::SigmaT::t_xs);
	lMatrix_d a_xs(ctx,lw_sigmaT(i_global).cast(),FIDs::SigmaT::a_xs);
	lMatrix_d s_xs(ctx,lw_sigmaS(i_global).cast(),FIDs::SigmaS::s_xs);
	lMatrix_i mat(ctx,lw_mat(i_global).cast(),FIDs::Mat::mat);

	printf("finished setting up the matrices\n");
	for(int k=0;k<nz_l;k++)
		for(int j=0;j<ny_l;j++)
			for(int i=0;i<nx_l;i++)
			{
				mat(i,j,k) = i_global%nmat;
			}




	for(int k=0;k<nz_l;k++)
		for(int j=0;j<ny_l;j++)
			for(int i=0;i<nx_l;i++)
				for(int ig=0;ig<ng;ig++)
			{
				if(mat(i,j,k).cast() == 0)
				{
					a_xs(i,j,k,ig) = 0.5 + ig*0.005;
				}
				else if(mat(i,j,k).cast() == 1)
				{
					a_xs(i,j,k,ig) = 0.8 + ig*0.005;
				}

				for(int l=0;l<nmom;l++)
					s_xs(l,i,j,k,ig) = slgg(mat(i,j,k),l,ig,ig).cast();


				t_xs(i,j,k,ig) = (mat(i,j,k)+1.0) + ig*0.01;
			}


//	for(int k=0;k<nz_l;k++)
//		for(int j=0;j<ny_l;j++)
//			for(int i=0;i<nx_l;i++)
//				printf("mat(%i,%i,%i) = %i\n",i,j,k,mat(i,j,k).cast());


	// Only doing src opt 1
	int i1 = nx/4;
	int i2 = 3*nx/4-1;
	int j1 = ny/4;
	int j2 = 3*ny/4-1;
	int k1 = nz/4;
	int k2 = 3*nz/4-1;

	int iC = i_global%nCx;
	int jC = (i_global/nCx)%nCy;
	int kC = i_global/(nCx*nCy);

	for(int k=0;k<nz_l;k++)
		for(int j=0;j<ny_l;j++)
			for(int i=0;i<nx_l;i++)
			{
				if(( iC+i >= i1 && iC+i <= i2) &&
					(jC+j >= j1 && jC+j <= j2) &&
					(kC+k >= k1 && kC+k <= k2))
				{
					for(int ig=0;ig<ng;ig++)
						qi(i,j,k,ig) = 1.0;
				}
				else
				{
					for(int ig=0;ig<ng;ig++)
						qi(i,j,k,ig) = 0.0;
				}

			}





}



void Initializer::register_task()
{
	MustEpochKernelLauncher<Initializer>::register_cpu();
}


#include <iostream>
#include <boost/tokenizer.hpp>
#include <string>
#include <boost/program_options.hpp>
#include <LRWrapper.h>
#include <LegionMatrix.h>
#include <boost/foreach.hpp>
#include <boost/algorithm/string.hpp>
#include <sstream>
#include "SlappyParams.h"
#include "Initializer.h"
#include "BetterMapper.h"
#include <MustEpochKernelLauncher.h>
#include <LegionHelper.h>
#include "InGroupSource.h"
#include "Sweep.h"
#include "SweepMapper.h"
//#include "gnuplot_i.h"

#include "CrossGroupSrc.h"
enum Task_IDs
{
	TOP_LEVEL_TASK_ID
};









using namespace Dragon;

int setup_diags(Context ctx, HighLevelRuntime* runtime,
                 LRWrapper& diags, LRWrapper& lines,
				 int& ndiag, int nx_l, int ny_l, int nz_l);

void setup_slgg(Context ctx, HighLevelRuntime* runtime,LRWrapper lw_slgg,int nmat,int nmom,int ng)
{
	double sigS[nmat][ng];
	for(int g=0;g<ng;g++)
		for(int m=0;m<nmat;m++)
		{
			if(g == 0)
			{
				if(m == 0)
					sigS[m][g] = 0.5;
				else
					sigS[m][g] = 1.2;
			}
			else
				sigS[m][g] = sigS[m][g-1]+0.005;
		}

RegionRequirement rr(lw_slgg.lr,READ_WRITE,EXCLUSIVE,lw_slgg.lr);
		rr.add_field(0);

		printf("Mapping slgg\n");
PhysicalRegion pr = runtime->map_region(ctx,rr);

LegionMatrix<double> slgg(lw_slgg,pr,0);



for(int g2 = 0;g2<ng;g2++)
	for(int g1=0;g1<ng;g1++)
		for(int m=0;m<nmat;m++)
		{
			if(m == 0)
			{
				if(g1 == g2)
					slgg(m,0,g1,g2) = 0.2*sigS[m][g1];
				else if(g2 < g1)
					if(g1 > 0)
						slgg(m,0,g1,g2) = 0.1*sigS[m][g1]/((double)g1);
					else
						slgg(m,0,g1,g2) = 0.3*sigS[m][g1];
				else
					if(g1 < ng-1)
						slgg(m,0,g1,g2) = 0.7*sigS[m][g1]/((double)(ng-g1-1));
					else
						slgg(m,0,g1,g2) = 0.9*sigS[m][g1];
			}
			else
			{
				if(g1 == g2)
					slgg(m,0,g1,g2) = 0.5*sigS[m][g1];
				else if(g2 < g1)
					if(g1 > 0)
						slgg(m,0,g1,g2) = 0.1*sigS[m][g1]/((double)g1);
					else
						slgg(m,0,g1,g2) = 0.6*sigS[m][g1];
				else
					if(g1 < ng-1)
						slgg(m,0,g1,g2) = 0.4*sigS[m][g1]/((double)(ng-g1-1));
					else
						slgg(m,0,g1,g2) = 0.9*sigS[m][g1];
			}
		}

if(nmom > 1)
{
	for(int m=0;m<nmat;m++)
	{
		double coef1,coef2;
		if(m==0)
		{coef1=0.1;coef2=0.5;}
		else
		{coef1=0.8;coef2=0.6;}
		for(int g2=0;g2<ng;g2++)
			for(int g1=0;g1<ng;g1++)
			{
				slgg(m,1,g1,g2) = coef1*slgg(m,0,g1,g2);
				for(int n=2;n<nmom;n++)
					slgg(m,n,g1,g2) = coef2*slgg(m,n-1,g1,g2);
			}


	}
}

runtime->unmap_region(ctx,pr);

}


void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{

	runtime->disable_profiling();
	namespace po = boost::program_options;
	using namespace std;

	const InputArgs &command_args = HighLevelRuntime::get_input_args();
	SlappyParams params(command_args);


	int nx=params.nx;
	int ny=params.ny;
	int nz = params.nz;
	int nx_l(params.nx_l),ny_l(params.ny_l),nz_l(params.nz_l);
	int nCx(params.nCx), nCy(params.nCy), nCz(params.nCz);
	int ng(params.ng),nang(params.nang),nmom(params.nmom);
	int noct(params.noct);
	int nmat(params.nmat);
	int cmom(params.cmom);

	int& nDiag = params.nDiag;
	int& nPoints = params.nPoints;


	LRWrapper allArrays;
	allArrays.create(ctx,runtime,"allArrays",{1},0,LRWrapper(),1,LRWrapper(),2,LRWrapper(),3,LRWrapper());
	// Contains the by-chunk decomposition for all chunk dependent arrays
	LRWrapper chunkArrays;
	chunkArrays.create(ctx,runtime,"chunkArrays",{nCx,nCy,nCz},
	                   FIDs::Array::angFluxes,LRWrapper(),
	                   FIDs::Array::scalFluxes,LRWrapper(),
	                   FIDs::Array::savedFlux,LRWrapper(),
	                   FIDs::Array::sigmaT,LRWrapper(),
	                   FIDs::Array::sigmaS,LRWrapper(),
	                   FIDs::Array::mat,LRWrapper(),
	                   FIDs::Array::bci,LRWrapper(),
	                   FIDs::Array::bcj,LRWrapper(),
	                   FIDs::Array::bck,LRWrapper());

	// Setup all of the interior logical regions
	{
	RegionRequirement rr(chunkArrays.lr,WRITE_ONLY,EXCLUSIVE,chunkArrays.lr);
			rr.add_field(FIDs::Array::angFluxes);
			rr.add_field(FIDs::Array::scalFluxes);
			rr.add_field(FIDs::Array::savedFlux);
			rr.add_field(FIDs::Array::sigmaT);
			rr.add_field(FIDs::Array::sigmaS);
			rr.add_field(FIDs::Array::mat);
			rr.add_field(FIDs::Array::bci);
			rr.add_field(FIDs::Array::bcj);
			rr.add_field(FIDs::Array::bck);



	PhysicalRegion pr = runtime->map_region(ctx,rr);

	LegionMatrix<LRWrapper> angFluxes(chunkArrays,pr,FIDs::Array::angFluxes);
	LegionMatrix<LRWrapper> scalFluxes(chunkArrays,pr,FIDs::Array::scalFluxes);
	LegionMatrix<LRWrapper> savedFluxes(chunkArrays,pr,FIDs::Array::savedFlux);
	LegionMatrix<LRWrapper> sigmaT(chunkArrays,pr,FIDs::Array::sigmaT);
	LegionMatrix<LRWrapper> sigmaS(chunkArrays,pr,FIDs::Array::sigmaS);
	LegionMatrix<LRWrapper> mat(chunkArrays,pr,FIDs::Array::mat);
	LegionMatrix<LRWrapper> bci(chunkArrays,pr,FIDs::Array::bci);
	LegionMatrix<LRWrapper> bcj(chunkArrays,pr,FIDs::Array::bcj);
	LegionMatrix<LRWrapper> bck(chunkArrays,pr,FIDs::Array::bck);




	for(int k=0;k<nCz;k++)
		for(int j=0;j<nCy;j++)
			for(int i=0;i<nCx;i++)
			{
				{
				char* cell_id;
				asprintf(&cell_id," %i, %i, %i,",i,j,k);
				auto cn = [&](const char* name)
		{
					std::string res(name);
					return res+cell_id;
		};
				angFluxes(i,j,k) = (createLRWrapper(ctx,runtime,cn("angFluxes").c_str(),{nang,nx_l,ny_l,nz_l,ng,noct},
				                   FIDs::AngFlux::ptr_in,double(),
				                   FIDs::AngFlux::ptr_out,double()));

				scalFluxes(i,j,k) = createLRWrapper(ctx,runtime,cn("scalFluxes").c_str(),{cmom,nx_l,ny_l,nz_l,ng},
								   FIDs::ScalFlux::q2grp,double(),
								   FIDs::ScalFlux::qtot,double(),
				                   FIDs::ScalFlux::flux,double()
				                   );
				savedFluxes(i,j,k) = createLRWrapper(ctx,runtime,cn("savedFluxes").c_str(),{nx_l,ny_l,nz_l,ng},
				                   FIDs::SavedFlux::fluxpo,double(),
				                   FIDs::SavedFlux::fluxpi,double());

				sigmaT(i,j,k) = createLRWrapper(ctx,runtime,cn("sigmaT").c_str(),{nx_l,ny_l,nz_l,ng},
				                   FIDs::SigmaT::t_xs,double(),
				                   FIDs::SigmaT::a_xs,double(),
				                   FIDs::SigmaT::qi,double());

				sigmaS(i,j,k) = createLRWrapper(ctx,runtime,cn("sigmaS").c_str(),{nmom,nx_l,ny_l,nz_l,ng},
				                   FIDs::SigmaS::s_xs,double());

				mat(i,j,k) = createLRWrapper(ctx,runtime,cn("mat").c_str(),{nx_l,ny_l,nz_l},
				                   FIDs::Mat::mat,int());

				bci(i,j,k) = createLRWrapper(ctx,runtime,cn("bci").c_str(),{nang,ny_l,nz_l,ng},
				                   FIDs::BCi::ib_out,double(),1,double());
				bcj(i,j,k) = createLRWrapper(ctx,runtime,cn("bcj").c_str(),{nang,nx_l,nz_l,ng},
				                   FIDs::BCj::jb_out,double(),1,double());
				bck(i,j,k) = createLRWrapper(ctx,runtime,cn("bck").c_str(),{nang,nx_l,ny_l,ng},
				                   FIDs::BCk::kb_out,double(),1,double());
				}


			}


	runtime->unmap_region(ctx,pr);
	runtime->unmap_all_regions(ctx);




	}

	LRWrapper lw_slgg;
	lw_slgg.create(ctx,runtime,"slgg",{nmat,nmom,ng,ng},0,double());
	setup_slgg(ctx,runtime,lw_slgg,nmat,nmom,ng);

	// Setup all of the interior logical regions

	LRWrapper diags;
	LRWrapper lines;

	printf("Setting up Diags\n");
	nPoints = setup_diags(ctx,runtime,diags,lines,nDiag,nx_l,ny_l,nz_l);







	// Initialize stuff
	Initializer setup_stuff(params);

	auto init_op = genMustEpochKernel(setup_stuff,ctx,runtime,chunkArrays,lw_slgg);
//	runtime->unmap_all_regions(ctx);

	init_op.execute(ctx,runtime);
	printf("finished init op\n");

//	{
//
//	RegionRequirement rr(chunkArrays.lr,READ_ONLY,SIMULTANEOUS,chunkArrays.lr);
//			rr.add_field(FIDs::Array::sigmaT);
//			rr.add_field(FIDs::Array::sigmaS);
//
//
//
//
//	PhysicalRegion pr = runtime->map_region(ctx,rr);
//	LegionMatrix<LRWrapper> sigmaT(chunkArrays,pr,FIDs::Array::sigmaT);
//	LegionMatrix<LRWrapper> sigmaS(chunkArrays,pr,FIDs::Array::sigmaS);
//
//	float* x = (float*)malloc(nx*sizeof(float));
//	float* y = (float*)malloc(ny*sizeof(float));
//	float* qi_z = (float*)malloc(nx*ny*sizeof(float));
//
//
//	for(int jC=0;jC<nCy;jC++)
//		for(int iC=0;iC<nCx;iC++)
//		{
//			RegionRequirement rr_sigmaT(sigmaT(iC,jC,0).cast().lr,READ_ONLY,EXCLUSIVE,sigmaT(iC,jC,0).cast().lr);
//			rr_sigmaT.add_field(FIDs::SigmaT::t_xs);
//			rr_sigmaT.add_field(FIDs::SigmaT::qi);
//
//			RegionRequirement rr_sigmaS(sigmaS(iC,jC,0).cast().lr,READ_ONLY,EXCLUSIVE,sigmaS(iC,jC,0).cast().lr);
//			rr_sigmaS.add_field(FIDs::SigmaS::s_xs);
//
//			PhysicalRegion pr_sigmaT = runtime->map_region(ctx,rr_sigmaT);
//			PhysicalRegion pr_sigmaS = runtime->map_region(ctx,rr_sigmaS);
//
//			LegionMatrix<double> t_xs(sigmaT(iC,jC,0).cast(),pr_sigmaT,FIDs::SigmaT::t_xs);
//			LegionMatrix<double> qi(sigmaT(iC,jC,0).cast(),pr_sigmaT,FIDs::SigmaT::qi);
//			LegionMatrix<double> s_xs(sigmaS(iC,jC,0).cast(),pr_sigmaS,FIDs::SigmaS::s_xs);
//
//			for(int j=0;j<ny_l;j++)
//				for(int i=0;i<nx_l;i++)
//				{
//					int ix = i+nx_l*iC;
//					int iy = j+ny_l*jC;
//
//					x[ix] = ix;
//					y[iy] = iy;
////					printf("t_xs(%i, %i, 0) = %e\n",ix,iy,t_xs(i,j,0,0).cast());
//					qi_z[ix+nx*iy] = t_xs(i,j,0,0).cast();
//
//				}
//
//
//
//			runtime->unmap_region(ctx,pr_sigmaT);
//			runtime->unmap_region(ctx,pr_sigmaS);
//
//
//		}
//
//	gnuplot_ctrl* qi_plt = gnuplot_init();
//
//	gnuplot_plot_xyz(qi_plt,x,y,qi_z,nx,ny,"qi");
//	getchar();
//
//	runtime->unmap_region(ctx,pr);
//
//	runtime->unmap_all_regions(ctx);
//
//	}


	{
	RegionRequirement rr(allArrays.lr,READ_WRITE,EXCLUSIVE,allArrays.lr);
	rr.add_field(0);
	rr.add_field(1);
	rr.add_field(2);
	rr.add_field(3);

PhysicalRegion pr = runtime->map_region(ctx,rr);
	{


	LegionMatrix<LRWrapper> _allArrays(allArrays,pr,0);
	LegionMatrix<LRWrapper> _slgg(allArrays,pr,1);
	LegionMatrix<LRWrapper> _diag(allArrays,pr,2);
	LegionMatrix<LRWrapper> _lines(allArrays,pr,3);


	_allArrays(0) = chunkArrays;
	_slgg(0) = lw_slgg;
	_diag(0) = diags;
	_lines(0) = lines;

	}

	runtime->unmap_region(ctx,pr);

	}




	CrossGroupSrc outer(params);

	auto outer_op = genSingleKernel(outer, ctx, runtime, allArrays, chunkArrays, lw_slgg, diags,lines);

	printf("Executing outer op\n");
	runtime->execute_task(ctx,outer_op).get_void_result();
	printf("finished!\n");
}

static void update_mappers(Machine machine, HighLevelRuntime *rt,
						   const std::set<Processor> &local_procs)
{
	printf("Updating mappers\n");
  for (std::set<Processor>::const_iterator it = local_procs.begin();
		it != local_procs.end(); it++)
  {
//	rt->add_mapper(1,new SweepMapper(machine, rt, *it), *it);
    rt->replace_default_mapper(new BetterMapper(machine, rt, *it), *it);
  }
}

int main(int argc, char **argv)
{

  char hostname[512];
  gethostname(hostname,512);

  srand(23948723);
//
//	  dup2(fileno(err_file),STDERR_FILENO);
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/);


//	  fclose(err_file);



//	  std::stringstream buffer;
  setbuf(stdout,NULL);
  setvbuf(stderr,NULL,_IOLBF,1024);

  system("export GASNET_BACKTRACE=1");

//  TaskHelper::register_hybrid_variants<FieldKernelOp<Initializer>>();
//  TaskHelper::register_hybrid_variants<FieldKernelOp<CrossGroupSrc>>();
//  TaskHelper::register_hybrid_variants<FieldKernelOp<OuterSrcScat>>();
//  TaskHelper::register_hybrid_variants<FieldKernelOp<OuterConv>>();
//  TaskHelper::register_hybrid_variants<FieldKernelOp<SaveFluxes>>();
//  TaskHelper::register_hybrid_variants<FieldKernelOp<InGroupSource>>();
//  TaskHelper::register_hybrid_variants<FieldKernelOp<InnerSrcScat>>();

//  RegisterDim3Variants();

  TaskHelper::register_hybrid_variants<LegionHelper::Setter<bool>>();
  TaskHelper::register_hybrid_variants<LegionHelper::Setter<int>>();
  TaskHelper::register_hybrid_variants<LegionHelper::Setter<double>>();
  TaskHelper::register_hybrid_variants<LegionHelper::Setter<LRWrapper>>();
  TaskHelper::register_cpu_variants<LegionHelper::Getter<int>>();
  TaskHelper::register_cpu_variants<LegionHelper::Getter<bool>>();

  CrossGroupSrc::register_cpu();
  InnerSrcScat::register_cpu();






  Initializer::register_task();

  register_sweep_tasks();

//  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<0>>>();
//  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<1>>>();
//  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<2>>>();
//  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<3>>>();
//  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<4>>>();
//  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<5>>>();
//  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<6>>>();
//  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<7>>>();
//  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<8>>>();
//  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<9>>>();
//  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<10>>>();
//  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<11>>>();
//  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<12>>>();
//  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<13>>>();
//  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<14>>>();
//  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<15>>>();


	  HighLevelRuntime::set_registration_callback(update_mappers);

  return HighLevelRuntime::start(argc, argv);
}

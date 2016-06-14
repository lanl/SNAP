/*
 * Sweep.cc
 *
 *  Created on: Jun 16, 2015
 *      Author: payne
 */

#include "Sweep.h"
#include <utilities.h>


SweepBase::SweepBase(Context ctx, HighLevelRuntime* runtime,
                           SlappyParams _params,
                           LegionMatrix<LRWrapper> lr_scalFlux,
                           LegionMatrix<LRWrapper> lr_sigmaT,
                           LegionMatrix<LRWrapper> lr_bci,
                           LegionMatrix<LRWrapper> lr_bcj,
                           LegionMatrix<LRWrapper> lr_bck,
                           LRWrapper lr_dogrp,
             			   int _iproc, int _jproc, int _kproc) :params(_params)
{

	lp_scalFlux = new LPWrapper();
	lp_sigmaT = new LPWrapper();
	lp_bci_out = new LPWrapper();
	lp_bcj_out = new LPWrapper();
	lp_bck_out = new LPWrapper();

	lp_dogrp = new LPWrapper();
	iproc = _iproc;
	jproc = _jproc;
	kproc = _kproc;
	n_split = std::min(1,params.ng);

	char* split_val;
	asprintf(&split_val,"%%%i",n_split);

//	lp_scalFlux->slicedPart(ctx,runtime,lr_scalFlux(iproc,jproc,kproc),":",":",":",":",split_val);
//	lp_sigmaT->slicedPart(ctx,runtime,lr_sigmaT(iproc,jproc,kproc),":",":",":",split_val);
//
//	lp_bci_out->slicedPart(ctx,runtime,lr_bci(iproc,jproc,kproc),":",":",":",split_val);
//	lp_bcj_out->slicedPart(ctx,runtime,lr_bcj(iproc,jproc,kproc),":",":",":",split_val);
//	lp_bck_out->slicedPart(ctx,runtime,lr_bck(iproc,jproc,kproc),":",":",":",split_val);
//
//	lp_dogrp->slicedPart(ctx,runtime,lr_dogrp,split_val);


	lp_scalFlux->singlePart(ctx,runtime,lr_scalFlux(iproc,jproc,kproc));
	lp_sigmaT->singlePart(ctx,runtime,lr_sigmaT(iproc,jproc,kproc));

	lp_bci_out->singlePart(ctx,runtime,lr_bci(iproc,jproc,kproc));
	lp_bcj_out->singlePart(ctx,runtime,lr_bcj(iproc,jproc,kproc));
	lp_bck_out->singlePart(ctx,runtime,lr_bck(iproc,jproc,kproc));

	lp_dogrp->singlePart(ctx,runtime,lr_dogrp);



}

LegionRuntime::Logger::Category log_sweep("log_sweep");

// Output the start time in microseconds
template<int iOct>
void Sweep<iOct>::on_start(void)
{
	struct timespec tp;
//	clock_gettime(CLOCK_MONOTONIC,&tp);
//	log_sweep.info("<sweep_prof>: [0,%i,%i,%i,%i] = %lu",iproc,jproc,kproc, id+2*(jd+2*kd),(long long int)(1.0e6*tp.tv_sec+tp.tv_nsec/1.0e3));
}

// Output the finish time in microseconds
template<int iOct>
void Sweep<iOct>::on_finish(void)
{
	struct timespec tp;
//	clock_gettime(CLOCK_MONOTONIC,&tp);
//	log_sweep.info("<sweep_prof>: [1,%i,%i,%i,%i] = %lu",iproc,jproc,kproc, id+2*(jd+2*kd),(long long int)(1.0e6*tp.tv_sec+tp.tv_nsec/1.0e3));
}


template<int iOct>
EpochKernelArgs Sweep<iOct>::genArgs(Context ctx, HighLevelRuntime* runtime,
                                  	   SweepBase* base,
                                       LegionMatrix<LRWrapper> lr_angFlux,
                        			   LegionMatrix<LRWrapper> lr_bci,
                        			   LegionMatrix<LRWrapper> lr_bcj,
                        			   LegionMatrix<LRWrapper> lr_bck,
                         			   LPWrapper lp_diag, // Copied Partitioning
                         			   LPWrapper lp_lines, // Copied Partitioning
                        			   int _id, int _jd, int _kd)
{
	iproc = base->iproc;
	jproc = base->jproc;
	kproc = base->kproc;

	LPWrapper& lp_scalFlux = *base->lp_scalFlux;
	LPWrapper& lp_sigmaT = *base->lp_sigmaT;
	LPWrapper& lp_bci_out = *base->lp_bci_out;
	LPWrapper& lp_bcj_out = *base->lp_bcj_out;
	LPWrapper& lp_bck_out = *base->lp_bck_out;

	LPWrapper& lp_dogrp = *base->lp_dogrp;

	id = _id;
	jd = _jd;
	kd = _kd;
	n_split = base->n_split;

	ioct = id+2*(jd+2*kd);



	int nCx(params.nCx), nCy(params.nCy), nCz(params.nCz);
	int nx_l(params.nx_l), ny_l(params.ny_l), nz_l(params.nz_l);

	int iproc_hi,iproc_lo;
	int jproc_hi,jproc_lo;
	int kproc_hi,kproc_lo;
	int inc,jnc,knc;

	if(id == 0)
	{iproc_hi=-1;iproc_lo=nCx-1; inc=-1;}
	else
	{iproc_lo=0;iproc_hi=nCx; inc=1;}

	if(jd == 0)
	{jproc_hi=-1;jproc_lo=nCy-1; jnc=-1;}
	else
	{jproc_lo=0;jproc_hi=nCy; jnc=1;}

	if(kd == 0)
	{kproc_hi=-1;kproc_lo=nCz-1; knc=-1;}
	else
	{kproc_lo=0;kproc_hi=nCz; knc=1;}

	if(iproc == 0)
		firstx = true;
	else
		firstx = false;

	if(iproc == nCx-1)
		lastx = true;
	else
		lastx = false;


	if(jproc == 0)
		firsty = true;
	else
		firsty = false;

	if(jproc == nCy-1)
		lasty = true;
	else
		lasty = false;

	if(kproc == 0)
		firstz = true;
	else
		firstz = false;

	if(kproc == nCz-1)
		lastz = true;
	else
		lastz = false;

	int xlop,xhip;
	int ylop,yhip;
	int zlop,zhip;
	int xp_snd,xp_rcv;
	int yp_snd,yp_rcv;
	int zp_snd,zp_rcv;

	// Processor setup
	if(iproc > 0)
		xlop = iproc-1;
	else
		xlop = iproc;

	if(iproc < (nCx-1))
		xhip = iproc+1;
	else
		xhip = iproc;

	// Processor setup
	if(jproc > 0)
		ylop = jproc-1;
	else
		ylop = jproc;

	if(jproc < (nCy-1))
		yhip = jproc+1;
	else
		yhip = jproc;

	// Processor setup
	if(kproc > 0)
		zlop = kproc-1;
	else
		zlop = kproc;

	if(kproc < (nCz-1))
		zhip = kproc+1;
	else
		zhip = kproc;

	if(id == 0)
	{ilo=nx_l-1;ihi=0;ist=-1;xp_snd=xlop;xp_rcv=xhip;}
	else
	{ilo=0;ihi=nx_l-1;ist=1;xp_snd=xhip;xp_rcv=xlop;}

	if(jd == 0)
	{jlo=ny_l-1;jhi=0;jst=-1;yp_snd=ylop;yp_rcv=yhip;}
	else
	{jlo=0;jhi=ny_l-1;jst=1;yp_snd=yhip;yp_rcv=ylop;}

	if(kd == 0)
	{klo=nz_l-1;khi=0;kst=-1;zp_snd=zlop;zp_rcv=zhip;}
	else
	{klo=0;khi=nz_l-1;kst=1;zp_snd=zhip;zp_rcv=zlop;}

	char split_val[8];
	sprintf(split_val,"%%%i",n_split);

	LPWrapper* lp_bci_in = new LPWrapper();
	LPWrapper* lp_bcj_in = new LPWrapper();
	LPWrapper* lp_bck_in = new LPWrapper();
//	if(xp_rcv != iproc)
//		lp_bci_in->slicedPart(ctx,runtime,lr_bci(xp_rcv,jproc,kproc),":",":",":",split_val);
//	else
//		lp_bci_in = &lp_bci_out;
//
//	if(yp_rcv != jproc)
//		lp_bcj_in->slicedPart(ctx,runtime,lr_bcj(iproc,yp_rcv,kproc),":",":",":",split_val);
//	else
//		lp_bcj_in = &lp_bcj_out;
//
//	if(zp_rcv != kproc)
//		lp_bck_in->slicedPart(ctx,runtime,lr_bck(iproc,jproc,zp_rcv),":",":",":",split_val);
//	else
//		lp_bck_in = &lp_bck_out;

	if(xp_rcv != iproc)
		lp_bci_in->singlePart(ctx,runtime,lr_bci(xp_rcv,jproc,kproc));
	else
		lp_bci_in = &lp_bci_out;

	if(yp_rcv != jproc)
		lp_bcj_in->singlePart(ctx,runtime,lr_bcj(iproc,yp_rcv,kproc));
	else
		lp_bcj_in = &lp_bcj_out;

	if(zp_rcv != kproc)
		lp_bck_in->singlePart(ctx,runtime,lr_bck(iproc,jproc,zp_rcv));
	else
		lp_bck_in = &lp_bck_out;

	LPWrapper* lp_angFlux = new LPWrapper();
	lp_angFlux->slicedPart(ctx,runtime,lr_angFlux(iproc,jproc,kproc),":",":",":",":",":",id+2*(jd+2*kd));


	EpochKernelArgs args;
	args.add_arg(lp_dogrp,0,READ_ONLY,EXCLUSIVE);
	args.add_arg(lp_diag,FID_DiagLen,READ_ONLY,EXCLUSIVE);
	args.add_arg(lp_diag,FID_DiagStart,READ_ONLY,EXCLUSIVE);

	args.add_arg(lp_lines,FID_ic,READ_ONLY,EXCLUSIVE);
	args.add_arg(lp_lines,FID_j,READ_ONLY,EXCLUSIVE);
	args.add_arg(lp_lines,FID_k,READ_ONLY,EXCLUSIVE);

	using namespace FIDs;
	args.add_arg(*lp_angFlux,AngFlux::ptr_in,READ_ONLY,EXCLUSIVE);
	args.add_arg(*lp_angFlux,AngFlux::ptr_out,READ_WRITE,EXCLUSIVE);

	args.add_arg(lp_scalFlux,ScalFlux::qtot,READ_ONLY,EXCLUSIVE);
	args.add_arg(lp_scalFlux,ScalFlux::flux,READ_WRITE,EXCLUSIVE);
	args.add_arg(lp_sigmaT,SigmaT::t_xs,READ_ONLY,EXCLUSIVE);

	args.add_arg(lp_bci_out,id,READ_WRITE,EXCLUSIVE);
	args.add_arg(lp_bcj_out,jd,READ_WRITE,EXCLUSIVE);
	args.add_arg(lp_bck_out,kd,READ_WRITE,EXCLUSIVE);

	if(xp_rcv != iproc)
		args.add_arg(*lp_bci_in,id,READ_ONLY,EXCLUSIVE);
	else
		args.add_arg(lp_bci_out,id,READ_WRITE,EXCLUSIVE);

	if(yp_rcv != jproc)
		args.add_arg(*lp_bcj_in,jd,READ_ONLY,EXCLUSIVE);
	else
		args.add_arg(lp_bcj_out,jd,READ_WRITE,EXCLUSIVE);



	if(zp_rcv != kproc)
		args.add_arg(*lp_bck_in,kd,READ_ONLY,EXCLUSIVE);
	else
		args.add_arg(lp_bck_out,kd,READ_WRITE,EXCLUSIVE);






//	args.loop_domain.rect_data[0] = 0;
//	args.loop_domain.rect_data[1] = params.nDiag-1;


	return args;

}


template<int iOct>
__host__
void Sweep<iOct>::evaluate(int iDom,LegionMatrix<bool> dogrp,
                            LegionMatrix<int> diag_len,
                            LegionMatrix<int> diag_start,
                            LegionMatrix<int> _i,
                            LegionMatrix<int> _j,
                            LegionMatrix<int> _k,
                            LegionMatrix<double> ptr_in,
                            LegionMatrix<double> ptr_out,
                            LegionMatrix<double> qtot,
                            LegionMatrix<double> flux,
                            LegionMatrix<double> t_xs,
                            LegionMatrix<double> bci_out,
						    LegionMatrix<double> bcj_out,
						    LegionMatrix<double> bck_out,
                            LegionMatrix<double> bci_in,
						    LegionMatrix<double> bcj_in,
						    LegionMatrix<double> bck_in)
{

//	int igrp = iDom;
//	int iDiag = iDom%params.nDiag;
	int seed = rand()%5928347;
	for(int i=0;i<iDom;i++)
		seed = (seed + rand()%5928347)%5928347;
//	srand(seed);

	LegionMatrix<double> psii = bci_out;
	LegionMatrix<double> psij = bcj_out;
	LegionMatrix<double> psik = bck_out;

//	auto vdelt = [&](int _g){return 2.0/(params.dt*(params.ng-_g));};
//	int nang = params.nang;
//
//	double dm = 1.0/((double)nang);
//	auto mu = [&](int _iang){return (_iang+0.5)*dm;};
//	auto eta = [&](int _iang){return 1.0 - 0.5*dm - _iang*dm;};
//	auto xi = [&](int _iang)
//			{
//				double mut = mu(_iang);
//				double etat = eta(_iang);
//				double t = mut*mut+etat*etat;
//				return sqrt(1.0-t);
//			};
//	double w = 0.125/((double)nang);
//	double hi = 2.0/params.dx;
//	auto hj = [&](int _iang){return 2.0/params.dy * eta(_iang);};
//	auto hk = [&](int _iang){return 2.0/params.dz * xi(_iang);};
	auto dinv = [&](int _iang,int _i2, int _j2, int _k2, int _g)
					{return 1.0/(t_xs(_i2,_j2,_k2,_g).cast() + vdelt(_g) + mu(_iang)*hi + hj(_iang) + hk(_iang));};



	double psi[nang];
	double pc[nang];
	double den[nang];
	double hv[4*nang];
	double fxhv[4*nang];
	double f = 53.0;
	for(int g=iDom*n_split;g<((iDom+1)*n_split);g++)
	if(dogrp(g).cast())
	{


//		printf("in the sweep for group %i oct %i\n",g,ioct);
		for(int idiag=0;idiag<params.nDiag;idiag++)
		{
			int n_start = diag_start(idiag);

		for(int n=0;n<diag_len(idiag);n++)
		{
			double sum_hv = 0.0;

			int i,j,k;

			i = _i(n_start+n);


			if(ist < 0)
				i = params.nx_l - i -1;

			if(i < params.nx_l)
			{
//					printf("i == %i\n",i);
				j = _j(n_start+n);
				k = _k(n_start+n);
				if(jst<0) j = params.ny_l - j -1;
				if(kst<0) k = params.nz_l - k -1;

				assert(i < params.nx_l && i >= 0);
				assert(j < params.ny_l && j >= 0);
				assert(k < params.nz_l && k >= 0);



//				// x bc's
//				if(i==(nx-1) && ist==-1)
//					for(int ia=0;ia<nang;ia++) psii(ia,j,k,g) = 0;
//				else if(i==0 && ist==1)
//					for(int ia=0;ia<nang;ia++) psii(ia,j,k,g) = 0;

				// x bc's
				if(i == ilo)
				{
				if(id==0 && lastx)
					for(int ia=0;ia<nang;ia++) psii(ia,i,k,g) = 0;
				else if(id==1 && firstx)
					for(int ia=0;ia<nang;ia++) psii(ia,i,k,g) = 0;
				else
					for(int ia=0;ia<nang;ia++) psii(ia,i,k,g) = (double)(bci_in(ia,i,k,g));
				}

				// y bc's
				if(j == jlo)
				{
				if(jd==0 && lasty)
					for(int ia=0;ia<nang;ia++) psij(ia,i,k,g) = 0;
				else if(jd==1 && firsty)
					for(int ia=0;ia<nang;ia++) psij(ia,i,k,g) = 0;
				else
					for(int ia=0;ia<nang;ia++) psij(ia,i,k,g) = (double)(bcj_in(ia,i,k,g));
				}

				// z bc's
				if(k == klo)
				{
				if(kd==0 && lastz)
					for(int ia=0;ia<nang;ia++) psik(ia,i,j,g) = 0;
				else if(kd==1 && firstz)
					for(int ia=0;ia<nang;ia++) psik(ia,i,j,g) = 0;
				else
					for(int ia=0;ia<nang;ia++) psik(ia,i,j,g) = (double)(bck_in(ia,i,j,g));
				}

				angL([&](int ia){psi[ia] = qtot(0,i,j,k,g).cast();});

//				if(src_opt == 3)
//					angL([&](int ia){psi[ia] += qim(ia,i,j,k,ioct,g).cast();});

				int mom = 1;
				for(int l=1;l<params.nmom;l++)
				for(int m=0;m<(2*l + 1);m++)
				{
					angL([&](int ia){psi[ia] += ec(ia,l,m)*qtot(l,i,j,k,g).cast();});
					mom++;
				}

				angL([&](int ia){pc[ia] = psi[ia] + psii(ia,j,k,g)*mu(ia)*hi
					+ psij(ia,i,k,g)*hj(ia)
					+ psik(ia,i,j,g)*hk(ia);});

				if(vdelt(g) != 0.0) angL([&](int ia){pc[ia] += vdelt(g)*(ptr_in(ia,i,j,k,ioct,g).cast());});

				if(params.fixup == 0)
				{
					angL([&](int ia){psi[ia] = pc[ia]*dinv(ia,i,j,k,g);});
					angL([&](int ia){psii(ia,j,k,g) = 2.0*psi[ia] - psii(ia,j,k,g);});
					angL([&](int ia){psij(ia,i,k,g) = 2.0*psi[ia] - psij(ia,i,k,g);});

					if(params.ndimen == 3) angL([&](int ia){psik(ia,i,j,g) = 2.*psi[ia] - psik(ia,i,j,g);});
					if(vdelt(g) != 0.0)
						angL([&](int ia){ptr_out(ia,i,j,k,ioct,g) = 2.0*psi[ia] - ptr_in(ia,i,j,k,ioct,g); });
				}
				else
				{
					angL([&](int ia){hv[ia] = 1.0;hv[ia+nang] = 1.0;hv[ia+2*nang] = 1.0;hv[ia+3*nang] = 1.0;});
					for(int l=0;l<4;l++)
						angL([&](int ia){sum_hv += hv[ia+nang*l];});

					angL([&](int ia){pc[ia] *= dinv(ia,i,j,k,g);});

					int iprotect = 0;
					while(iprotect < 4)
					{

						angL([&](int ia){fxhv[ia] = 2.*pc[ia] - psii(ia,j,k,g);});
						angL([&](int ia){fxhv[ia+nang] = 2.*pc[ia] - psij(ia,i,k,g);});
						if(params.ndimen == 3)angL([&](int ia){fxhv[ia+nang*2] = 2.*pc[ia] - psik(ia,i,j,g);});
						if(vdelt(g) != 0.0) angL([&](int ia){fxhv[ia+nang*3] = 2.*pc[ia] - ptr_in(ia,i,j,k,ioct,g);});

						for(int l=0;l<4;l++)
							angL([&](int ia){if(fxhv[ia+nang*l] < 0.0) hv[ia+nang*l] = 0;});


						double sum_hv2 = 0.0;

						for(int l=0;l<4;l++)
							angL([&](int ia){sum_hv2 += hv[ia+nang*l];});

						if(sum_hv == sum_hv2) break;

						sum_hv = sum_hv2;

						angL([&](int ia){pc[ia] = psii(ia,j,k,g)*mu(ia)*hi*(1.0+hv[ia])
												+ psij(ia,i,k,g)*hj(ia)*(1.0+hv[ia+nang])
												+ psik(ia,i,j,g)*hk(ia)*(1.0+hv[ia+2*nang]);});
						if(vdelt(g) != 0.0)
							angL([&](int ia){pc[ia] += vdelt(g)*ptr_in(ia,i,j,k,ioct,g)*(1.0+hv[ia+nang*3]);});

						angL([&](int ia){pc[ia] = psi[ia] + 0.5*pc[ia];});

						angL([&](int ia){den[ia] = t_xs(i,j,k,g) + mu(ia)*hi*hv[ia]
						                 + hj(ia)*hv[ia+nang]
						                 + hk(ia)*hv[ia+2*nang]
						                 + vdelt(g)*hv[ia+3*nang];});

						angL([&](int ia){if(den[ia] > 1.0e-1) pc[ia] /= den[ia]; else pc[ia] = 0.0;});


						iprotect++;
					}


					angL([&](int ia){psi[ia] = pc[ia];});
					angL([&](int ia){psii(ia,j,k,g) = fxhv[ia]*hv[ia];});
					angL([&](int ia){psij(ia,i,k,g) = fxhv[ia+nang]*hv[ia+nang];});
					if(params.ndimen == 3)
						angL([&](int ia){psik(ia,i,j,g) = fxhv[ia+2*nang]*hv[ia+2*nang];});
					if(vdelt(g) != 0)
						angL([&](int ia){ptr_out(ia,i,j,k,ioct,g) = fxhv[ia+3*nang]*hv[ia+3*nang];});

				}// else fixup

				// Clear the flux arrays
//				if(ioct == 0)
//					for(int l=0;l<params.cmom;l++)
//						flux(l,i,j,k,g) = 0.0;

				// Calculate min and max scalr fluxes
				// TODO


//				// Save edge fluxes
//				if(j == jhi)
//					if(jd == 1 && lasty)
//					{
//
//					}
//					else if(jd == 0 && firsty)
//						angL([&](int ia){jb_out(ia,i,k,g) = psij(ia,i,k,g).cast();});
//
//				if(j == jhi)
//					if(jd == 1 && lasty)
//					{
//
//					}
//					else if(jd == 0 && firsty)
//						angL([&](int ia){jb_out(ia,i,k,g) = psij(ia,i,k,g).cast();});
//
//
//				if(k == khi)
//					if(kd == 1 && lastz)
//					{
//
//					}
//					else if(kd == 0 && firstz)
//						angL([&](int ia){kb_out(ia,i,k,g) = psik(ia,i,k,g).cast();});


			}//	if(i < nx)




		}
		}

	}
	hi = f;

}




template<int i>__attribute__ ((noinline))
		MEKLInterface* genSweep(int igen,Context ctx, HighLevelRuntime* runtime,
                           SweepBase* base,
                           LegionMatrix<LRWrapper> lr_angFlux,
                           LegionMatrix<LRWrapper> lr_bci,
                           LegionMatrix<LRWrapper> lr_bcj,
                           LegionMatrix<LRWrapper> lr_bck,
                           LPWrapper lp_diag, // Copied Partitioning
                           LPWrapper lp_lines, // Copied Partitioning
                           int _id, int _jd, int _kd,
						   int iOct)
{
	if(igen == i)
	{
		Sweep<i>* sweep0 = new Sweep<i>(base);

		return &genMustEpochKernel(*sweep0,ctx,runtime,base,lr_angFlux,lr_bci,lr_bcj,lr_bck,lp_diag,lp_lines,_id,_jd,_kd);
	}
	else
		return genSweep<i-1>(igen,ctx,runtime,base,lr_angFlux,lr_bci,lr_bcj,lr_bck,lp_diag,lp_lines,_id,_jd,_kd,iOct);
}

template<>__attribute__ ((noinline))
		MEKLInterface* genSweep<0>(int igen,Context ctx, HighLevelRuntime* runtime,
		                              SweepBase* base,
		                              LegionMatrix<LRWrapper> lr_angFlux,
		                              LegionMatrix<LRWrapper> lr_bci,
		                              LegionMatrix<LRWrapper> lr_bcj,
		                              LegionMatrix<LRWrapper> lr_bck,
		                              LPWrapper lp_diag, // Copied Partitioning
		                              LPWrapper lp_lines, // Copied Partitioning
		                              int _id, int _jd, int _kd,
		                              int iOct)
{
	Sweep<0>* sweep0 = new Sweep<0>(base);

	return &genMustEpochKernel(*sweep0,ctx,runtime,base,lr_angFlux,lr_bci,lr_bcj,lr_bck,lp_diag,lp_lines,_id,_jd,_kd);
}

std::vector<MEKLInterface*> genSweeps(Context ctx, HighLevelRuntime* runtime,
                                     	SlappyParams params,
                                     	LegionMatrix<LRWrapper> lr_scalFlux,
                                     	LegionMatrix<LRWrapper> lr_sigmaT,
                                        LegionMatrix<LRWrapper> lr_angFlux,
                                        LegionMatrix<LRWrapper> lr_bci,
                                        LegionMatrix<LRWrapper> lr_bcj,
                                        LegionMatrix<LRWrapper> lr_bck,
                                        LRWrapper dogrp_i,
                                        LPWrapper lp_diag, // Copied Partitioning
                                        LPWrapper lp_lines // Copied Partitioning
                                        )
{
	std::vector<MEKLInterface*> res;

	std::vector<SweepBase*> chunk_sweeps;
	for(int kC=0;kC<params.nCz;kC++)
		for(int jC=0;jC<params.nCy;jC++)
			for(int iC=0;iC<params.nCx;iC++)
				chunk_sweeps.push_back(new SweepBase(ctx,runtime,params,lr_scalFlux,lr_sigmaT,lr_bci,lr_bcj,lr_bck,dogrp_i,iC,jC,kC));

	for(int kd=0;kd<2;kd++)
		for(int jd=0;jd<2;jd++)
			for(int id=0;id<2;id++)
			{
				int nCx(params.nCx), nCy(params.nCy), nCz(params.nCz);
				int nx_l(params.nx_l), ny_l(params.ny_l), nz_l(params.nz_l);
			//	lw_proc_map.create(ctx,runtime,{nCx*nCy*nCz},0,int(),1,int(),2,int());


				int iproc_hi,iproc_lo;
				int jproc_hi,jproc_lo;
				int kproc_hi,kproc_lo;
				int inc,jnc,knc;


				if(id == 0)
				{iproc_hi=-1;iproc_lo=nCx-1; inc=-1;}
				else
				{iproc_lo=0;iproc_hi=nCx; inc=1;}

				if(jd == 0)
				{jproc_hi=-1;jproc_lo=nCy-1; jnc=-1;}
				else
				{jproc_lo=0;jproc_hi=nCy; jnc=1;}

				if(kd == 0)
				{kproc_hi=-1;kproc_lo=nCz-1; knc=-1;}
				else
				{kproc_lo=0;kproc_hi=nCz; knc=1;}

				for(int kC=kproc_lo;kC!=kproc_hi;kC+=knc) // Loop over k processors
					for(int jC=jproc_lo;jC!=jproc_hi;jC+=jnc) // Loop over j processors
						for(int iC=iproc_lo;iC!=iproc_hi;iC+=inc) // Loop over i processors
							res.push_back(genSweep<7>(id+2*jd+4*kd,ctx,runtime,chunk_sweeps[iC+nCx*(jC+nCy*kC)],
							                               lr_angFlux,lr_bci,lr_bcj,lr_bck,lp_diag,lp_lines,
							                               id,jd,kd,id+2*jd+4*kd));
			}
	return res;
}

void register_sweep_tasks()
{
	MustEpochKernelLauncher<Sweep<0>>::register_cpu();
	MustEpochKernelLauncher<Sweep<1>>::register_cpu();
	MustEpochKernelLauncher<Sweep<2>>::register_cpu();
	MustEpochKernelLauncher<Sweep<3>>::register_cpu();
	MustEpochKernelLauncher<Sweep<4>>::register_cpu();
	MustEpochKernelLauncher<Sweep<5>>::register_cpu();
	MustEpochKernelLauncher<Sweep<6>>::register_cpu();
	MustEpochKernelLauncher<Sweep<7>>::register_cpu();

//	MustEpochKernelLauncher<Sweep<8>>::register_cpu();
//	MustEpochKernelLauncher<Sweep<9>>::register_cpu();
//	MustEpochKernelLauncher<Sweep<10>>::register_cpu();
//	MustEpochKernelLauncher<Sweep<11>>::register_cpu();
//	MustEpochKernelLauncher<Sweep<12>>::register_cpu();
//	MustEpochKernelLauncher<Sweep<13>>::register_cpu();
//	MustEpochKernelLauncher<Sweep<14>>::register_cpu();
//	MustEpochKernelLauncher<Sweep<15>>::register_cpu();


}
























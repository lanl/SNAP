/*
 * Sweep.h
 *
 *  Created on: Jun 16, 2015
 *      Author: payne
 */

#ifndef SWEEP_H_
#define SWEEP_H_

#include <stdio.h>
#include <legion.h>
#include <tuple>
#include <stdlib.h>
#include <unistd.h>
#include "SlappyParams.h"
#include <IndexKernelLauncher.h>
#include <LegionMatrix.h>


using namespace Dragon;

class SweepBase
{
public:
	SlappyParams params;

	LPWrapper* lp_scalFlux;
	LPWrapper* lp_sigmaT;
	LPWrapper* lp_bci_out;
	LPWrapper* lp_bcj_out;
	LPWrapper* lp_bck_out;

	LPWrapper* lp_dogrp;

	int iproc, jproc, kproc;

	int n_split;

public: // Required members
	static const int SINGLE = false;
	static const int INDEX = true;
	static const int MAPPER_ID = 0;
public:


	SweepBase(Context ctx, HighLevelRuntime* runtime,
                 SlappyParams _params,
                 LegionMatrix<LRWrapper> lr_scalFlux,
                 LegionMatrix<LRWrapper> lr_sigmaT,
                 LegionMatrix<LRWrapper> lr_bci,
                 LegionMatrix<LRWrapper> lr_bcj,
                 LegionMatrix<LRWrapper> lr_bck,
                 LRWrapper lr_dogrp,
   			   int _iproc, int _jproc, int _kproc);

	~SweepBase(){}
};

template<int iOct>
class Sweep
{
	SlappyParams params;



	int iproc, jproc, kproc;
	int ilo,jlo,klo;
	int ihi,jhi,khi;
	int id,jd,kd;
	int ioct;
	int ist,jst,kst;
	int firsty,lasty,firstz,lastz;
	int firstx,lastx;
	int n_split;

public: // Required members
	static const int SINGLE = true;
	static const int INDEX = false;
	static const int MAPPER_ID = 0;
public:
	enum {FID_DiagLen,FID_DiagStart};
	enum {FID_ic,FID_j,FID_k};

	Sweep(SweepBase* _base):params(_base->params),nang(params.nang),dm(1.0/((double)params.nang)),hi(2.0/params.dx){};

	void on_start(void);

	void on_finish(void);

	template<class T> inline
	void angL(T op)
	{
		for(int ia=0;ia<params.nang;ia++)
			op(ia);
	}
	double dm;
	int nang;
	double vdelt(int _g){return 2.0/(params.dt*(params.ng-_g));};

	double mu(int _iang){return (_iang+0.5)*dm;};
	double eta(int _iang){return 1.0 - 0.5*dm - _iang*dm;};
	double xi(int _iang)
			{
				double mut = mu(_iang);
				double etat = eta(_iang);
				double t = mut*mut+etat*etat;
				return sqrt(1.0-t);
			};
	double hi;
	double hj(int _iang){return 2.0/params.dy * eta(_iang);};
	double hk(int _iang){return 2.0/params.dz * xi(_iang);};


	double ec(int _iang,int _l, int _m){return std::pow(ist*mu(_iang),2*_l+1)*std::pow(kst*xi(_iang)*jst*eta(_iang),1*_m);};

	EpochKernelArgs genArgs(Context ctx, HighLevelRuntime* runtime,
	                        SweepBase* base,
                            LegionMatrix<LRWrapper> lr_angFlux,
             			   LegionMatrix<LRWrapper> lr_bci,
             			   LegionMatrix<LRWrapper> lr_bcj,
             			   LegionMatrix<LRWrapper> lr_bck,
              			   LPWrapper lp_diag, // Copied Partitioning
              			   LPWrapper lp_lines, // Copied Partitioning
             			   int _id, int _jd, int _kd);


	__host__
	static void evaluate_s(int iDom,LegionMatrix<bool> dogrp,
                           LegionMatrix<int> diag_len,
                           LegionMatrix<int> diag_start,
                           LegionMatrix<int> i,
                           LegionMatrix<int> j,
                           LegionMatrix<int> k,
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
						    LegionMatrix<double> bck_in){};

	__host__
	void evaluate(int iDom,LegionMatrix<bool> dogrp,
                  LegionMatrix<int> diag_len,
                  LegionMatrix<int> diag_start,
                  LegionMatrix<int> i,
                  LegionMatrix<int> j,
                  LegionMatrix<int> k,
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
						    LegionMatrix<double> bck_in);
};

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
                                        );

void register_sweep_tasks();

#endif /* SWEEP_H_ */

/*
 * Dim3Sweep.h
 *
 *  Created on: Jun 16, 2015
 *      Author: payne
 */

#ifndef DIM3SWEEP_H_
#define DIM3SWEEP_H_
#include <stdio.h>
#include <legion.h>
#include <tuple>
#include <stdlib.h>
#include "Utils.h"
#include "FieldKernelOp.h"
#include <unistd.h>
#include "SlappyParams.h"
#include "Partitioners.h"



template<int IT>
class Dim3Sweep
{
public:
	TaskArgs targs;

	SlappyParams params;
	int igrp;

	LRWrapper lw_psii;
	LRWrapper lw_psij;
	LRWrapper lw_psik;

	LRWrapper lw_proc_map;


	int ilo,jlo,klo;
	int ihi,jhi,khi;
	int id,jd,kd;
	int ist,jst,kst;
	int firsty,lasty,firstz,lastz;
	int firstx,lastx;
	double hi;



	Coloring cl_psii_t;
	Coloring cl_psij_t;
	Coloring cl_psik_t;
	Coloring cl_chunkArrays_t;

	std::map<Color,std::vector<RegionRequirement>> rreq_tmp;

   CopiedPart lp_diag;
   CopiedPart lp_lines;


public: // Required members
	static const int SINGLE = true;
	static const int INDEX = false;
	static const int MAPPER_ID = 0;
public:


	Dim3Sweep(Context ctx, HighLevelRuntime* runtime,
              SlappyParams _params,
			   LRWrapper _lw_chunkArrays,
			   CopiedPart _lw_diag,
			   CopiedPart _lw_lines,
			   int _igrp);

	OpArgs genArgs(Context ctx, HighLevelRuntime* runtime,
				   LRWrapper _lw_chunkArrays,
				   int _id, int _jd, int _kd);


	__host__
	static void evaluate_s(int iDom,lrAccessor<LRWrapper> lr_angFlux,
	                         lrAccessor<LRWrapper> lr_scalFlux,
	                         lrAccessor<LRWrapper> lr_sigmaT,
	                         lrAccessor<LRWrapper> bci,
	                         lrAccessor<LRWrapper> bcj,
	                         lrAccessor<LRWrapper> bck){};

	__host__
	void evaluate(int iDom,lrAccessor<LRWrapper> lr_angFlux,
                  lrAccessor<LRWrapper> lr_scalFlux,
                  lrAccessor<LRWrapper> lr_sigmaT,
                  lrAccessor<LRWrapper> bci,
                  lrAccessor<LRWrapper> bcj,
                  lrAccessor<LRWrapper> bck);


};

std::vector<FieldKernelLauncher> genOctSweeps(Context ctx, HighLevelRuntime* runtime,
		   SlappyParams _params,
		   LRWrapper _lw_chunkArrays,
		   CopiedPart _lw_diag,
		   CopiedPart _lw_lines,
		   int _igrp);

double genEvaluates(int iGen,Context ctx, HighLevelRuntime* runtime,
        SlappyParams _params,
		   LRWrapper _lw_chunkArrays,
		   CopiedPart _lw_diag,
		   CopiedPart _lw_lines,
		   int _igrp,int iDom,lrAccessor<LRWrapper> lr_angFlux,
                         lrAccessor<LRWrapper> lr_scalFlux,
                         lrAccessor<LRWrapper> lr_sigmaT,
                         lrAccessor<LRWrapper> bci,
                         lrAccessor<LRWrapper> bcj,
                         lrAccessor<LRWrapper> bck);

void RegisterDim3Variants();



#endif /* DIM3SWEEP_H_ */

/*
 * InGroupSource.h
 *
 *  Created on: Jun 16, 2015
 *      Author: payne
 */

#ifndef INGROUPSOURCE_H_
#define INGROUPSOURCE_H_

#include <stdio.h>
#include <legion.h>
#include <tuple>
#include <stdlib.h>
#include <unistd.h>
#include "SlappyParams.h"
#include <MustEpochKernelLauncher.h>


class SourceFIDs
{
	enum
	{
		q2grp,qtot,LF
	};
};

//class InGroupSource
//{
//public:
//	TaskArgs targs;
//
//	SlappyParams params;
//	LRWrapper lw_allArrays;
//public: // Required members
//	static const int SINGLE = false;
//	static const int INDEX = true;
//	static const int MAPPER_ID = 0;
//public:
//
//	InGroupSource(SlappyParams _params) : params(_params){}
//
//	OpArgs genArgs(Context ctx, HighLevelRuntime* runtime,
//                   LRWrapper _lw_allArrays,
//    			   LRWrapper _lw_chunkArrays,
//    			   LRWrapper _lw_slgg,
//    			   LRWrapper _lw_diag,
//    			   LRWrapper _lw_lines,
//    			   LRWrapper _lw_dogrp);
//
//
//	__host__
//	static void evaluate_s(int iglbl,lrAccessor<LRWrapper> lr_chunkArrays,
//	               			  lrAccessor<LRWrapper> lr_slgg,
//	               			  lrAccessor<LRWrapper> lr_diag,
//	               			  lrAccessor<LRWrapper> lr_lines,
//	               			  lrAccessor<bool> lr_dogrp,
//	               			  lrAccessor<LRWrapper> lr_angFluxes,
//	               			  lrAccessor<LRWrapper> lr_scalFluxes,
//	               			  lrAccessor<LRWrapper> lr_savedFluxes,
//	               			  lrAccessor<LRWrapper> lr_sigmaT,
//	               			  lrAccessor<LRWrapper> lr_sigmaS,
//	               			  lrAccessor<LRWrapper> lr_mat,
//	               			  lrAccessor<LRWrapper> lr_bci,
//	               			  lrAccessor<LRWrapper> lr_bcj,
//	               			  lrAccessor<LRWrapper> lr_bck){}
//
//	__host__
//	void evaluate(int iglbl,lrAccessor<LRWrapper> lr_chunkArrays,
//       			  lrAccessor<LRWrapper> lr_slgg,
//       			  lrAccessor<LRWrapper> lr_diag,
//       			  lrAccessor<LRWrapper> lr_lines,
//       			  lrAccessor<bool> lr_dogrp,
//       			  lrAccessor<LRWrapper> lr_angFluxes,
//       			  lrAccessor<LRWrapper> lr_scalFluxes,
//       			  lrAccessor<LRWrapper> lr_savedFluxes,
//       			  lrAccessor<LRWrapper> lr_sigmaT,
//       			  lrAccessor<LRWrapper> lr_sigmaS,
//       			  lrAccessor<LRWrapper> lr_mat,
//       			  lrAccessor<LRWrapper> lr_bci,
//       			  lrAccessor<LRWrapper> lr_bcj,
//       			  lrAccessor<LRWrapper> lr_bck);
//};

class InnerSrcScat
{
public:


	SlappyParams params;


public: // Required members
	static const int SINGLE = true;
	static const int INDEX = false;
	static const int MAPPER_ID = 0;
public:

	LPWrapper lp_chunkArrays;
	LPWrapper lp_dogrp;

	InnerSrcScat(SlappyParams _params) : params(_params) {}

	EpochKernelArgs genArgs(Context ctx, HighLevelRuntime* runtime,
	               LRWrapper _lw_chunkArrays,
	               LRWrapper _lw_dogrp);


	__host__
	static void evaluate_s(const Task* task, Context ctx, HighLevelRuntime* rt,
	                       LegionMatrix<bool> _dgrp,
	                       LegionMatrix<LRWrapper> _s_xs,
	                       LegionMatrix<LRWrapper> _scalFlux){};

	__host__
	void evaluate(const Task* task, Context ctx, HighLevelRuntime* rt,
                  LegionMatrix<bool> _dgrp,
                  LegionMatrix<LRWrapper> _s_xs,
                  LegionMatrix<LRWrapper> _scalFlux);

	static void register_cpu();


};


#endif /* INGROUPSOURCE_H_ */

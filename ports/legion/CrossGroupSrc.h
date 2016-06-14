/*
 * CrossGroupSrc.h
 *
 *  Created on: Jun 16, 2015
 *      Author: payne
 */

#ifndef CROSSGROUPSRC_H_
#define CROSSGROUPSRC_H_

#include <stdio.h>
#include <legion.h>
#include <tuple>
#include <stdlib.h>
#include <SingleKernelLauncher.h>
#include <unistd.h>
#include "SlappyParams.h"


/*
 * This class is analogous to snap's "outer" iteration
 */
using namespace Dragon;
class CrossGroupSrc
{
public:
	SlappyParams params;
	LRWrapper lw_allArrays;





public: // Required members
	static const int SINGLE = true;
	static const int INDEX = false;
	static const int MAPPER_ID = 0;
public:

	CrossGroupSrc(SlappyParams _params) : params(_params) {}

	SingleKernelArgs genArgs(Context ctx, HighLevelRuntime* runtime,
	               LRWrapper _lw_allArrays,
				   LRWrapper _lw_chunkArrays,
				   LRWrapper _lw_slgg,
				   LRWrapper _lw_diag,
				   LRWrapper _lw_lines);


	__host__
	static void evaluate_s(const Task* task, Context ctx, HighLevelRuntime* rt,
	                       LegionMatrix<LRWrapper> lr_chunkArrays,
	         			  LegionMatrix<LRWrapper> lr_slgg,
	         			  LegionMatrix<LRWrapper> lr_diag, LegionMatrix<LRWrapper> lr_lines){};

	__host__
	void evaluate(const Task* task, Context ctx, HighLevelRuntime* rt,
	              LegionMatrix<LRWrapper> lr_chunkArrays,
				  LegionMatrix<LRWrapper> lr_slgg,
				  LegionMatrix<LRWrapper> lr_diag, LegionMatrix<LRWrapper> lr_lines);

	int any_left(Context ctx,HighLevelRuntime* rt,LRWrapper _dogrp,LRWrapper _df);

	static void register_cpu();

};

class OuterSrcScat
{
public:


	SlappyParams params;
	LRWrapper lw_slgg;
	unsigned iDomain;

public: // Required members
	static const int SINGLE = true;
	static const int INDEX = false;
	static const int MAPPER_ID = 0;
public:

	OuterSrcScat(SlappyParams _params) : params(_params) {}

	EpochKernelArgs genArgs(Context ctx, HighLevelRuntime* runtime,
	               LRWrapper _lw_chunkArrays,
	               LRWrapper _lw_slgg,
	               LRWrapper _lw_dogrp);


	__host__
	static void evaluate_s(const Task* task, Context ctx, HighLevelRuntime* rt,
	   					LegionMatrix<bool> _dgrp,
	   					LegionMatrix<LRWrapper> _qi,
	   					LegionMatrix<LRWrapper> _scalFlux,
	   					LegionMatrix<double> slgg,
	   					LegionMatrix<LRWrapper> _mat){};

	__host__
	void evaluate(const Task* task, Context ctx, HighLevelRuntime* rt,
					LegionMatrix<bool> _dgrp,
					LegionMatrix<LRWrapper> _qi,
					LegionMatrix<LRWrapper> _scalFlux,
					LegionMatrix<double> slgg,
					LegionMatrix<LRWrapper> _mat);

	static void register_cpu();


};

class OuterConv
{
public:


	SlappyParams params;
	LRWrapper lw_df;
	FieldID save_fid;

public: // Required members
	static const int SINGLE = true;
	static const int INDEX = false;
	static const int MAPPER_ID = 0;
public:

	OuterConv(SlappyParams _params) : params(_params) {}

	EpochKernelArgs genArgs(Context ctx, HighLevelRuntime* runtime,
                   LRWrapper chunkArrays,LRWrapper _dogrp,LRWrapper _df,
                   FieldID _save_fid);


	__host__
	static void evaluate_s(const Task* task, Context ctx, HighLevelRuntime* rt,
	                       LegionMatrix<LRWrapper> _flux,
	                       LegionMatrix<LRWrapper> _fluxpo,
	                       LegionMatrix<bool> _dogrp,
	                       LegionMatrix<double> _df){};

	__host__
	void evaluate(const Task* task, Context ctx, HighLevelRuntime* rt,
	              LegionMatrix<LRWrapper> _flux,
	              LegionMatrix<LRWrapper> _fluxpo,
	              LegionMatrix<bool> _dogrp,
	              LegionMatrix<double> _df);

	static void register_cpu();


};

class SaveFluxes
{
public:


	SlappyParams params;
	FieldID save_fid;

public: // Required members
	static const int SINGLE = true;
	static const int INDEX = false;
	static const int MAPPER_ID = 0;
public:

	SaveFluxes(SlappyParams _params) : params(_params){}

	EpochKernelArgs genArgs(Context ctx, HighLevelRuntime* runtime,
                   LRWrapper chunkArrays,LRWrapper _dogrp,
                   FieldID _save_fid);


	__host__
	static void evaluate_s(const Task* task, Context ctx, HighLevelRuntime* rt,
	                       LegionMatrix<LRWrapper> _flux,
	                       LegionMatrix<LRWrapper> _fluxpo,
	                       LegionMatrix<bool> _dogrp){};

	__host__
	void evaluate(const Task* task, Context ctx, HighLevelRuntime* rt,
	              LegionMatrix<LRWrapper> _flux,
	              LegionMatrix<LRWrapper> _fluxpo,
	              LegionMatrix<bool> _dogrp);

	static void register_cpu();


};


#endif /* CROSSGROUPSRC_H_ */

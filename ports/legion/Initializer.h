/*
 * Initializer.h
 *
 *  Created on: Jun 21, 2015
 *      Author: payne
 */

#ifndef INITIALIZER_H_
#define INITIALIZER_H_
#include <stdio.h>
#include <legion.h>
#include <tuple>
#include <stdlib.h>
#include <LegionMatrix.h>
#include <LRWrapper.h>
#include "SlappyParams.h"
#include <unistd.h>
#include <MustEpochKernelLauncher.h>


using namespace Dragon;

class Initializer
{
public:

	SlappyParams params;

public: // Required members
	static const int SINGLE = true;
	static const int INDEX = false;
	static const int MAPPER_ID = 0;
public:

	Initializer(SlappyParams _params) : params(_params) {}
	EpochKernelArgs genArgs(Context ctx, HighLevelRuntime* runtime,LRWrapper chunkArrays,LRWrapper slgg);


	__host__
	static void evaluate_s(const Task* task,Context ctx,HighLevelRuntime* rt,
	         			  LegionMatrix<LRWrapper> lw_scalFluxes,
	         			  LegionMatrix<LRWrapper> lw_sigmaT,
	         			  LegionMatrix<LRWrapper> lw_sigmaS,
	         			  LegionMatrix<LRWrapper> lw_mat,
	         			  LegionMatrix<double> la_slgg){};

	__host__
	void evaluate(const Task* task,Context ctx,HighLevelRuntime* rt,
				  LegionMatrix<LRWrapper> lw_scalFluxes,
				  LegionMatrix<LRWrapper> lw_sigmaT,
				  LegionMatrix<LRWrapper> lw_sigmaS,
				  LegionMatrix<LRWrapper> lw_mat,
				  LegionMatrix<double> la_slgg);

	static void register_task();
};




#endif /* INITIALIZER_H_ */

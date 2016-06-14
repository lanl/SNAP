/*
 * Sweep3D.h
 *
 *  Created on: Aug 13, 2015
 *      Author: payne
 */

#ifndef SWEEP3D_H_
#define SWEEP3D_H_

#include <stdio.h>
#include <legion.h>
#include <tuple>
#include <stdlib.h>
#include <unistd.h>
#include "SlappyParams.h"
#include <IndexKernelLauncher.h>
#include <LegionMatrix.h>


using namespace Dragon;

class Sweep3D
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
	double hi;

public: // Required members
	static const int SINGLE = false;
	static const int INDEX = false;
	static const int MAPPER_ID = 0;
public:
	enum {FID_DiagLen,FID_DiagStart};
	enum {FID_ic,FID_j,FID_k};

	Sweep(SweepBase _base):params(_base.params){};

	IndexKernelArgs genArgs(Context ctx, HighLevelRuntime* runtime,
	                        SweepBase base,
                            LegionMatrix<LRWrapper> lr_angFlux,
             			   LegionMatrix<LRWrapper> lr_bci,
             			   LegionMatrix<LRWrapper> lr_bcj,
             			   LegionMatrix<LRWrapper> lr_bck,
              			   LPWrapper lp_diag, // Copied Partitioning
              			   LPWrapper lp_lines, // Copied Partitioning
             			   int _id, int _jd, int _kd);


	__host__
	static void evaluate_s(int iDom,LegionMatrix<bool> dogrp){}//,
//                           LegionMatrix<int> diag_len,
//                           LegionMatrix<int> diag_start,
//                           LegionMatrix<int> i,
//                           LegionMatrix<int> j,
//                           LegionMatrix<int> k,
//                            LegionMatrix<double> ptr_in,
//                            LegionMatrix<double> ptr_out,
//                            LegionMatrix<double> qtot,
//                            LegionMatrix<double> flux,
//                            LegionMatrix<double> t_xs,
//                            LegionMatrix<double> bci_out,
//						    LegionMatrix<double> bcj_out,
//						    LegionMatrix<double> bck_out,
//                            LegionMatrix<double> bci_in,
//						    LegionMatrix<double> bcj_in,
//						    LegionMatrix<double> bck_in){};

	__host__
	void evaluate(int iDom,LegionMatrix<bool> dogrp);//,
//                  LegionMatrix<int> diag_len,
//                  LegionMatrix<int> diag_start,
//                  LegionMatrix<int> i,
//                  LegionMatrix<int> j,
//                  LegionMatrix<int> k,
//                            LegionMatrix<double> ptr_in,
//                            LegionMatrix<double> ptr_out,
//                            LegionMatrix<double> qtot,
//                            LegionMatrix<double> flux,
//                            LegionMatrix<double> t_xs,
//                            LegionMatrix<double> bci_out,
//						    LegionMatrix<double> bcj_out,
//						    LegionMatrix<double> bck_out,
//                            LegionMatrix<double> bci_in,
//						    LegionMatrix<double> bcj_in,
//						    LegionMatrix<double> bck_in);
};

#endif /* SWEEP3D_H_ */

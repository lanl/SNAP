/*
 * Dim3Sweep.cc
 *
 *  Created on: Jun 16, 2015
 *      Author: payne
 */

#include "Dim3Sweep.h"
#include <math.h>
template<int IT>__attribute__ ((noinline))
Dim3Sweep<IT>::Dim3Sweep(Context ctx, HighLevelRuntime* runtime,
                     SlappyParams _params,
      			   LRWrapper _lw_chunkArrays,
      			   CopiedPart _lw_diag,
      			   CopiedPart _lw_lines,
      			   int _igrp) : params(_params), igrp(_igrp), lp_diag(_lw_diag),lp_lines(_lw_lines)
{
	int nCx(params.nCx), nCy(params.nCy), nCz(params.nCz);
	int nx_l(params.nx_l), ny_l(params.ny_l), nz_l(params.nz_l);
	int nang(params.nang);
	using namespace FIDs;
//	lw_psii.create(ctx,runtime,{nang,ny_l,nz_l,params.ng,nCx,nCy,nCz},0,double());
//	lw_psij.create(ctx,runtime,{nang,nx_l,nz_l,params.ng,nCx,nCy,nCz},0,double());
//	lw_psik.create(ctx,runtime,{nang,nx_l,ny_l,params.ng,nCx,nCy,nCz},0,double());

//	LegionMatrix<LRWrapper> scalFluxes(ctx,_lw_chunkArrays,FIDs::Array::scalFluxes);
//	LegionMatrix<LRWrapper> sigmaT(ctx,_lw_chunkArrays,FIDs::Array::sigmaT);
//	LegionMatrix<LRWrapper> bci(ctx,_lw_chunkArrays,FIDs::Array::bci);
//	LegionMatrix<LRWrapper> bcj(ctx,_lw_chunkArrays,FIDs::Array::bcj);
//	LegionMatrix<LRWrapper> bck(ctx,_lw_chunkArrays,FIDs::Array::bck);
//
//	std::vector<LegionMatrix<LRWrapper>> wrapper_vec;
//	wrapper_vec.push_back(scalFluxes);
//	wrapper_vec.push_back(sigmaT);
//	wrapper_vec.push_back(bci);
//	wrapper_vec.push_back(bcj);
//	wrapper_vec.push_back(bck);
//
//	std::map<FieldID,int> wrapper_map;
//	wrapper_map[Array::scalFluxes] = 0;
//	wrapper_map[Array::sigmaT] = 1;
//	wrapper_map[Array::bci] = 2;
//	wrapper_map[Array::bcj] = 3;
//	wrapper_map[Array::bck] = 4;
//
////	typename std::map<PrivilegeMode,std::vector<FieldID>> PrivPair;
//	std::map<FieldID,std::map<PrivilegeMode,std::vector<FieldID>>> priv_field_map;
//
//
//	typedef std::pair<FieldID,FieldID> FieldPair;
//	std::map<FieldPair,PrivilegeMode> child_privs;
//
//	child_privs[FieldPair(Array::scalFluxes,ScalFlux::qtot)] = READ_ONLY;
//	child_privs[FieldPair(Array::scalFluxes,ScalFlux::flux)] = READ_WRITE;
//
//	child_privs[FieldPair(Array::sigmaT,SigmaT::t_xs)] = READ_ONLY;
//
//	child_privs[FieldPair(Array::bci,0)] = READ_WRITE;
//	child_privs[FieldPair(Array::bcj,0)] = READ_WRITE;
//	child_privs[FieldPair(Array::bck,0)] = READ_WRITE;
//
//	for(auto field : wrapper_map)
//	{
//		std::vector<FieldID> ro_fids;
//		std::vector<FieldID> rw_fids;
//		for(auto cpriv : child_privs)
//		{
//			if(cpriv.first.first == field.first)
//			{
//				if(cpriv.second == READ_ONLY)
//					ro_fids.push_back(cpriv.first.second);
//				else if(cpriv.second == READ_WRITE)
//					rw_fids.push_back(cpriv.first.second);
//
//			}
//		}
//		priv_field_map[field.first][READ_ONLY] = ro_fids;
//		priv_field_map[field.first][READ_WRITE] = rw_fids;
//
//
//	}
//
//
//
//
////	int iproc_hi,iproc_lo;
////	int jproc_hi,jproc_lo;
////	int kproc_hi,kproc_lo;
////	int inc,jnc,knc;
////
////	int firsty,lasty,firstz,lastz;
////	int firstx,lastx;
////	if(id == 0)
////	{iproc_hi=-1;iproc_lo=nCy-1; inc=-1;}
////	else
////	{iproc_lo=0;iproc_hi=nCy; inc=1;}
////
////	if(jd == 0)
////	{jproc_hi=-1;jproc_lo=nCy-1; jnc=-1;}
////	else
////	{jproc_lo=0;jproc_hi=nCy; jnc=1;}
////
////	if(kd == 0)
////	{kproc_hi=-1;kproc_lo=nCz-1; knc=-1;}
////	else
////	{kproc_lo=0;kproc_hi=nCz; knc=1;}
//
//
//
//
//	auto append = [](ColoredPoints<ptr_t> &point,ColoredPoints<ptr_t> new_points)
//				{
//
//					if(point.points.size() == 0 && point.ranges.size() == 0)
//						point = new_points;
//					if(new_points.ranges.size() > 0)
//						point.ranges.insert(new_points.ranges.begin(),new_points.ranges.end());
//					if(new_points.points.size() > 0)
//							point.points.insert(new_points.points.begin(),new_points.points.end());
//
//				};
//
//	for(int kproc=0;kproc<nCz;kproc+=1) // Loop over k processors
//		for(int jproc=0;jproc<nCy;jproc+=1) // Loop over j processors
//			for(int iproc=0;iproc<nCx;iproc+=1) // Loop over i processors
//			{
//				int iDom = iproc+nCx*(jproc+nCy*kproc);
//
//				int iC = iproc;
//				int jC = jproc;
//				int kC = kproc;
//
////				if(iproc == 0)
////					firstx = true;
////				else
////					firstx = false;
////
////				if(iproc == nCx-1)
////					lastx = true;
////				else
////					lastx = false;
////
////
////				if(jproc == 0)
////					firsty = true;
////				else
////					firsty = false;
////
////				if(jproc == nCy-1)
////					lasty = true;
////				else
////					lasty = false;
////
////				if(kproc == 0)
////					firstz = true;
////				else
////					firstz = false;
////
////				if(kproc == nCz-1)
////					lastz = true;
////				else
////					lastz = false;
////				int xlop,xhip;
////				int ylop,yhip;
////				int zlop,zhip;
////				int xp_snd,xp_rcv;
////				int yp_snd,yp_rcv;
////				int zp_snd,zp_rcv;
////
////				// Processor setup
////				if(jproc > 0)
////					ylop = jproc-1;
////				else
////					ylop = jproc;
////
////				if(jproc < (nCy-1))
////					yhip = jproc+1;
////				else
////					yhip = jproc;
////
////				// Processor setup
////				if(kproc > 0)
////					zlop = kproc-1;
////				else
////					zlop = kproc;
////
////				if(kproc < (nCz-1))
////					zhip = kproc+1;
////				else
////					zhip = kproc;
////
////				if(jd == 0)
////				{jlo=ny_l-1;jhi=0;jst=-1;yp_snd=ylop;yp_rcv=yhip;}
////				else
////				{jlo=0;jhi=ny_l-1;jst=1;yp_snd=yhip;yp_rcv=ylop;}
////
////				if(kd == 0)
////				{klo=nz_l-1;khi=0;kst=-1;zp_snd=zlop;zp_rcv=zhip;}
////				else
////				{klo=0;khi=nz_l-1;kst=1;zp_snd=zhip;zp_rcv=zlop;}
//
//				cl_psii_t[iDom] = lw_psii.GetColoring(":",":",":",iproc,jproc,kproc);
//				cl_psij_t[iDom] = lw_psij.GetColoring(":",":",":",iproc,jproc,kproc);
//				cl_psik_t[iDom] = lw_psik.GetColoring(":",":",":",iproc,jproc,kproc);
//				cl_chunkArrays_t[iDom] = _lw_chunkArrays.GetColoring(iproc,jproc,kproc);
//
////				std::map<FieldID,Coloring> cl_nested;
////
////				cl_nested[Array::scalFluxes][iDom] = scalFluxes(iproc,jproc,kproc).cast().GetColoring(":",":",":",":",igrp);
////				cl_nested[Array::sigmaT][iDom] = sigmaT(iproc,jproc,kproc).cast().GetColoring(":",":",":",igrp);
////				cl_nested[Array::bci][iDom] = bci(iproc,jproc,kproc).cast().GetColoring(":",":",":",igrp);
////				cl_nested[Array::bcj][iDom] = bcj(iproc,jproc,kproc).cast().GetColoring(":",":",":",igrp);
////				cl_nested[Array::bck][iDom] = bck(iproc,jproc,kproc).cast().GetColoring(":",":",":",igrp);
////
////				for(auto coloring : cl_nested)
////				{
////					ColoredPart lp_tmp(ctx,runtime,wrapper_vec[wrapper_map[coloring.first]](iC,jC,kC).cast().lr,coloring.second);
////					LogicalRegion lr_tmp = runtime->get_logical_subregion_by_color(ctx,lp_tmp.lPart,iDom);
////					for(auto privs : priv_field_map[coloring.first])
////					{
////						RegionRequirement rr(lr_tmp,privs.first,EXCLUSIVE,wrapper_vec[wrapper_map[coloring.first]](iC,jC,kC).cast().lr);
////						rr.add_fields(privs.second);
////						rreq_tmp[iDom].push_back(rr);
////					}
////				}
//
//				iDom++;
//			}

}

template<int IT>__attribute__ ((noinline))
OpArgs Dim3Sweep<IT>::genArgs(Context ctx, HighLevelRuntime* runtime,
			   LRWrapper _lw_chunkArrays,
			   int _id, int _jd, int _kd)
{

	using namespace FIDs;

	id = _id;
	jd = _jd;
	kd = _kd;


	std::map<Color,std::vector<RegionRequirement>> rreqs;

//	LegionMatrix<LRWrapper> angFluxes(ctx,_lw_chunkArrays,FIDs::Array::angFluxes);
////	LegionMatrix<LRWrapper> scalFluxes(_lw_chunkArrays,FIDs::Array::scalFluxes);
////	LegionMatrix<LRWrapper> sigmaT(_lw_chunkArrays,FIDs::Array::sigmaT);
//	LegionMatrix<LRWrapper> bci(ctx,_lw_chunkArrays,FIDs::Array::bci);
//	LegionMatrix<LRWrapper> bcj(ctx,_lw_chunkArrays,FIDs::Array::bcj);
//	LegionMatrix<LRWrapper> bck(ctx,_lw_chunkArrays,FIDs::Array::bck);
//
//	std::map<FieldID,FieldID> wrapper_map;
//	wrapper_map[Array::angFluxes] = Array::angFluxes;
////	wrapper_map[Array::scalFluxes] = scalFluxes;
////	wrapper_map[Array::sigmaT] = sigmaT;
////	wrapper_map[Array::bci] = bci;
////	wrapper_map[Array::bcj] = bcj;
////	wrapper_map[Array::bck] = bck;
//
////	typename std::map<PrivilegeMode,std::vector<FieldID>> PrivPair;
//	std::map<FieldID,std::map<PrivilegeMode,std::vector<FieldID>>> priv_field_map;
//
//
//	typedef std::pair<FieldID,FieldID> FieldPair;
//	std::map<FieldPair,PrivilegeMode> child_privs;
//
//	child_privs[FieldPair(Array::angFluxes,AngFlux::ptr_in)] = READ_ONLY;
//	child_privs[FieldPair(Array::angFluxes,AngFlux::ptr_out)] = READ_WRITE;
//
////	child_privs[FieldPair(Array::scalFluxes,ScalFlux::qtot)] = READ_ONLY;
////	child_privs[FieldPair(Array::scalFluxes,ScalFlux::flux)] = READ_WRITE;
////
////	child_privs[FieldPair(Array::sigmaT,SigmaT::t_xs)] = READ_ONLY;
////
////	child_privs[FieldPair(Array::bci,0)] = READ_WRITE;
////	child_privs[FieldPair(Array::bcj,0)] = READ_WRITE;
////	child_privs[FieldPair(Array::bck,0)] = READ_WRITE;
//
//	std::map<FieldPair,PrivilegeMode> rcv_privs;
//	rcv_privs[FieldPair(Array::bci,0)] = READ_ONLY;
//	rcv_privs[FieldPair(Array::bcj,0)] = READ_ONLY;
//	rcv_privs[FieldPair(Array::bck,0)] = READ_ONLY;
//
//	for(auto field : wrapper_map)
//	{
//		std::vector<FieldID> ro_fids;
//		std::vector<FieldID> rw_fids;
//		for(auto cpriv : child_privs)
//		{
//			if(cpriv.first.first == field.first)
//			{
//				if(cpriv.second == READ_ONLY)
//					ro_fids.push_back(cpriv.first.second);
//				else if(cpriv.second == READ_WRITE)
//					rw_fids.push_back(cpriv.first.second);
//
//			}
//		}
//		priv_field_map[field.first][READ_ONLY] = ro_fids;
//		priv_field_map[field.first][READ_WRITE] = rw_fids;
//
//
//	}

	LegionMatrix<LRWrapper> angFluxes(ctx,_lw_chunkArrays,FIDs::Array::angFluxes);
	LegionMatrix<LRWrapper> scalFluxes(ctx,_lw_chunkArrays,FIDs::Array::scalFluxes);
	LegionMatrix<LRWrapper> sigmaT(ctx,_lw_chunkArrays,FIDs::Array::sigmaT);
	LegionMatrix<LRWrapper> bci(ctx,_lw_chunkArrays,FIDs::Array::bci);
	LegionMatrix<LRWrapper> bcj(ctx,_lw_chunkArrays,FIDs::Array::bcj);
	LegionMatrix<LRWrapper> bck(ctx,_lw_chunkArrays,FIDs::Array::bck);

	std::vector<LegionMatrix<LRWrapper>> wrapper_vec;
	wrapper_vec.push_back(angFluxes);
	wrapper_vec.push_back(scalFluxes);
	wrapper_vec.push_back(sigmaT);
	wrapper_vec.push_back(bci);
	wrapper_vec.push_back(bcj);
	wrapper_vec.push_back(bck);

	std::map<FieldID,int> wrapper_map;
	wrapper_map[Array::angFluxes] = 0;
	wrapper_map[Array::scalFluxes] = 1;
	wrapper_map[Array::sigmaT] = 2;
	wrapper_map[Array::bci] = 3;
	wrapper_map[Array::bcj] = 4;
	wrapper_map[Array::bck] = 5;

//	typename std::map<PrivilegeMode,std::vector<FieldID>> PrivPair;
	std::map<FieldID,std::map<PrivilegeMode,std::vector<FieldID>>> priv_field_map;


	typedef std::pair<FieldID,FieldID> FieldPair;
	std::map<FieldPair,PrivilegeMode> child_privs;

	child_privs[FieldPair(Array::angFluxes,AngFlux::ptr_in)] = READ_ONLY;
	child_privs[FieldPair(Array::angFluxes,AngFlux::ptr_out)] = READ_WRITE;
	child_privs[FieldPair(Array::scalFluxes,ScalFlux::qtot)] = READ_ONLY;
	child_privs[FieldPair(Array::scalFluxes,ScalFlux::flux)] = READ_WRITE;

	child_privs[FieldPair(Array::sigmaT,SigmaT::t_xs)] = READ_ONLY;

	child_privs[FieldPair(Array::bci,0)] = READ_WRITE;
	child_privs[FieldPair(Array::bcj,0)] = READ_WRITE;
	child_privs[FieldPair(Array::bck,0)] = READ_WRITE;

	for(auto field : wrapper_map)
	{
		std::vector<FieldID> ro_fids;
		std::vector<FieldID> rw_fids;
		for(auto cpriv : child_privs)
		{
			if(cpriv.first.first == field.first)
			{
				if(cpriv.second == READ_ONLY)
					ro_fids.push_back(cpriv.first.second);
				else if(cpriv.second == READ_WRITE)
					rw_fids.push_back(cpriv.first.second);

			}
		}
		priv_field_map[field.first][READ_ONLY] = ro_fids;
		priv_field_map[field.first][READ_WRITE] = rw_fids;


	}



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

	Coloring cl_psii;
	Coloring cl_psij;
	Coloring cl_psik;
	Coloring cl_chunkArrays;

	Coloring cl_bci;
	Coloring cl_bcj;
	Coloring cl_bck;

	std::map<FieldID,Coloring> cl_nested;


//	RegionRequirement rr_map(lw_proc_map.lr,READ_WRITE,EXCLUSIVE,lw_proc_map.lr);
//	rr_map.add_field(0);
//	rr_map.add_field(1);
//	rr_map.add_field(2);
//
//	PhysicalRegion pr = runtime->map_region(ctx,rr_map);
//	{
//	LegionMatrix<int> proc_map_x(lw_proc_map,pr,0);
//	LegionMatrix<int> proc_map_y(lw_proc_map,pr,1);
//	LegionMatrix<int> proc_map_z(lw_proc_map,pr,2);



	auto append = [](ColoredPoints<ptr_t> &point,ColoredPoints<ptr_t> new_points)
				{

					if(point.points.size() == 0 && point.ranges.size() == 0)
						point = new_points;
					if(new_points.ranges.size() > 0)
						point.ranges.insert(new_points.ranges.begin(),new_points.ranges.end());
					if(new_points.points.size() > 0)
							point.points.insert(new_points.points.begin(),new_points.points.end());

				};

	auto gen_region_req = [&](ColoredPoints<ptr_t> points,LRWrapper lw, FieldID fid)
		{
			Coloring cl;
			cl[0] = points;
			ColoredPart lp_tmp(ctx,runtime,lw.lr,cl);
			LogicalRegion lr_tmp = runtime->get_logical_subregion_by_color(ctx,lp_tmp.lPart,0);
			std::vector<RegionRequirement> rr_out;
			for(auto privs : priv_field_map[fid])
			{
				if(privs.second.size() > 0){
				RegionRequirement rr(lr_tmp,0,privs.first,RELAXED,lw.lr);
				rr.add_fields(privs.second);
				rr_out.push_back(rr);
				}
			}
			return rr_out;
		};
	int iDom = 0;
	for(int kproc=kproc_lo;kproc!=kproc_hi;kproc+=knc) // Loop over k processors
		for(int jproc=jproc_lo;jproc!=jproc_hi;jproc+=jnc) // Loop over j processors
			for(int iproc=iproc_lo;iproc!=iproc_hi;iproc+=inc) // Loop over i processors
//			for(int ig=0;ig<params.ng;ig++)
			{
				int iDom_old = iproc+nCx*(jproc + nCy*kproc);
				// Remap the region requirements that we set up with construction
//				rreqs[iDom] = rreq_tmp[iDom_old];

//				printf("rreq[%i] has %i reqs\n",iDom,rreqs[iDom].size());
				int iC = iproc;
				int jC = jproc;
				int kC = kproc;

//				proc_map_x(iDom) = iC;
//				proc_map_y(iDom) = jC;
//				proc_map_z(iDom) = kC;

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

//				printf("for dom %i rcv = %i, %i, %i\n",iDom,xp_rcv,yp_rcv,zp_rcv);


//				cl_psii[iDom] = lw_psii.GetColoring(":",":",":",iproc,jproc,kproc);
//				cl_psij[iDom] = lw_psij.GetColoring(":",":",":",iproc,jproc,kproc);
//				cl_psik[iDom] = lw_psik.GetColoring(":",":",":",iproc,jproc,kproc);
//				cl_chunkArrays[iDom] = _lw_chunkArrays.GetColoring(iproc,jproc,kproc);

				cl_psii[iDom] = lw_psii.GetColoring(":",":",":",":",iproc,jproc,kproc);
				cl_psij[iDom] = lw_psij.GetColoring(":",":",":",":",iproc,jproc,kproc);
				cl_psik[iDom] = lw_psik.GetColoring(":",":",":",":",iproc,jproc,kproc);
				cl_chunkArrays[iDom] = _lw_chunkArrays.GetColoring(iproc,jproc,kproc);

				if(xp_rcv != iproc)
				cl_bci[iDom] = _lw_chunkArrays.GetColoring(xp_rcv,jproc,kproc);
				if(yp_rcv != jproc)
				cl_bcj[iDom] = _lw_chunkArrays.GetColoring(iproc,yp_rcv,kproc);
				if(zp_rcv != kproc)
				cl_bck[iDom] = _lw_chunkArrays.GetColoring(iproc,jproc,zp_rcv);

				append(cl_bci[iDom],cl_chunkArrays[iDom]);
				append(cl_bcj[iDom],cl_chunkArrays[iDom]);
				append(cl_bck[iDom],cl_chunkArrays[iDom]);


//				cl_nested[Array::angFluxes][iDom] = angFluxes(iproc,jproc,kproc).cast().GetColoring(":",":",":",":",ig,id+2*jd+4*kd);
//				cl_nested[Array::scalFluxes][iDom] = scalFluxes(iproc,jproc,kproc).cast().GetColoring(":",":",":",":",ig);
//				cl_nested[Array::sigmaT][iDom] = sigmaT(iproc,jproc,kproc).cast().GetColoring(":",":",":",ig);
//				cl_nested[Array::bci][iDom] = bci(iproc,jproc,kproc).cast().GetColoring(":",":",":",ig);
//				cl_nested[Array::bcj][iDom] = bcj(iproc,jproc,kproc).cast().GetColoring(":",":",":",ig);
//				cl_nested[Array::bck][iDom] = bck(iproc,jproc,kproc).cast().GetColoring(":",":",":",ig);

//				for(auto coloring : cl_nested)
//				{
//					LRWrapper tmp = wrapper_vec[wrapper_map[coloring.first]](iC,jC,kC).cast();
//					ColoredPart lp_tmp(ctx,runtime,tmp.lr,coloring.second);
//					LogicalRegion lr_tmp = runtime->get_logical_subregion_by_color(ctx,lp_tmp.lPart,iDom);
//					for(auto privs : priv_field_map[coloring.first])
//					{
//						RegionRequirement rr(lr_tmp,privs.first,EXCLUSIVE,tmp.lr);
//						rr.add_fields(privs.second);
//						rreqs[iDom].push_back(rr);
//					}
//				}

				std::vector<RegionRequirement> rr_angFLuxes;
				rr_angFLuxes = gen_region_req(angFluxes(iC,jC,kC).cast().GetColoring(":",":",":",":",":",id+2*jd+4*kd),
				                              angFluxes(iC,jC,kC).cast(),Array::angFluxes);

				std::vector<RegionRequirement> rr_scalFluxes;
				rr_scalFluxes = gen_region_req(scalFluxes(iC,jC,kC).cast().GetColoring(":",":",":",":",":"),
				                               scalFluxes(iC,jC,kC).cast(),Array::scalFluxes);

				std::vector<RegionRequirement> rr_sigmaT;
				rr_sigmaT = gen_region_req(sigmaT(iC,jC,kC).cast().GetColoring(":",":",":",":"),
				                           sigmaT(iC,jC,kC).cast(),Array::sigmaT);

				std::vector<RegionRequirement> rr_bci;
				rr_bci = gen_region_req(bci(iC,jC,kC).cast().GetColoring(":",":",":",":"),
				                        bci(iC,jC,kC).cast(),Array::bci);

				std::vector<RegionRequirement> rr_bcj;
				rr_bcj = gen_region_req(bcj(iC,jC,kC).cast().GetColoring(":",":",":",":"),
				                        bcj(iC,jC,kC).cast(),Array::bcj);

				std::vector<RegionRequirement> rr_bck;
				rr_bck = gen_region_req(bck(iC,jC,kC).cast().GetColoring(":",":",":",":"),
				                        bck(iC,jC,kC).cast(),Array::bck);


				rreqs[iDom].insert(rreqs[iDom].end(),rr_angFLuxes.begin(),rr_angFLuxes.end());
				rreqs[iDom].insert(rreqs[iDom].end(),rr_scalFluxes.begin(),rr_scalFluxes.end());
//				rreqs[iDom].insert(rreqs[iDom].end(),rr_sigmaT.begin(),rr_sigmaT.end());

				rreqs[iDom].insert(rreqs[iDom].end(),rr_bci.begin(),rr_bci.end());
				rreqs[iDom].insert(rreqs[iDom].end(),rr_bcj.begin(),rr_bcj.end());
				rreqs[iDom].insert(rreqs[iDom].end(),rr_bck.begin(),rr_bck.end());


				if(xp_rcv != iproc)
				{
					LRWrapper tmp = bci(xp_rcv,jproc,kproc).cast();

					Coloring cl_tmp;
					cl_tmp[iDom] = tmp.GetColoring(":",":",":",":");
					ColoredPart lp_tmp(ctx,runtime,tmp.lr,cl_tmp);
					LogicalRegion lr_tmp = runtime->get_logical_subregion_by_color(ctx,lp_tmp.lPart,iDom);
					RegionRequirement rr(lr_tmp,0,READ_ONLY,RELAXED,tmp.lr);
					rr.add_field(0);
					rreqs[iDom].push_back(rr);

				}
				if(yp_rcv != jproc)
				{
					LRWrapper tmp = bcj(iproc,yp_rcv,kproc).cast();
					Coloring cl_tmp;
					cl_tmp[iDom] = tmp.GetColoring(":",":",":",":");
					ColoredPart lp_tmp(ctx,runtime,tmp.lr,cl_tmp);
					LogicalRegion lr_tmp = runtime->get_logical_subregion_by_color(ctx,lp_tmp.lPart,iDom);
					RegionRequirement rr(lr_tmp,0,READ_ONLY,RELAXED,tmp.lr);
					rr.add_field(0);
					rreqs[iDom].push_back(rr);
				}
				if(zp_rcv != kproc)
				{
					LRWrapper tmp = bck(iproc,jproc,zp_rcv).cast();
					Coloring cl_tmp;
					cl_tmp[iDom] = tmp.GetColoring(":",":",":",":");
//					for(auto range : cl_tmp[iDom].ranges)
//						printf("bck[%i] ranges = %u to %u\n",iDom,(unsigned)range.first,(unsigned)range.second);
					ColoredPart lp_tmp(ctx,runtime,tmp.lr,cl_tmp);
					LogicalRegion lr_tmp = runtime->get_logical_subregion_by_color(ctx,lp_tmp.lPart,iDom);
					RegionRequirement rr(lr_tmp,0,READ_ONLY,RELAXED,tmp.lr);
					rr.add_field(0);
					rreqs[iDom].push_back(rr);
				}
				iDom++;
			}
//	}
//	runtime->unmap_region(ctx,pr);

	ColoredPart lp_chunk(ctx,runtime,_lw_chunkArrays.lr,cl_chunkArrays);
	ColoredPart lp_bci(ctx,runtime,_lw_chunkArrays.lr,cl_bci);
	ColoredPart lp_bcj(ctx,runtime,_lw_chunkArrays.lr,cl_bcj);
	ColoredPart lp_bck(ctx,runtime,_lw_chunkArrays.lr,cl_bck);
//	ColoredPart lp_psii(ctx,runtime,lw_psii.lr,cl_psii);
//	ColoredPart lp_psij(ctx,runtime,lw_psij.lr,cl_psij);
//	ColoredPart lp_psik(ctx,runtime,lw_psik.lr,cl_psik);
//	SimpleSubPart lp_procMap(ctx,runtime,lw_proc_map.lr,nCx*nCy*nCz);

	typedef std::tuple<PVecItem,FieldID,PrivilegeMode> ArgTuple;
	std::vector<ArgTuple> args;
	args.push_back(ArgTuple(lp_chunk,Array::angFluxes,READ_ONLY));
	args.push_back(ArgTuple(lp_chunk,Array::scalFluxes,READ_ONLY));
	args.push_back(ArgTuple(lp_chunk,Array::sigmaT,READ_ONLY));
	args.push_back(ArgTuple(lp_bci,Array::bci,READ_ONLY));
	args.push_back(ArgTuple(lp_bcj,Array::bcj,READ_ONLY));
	args.push_back(ArgTuple(lp_bck,Array::bck,READ_ONLY));

//	args.push_back(ArgTuple(lp_psii,0,READ_WRITE));
//	args.push_back(ArgTuple(lp_psij,0,READ_WRITE));
//	args.push_back(ArgTuple(lp_psik,0,READ_WRITE));
////
//	args.push_back(ArgTuple(lp_diag,0,READ_ONLY));
//	args.push_back(ArgTuple(lp_diag,1,READ_ONLY));
//	args.push_back(ArgTuple(lp_lines,0,READ_ONLY));
//	args.push_back(ArgTuple(lp_lines,1,READ_ONLY));
//	args.push_back(ArgTuple(lp_lines,2,READ_ONLY));
//	args.push_back(ArgTuple(lp_procMap,0,READ_ONLY));
//	args.push_back(ArgTuple(lp_procMap,1,READ_ONLY));
//	args.push_back(ArgTuple(lp_procMap,2,READ_ONLY));

	return OpArgs(args,rreqs);

}


__host__
template<int IT> __attribute__ ((noinline))
void Dim3Sweep<IT>::evaluate(int iDom,lrAccessor<LRWrapper> lr_angFlux,
                         lrAccessor<LRWrapper> lr_scalFlux,
                         lrAccessor<LRWrapper> lr_sigmaT,
                         lrAccessor<LRWrapper> bci,
                         lrAccessor<LRWrapper> bcj,
                         lrAccessor<LRWrapper> bck)
//                         lrAccessor<double> psii,
//						 lrAccessor<double> psij,
//						 lrAccessor<double> psik,
//                         lrAccessor<double> diag_start,
//                         lrAccessor<double> diag_len,
//                         lrAccessor<double> lines_i,
//                         lrAccessor<double> lines_j,
//                         lrAccessor<double> lines_k)
{
	printf("In Dim3Sweep %i\n",iDom);

	srand(5928347);

	double f = 53.0;

	for(int idiag = 0;idiag<params.nDiag;idiag++)
	{
//		int nmax = 1000*(rand()%10 + 1);

		int nmax = 2000;
		f += (rand()%120)*0.1;

		for(int it=0;it<nmax;it++)
		{
			f = sqrt(f)*(f+1.0)+(rand()%120)*0.1;
		}
	}
	hi = f;

}

template<int i>__attribute__ ((noinline))
FieldKernelLauncher genDim3Sweep(int igen,Context ctx, HighLevelRuntime* runtime,
		   int _id, int _jd, int _kd,
		   SlappyParams _params,
		   LRWrapper _lw_chunkArrays,
		   CopiedPart _lw_diag,
		   CopiedPart _lw_lines,
		   int _igrp,
               int iOct)
{
	if(igen == i)
	{
		Dim3Sweep<i> sweep0(ctx,runtime,_params,_lw_chunkArrays,_lw_diag,_lw_lines,_igrp);

		return genFieldKernel(sweep0,ctx,runtime,_lw_chunkArrays,_id,_jd,_kd);
	}
	else
		return genDim3Sweep<i-1>(igen,ctx,runtime,_id,_jd,_kd,_params,_lw_chunkArrays,_lw_diag,_lw_lines,_igrp,iOct);
}

template<>__attribute__ ((noinline))
FieldKernelLauncher genDim3Sweep<0>(int igen,Context ctx, HighLevelRuntime* runtime,
                   		   int _id, int _jd, int _kd,
                   		   SlappyParams _params,
                   		   LRWrapper _lw_chunkArrays,
                   		   CopiedPart _lw_diag,
                   		   CopiedPart _lw_lines,
                   		   int _igrp,
                                  int iOct)
{
	Dim3Sweep<0> sweep0(ctx,runtime,_params,_lw_chunkArrays,_lw_diag,_lw_lines,_igrp);

	return genFieldKernel(sweep0,ctx,runtime,_lw_chunkArrays,_id,_jd,_kd);
}

std::vector<FieldKernelLauncher> genOctSweeps(Context ctx, HighLevelRuntime* runtime,
		   SlappyParams _params,
		   LRWrapper _lw_chunkArrays,
		   CopiedPart _lw_diag,
		   CopiedPart _lw_lines,
		   int _igrp)
{
	std::vector<FieldKernelLauncher> res;

	for(int kd=0;kd<2;kd++)
		for(int jd=0;jd<2;jd++)
			for(int id=0;id<2;id++)
			{
				res.push_back(genDim3Sweep<15>(id+2*jd+4*kd,ctx,runtime,id,jd,kd,_params,_lw_chunkArrays,_lw_diag,_lw_lines,0,id+2*jd+4*kd));
			}
	return res;
}






template<int IT>
double __attribute__ ((noinline)) genDim3Evaluate(int iGen,Context ctx, HighLevelRuntime* runtime,
                 SlappyParams _params,
         		   LRWrapper _lw_chunkArrays,
         		   CopiedPart _lw_diag,
         		   CopiedPart _lw_lines,
         		   int _igrp,int iDom,lrAccessor<LRWrapper>& lr_angFlux,
                         lrAccessor<LRWrapper>& lr_scalFlux,
                         lrAccessor<LRWrapper>& lr_sigmaT,
                         lrAccessor<LRWrapper>& bci,
                         lrAccessor<LRWrapper>& bcj,
                         lrAccessor<LRWrapper>& bck)
{
	if(iGen == IT)
	{	Dim3Sweep<IT> sweep(ctx,runtime,_params,_lw_chunkArrays,_lw_diag,_lw_lines,_igrp);
		 sweep.evaluate(iDom,lr_angFlux,lr_scalFlux,lr_sigmaT,bci,bcj,bck);
		 return sweep.hi;
	}
	else
	return genDim3Evaluate<IT-1>(iGen,ctx,runtime,_params,_lw_chunkArrays,_lw_diag,_lw_lines,_igrp,iDom,lr_angFlux,lr_scalFlux,lr_sigmaT,bci,bcj,bck);
}


template<>
double  __attribute__ ((noinline)) genDim3Evaluate<0>(int iGen,Context ctx, HighLevelRuntime* runtime,
                    SlappyParams _params,
            		   LRWrapper _lw_chunkArrays,
            		   CopiedPart _lw_diag,
            		   CopiedPart _lw_lines,
            		   int _igrp,int iDom,lrAccessor<LRWrapper>& lr_angFlux,
                         lrAccessor<LRWrapper>& lr_scalFlux,
                         lrAccessor<LRWrapper>& lr_sigmaT,
                         lrAccessor<LRWrapper>& bci,
                         lrAccessor<LRWrapper>& bcj,
                         lrAccessor<LRWrapper>& bck)
{
	Dim3Sweep<0> sweep(ctx,runtime,_params,_lw_chunkArrays,_lw_diag,_lw_lines,_igrp);
	sweep.evaluate(iDom,lr_angFlux,lr_scalFlux,lr_sigmaT,bci,bcj,bck);
	return sweep.hi;
}

double  __attribute__ ((noinline)) genEvaluates(int iGen,Context ctx, HighLevelRuntime* runtime,
        SlappyParams _params,
		   LRWrapper _lw_chunkArrays,
		   CopiedPart _lw_diag,
		   CopiedPart _lw_lines,
		   int _igrp,int iDom,lrAccessor<LRWrapper>& lr_angFlux,
                         lrAccessor<LRWrapper>& lr_scalFlux,
                         lrAccessor<LRWrapper>& lr_sigmaT,
                         lrAccessor<LRWrapper>& bci,
                         lrAccessor<LRWrapper>& bcj,
                         lrAccessor<LRWrapper>& bck)
{
	double res = 0;
	for(int i=0;i<15;i++)
	{
		res += genDim3Evaluate<15>(i,ctx,runtime,_params,_lw_chunkArrays,_lw_diag,_lw_lines,_igrp,iDom,lr_angFlux,lr_scalFlux,lr_sigmaT,bci,bcj,bck);
	}

	return res;
}

void RegisterDim3Variants()
{
	  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<0>>>();
	  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<1>>>();
	  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<2>>>();
	  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<3>>>();
	  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<4>>>();
	  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<5>>>();
	  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<6>>>();
	  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<7>>>();
	  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<8>>>();
	  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<9>>>();
	  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<10>>>();
	  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<11>>>();
	  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<12>>>();
	  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<13>>>();
	  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<14>>>();
	  TaskHelper::register_hybrid_variants<FieldKernelOp<Dim3Sweep<15>>>();
}










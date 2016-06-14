/*
 * geom.cc
 *
 *  Created on: Jun 19, 2015
 *      Author: payne
 */

#include <LRWrapper.h>
#include <LPWrapper.h>
#include <LegionHelper.h>

using namespace Dragon;

int setup_diags(Context ctx, HighLevelRuntime* runtime,
                 LRWrapper& diags, LRWrapper& lines,
				 int& ndiag, int nx_l, int ny_l, int nz_l)
{
	enum {FID_DiagLen,FID_DiagStart};
	enum {FID_ic,FID_j,FID_k};
	typedef struct
	{
		int x,y,z;
	} int3;
	// Just do the setup locally

	// arrays for the diags
	int diag_len[ndiag];
	int3* diag_lines[ndiag];
	int indx[ndiag];

	int diag_start[ndiag];

	size_t nPoints = 0; // Total number of lines - points

	memset(diag_start,0,ndiag*sizeof(int));
	memset(diag_len,0,ndiag*sizeof(int));
	memset(indx,0,ndiag*sizeof(int));

	// Figure out the length of each diagonal
	for(int k=0;k<nz_l;k++)
		for(int j=0;j<ny_l;j++)
			for(int i=0;i<nx_l;i++)
			{
				int nn = i+j+k;
				diag_len[nn] += 1;
			}

	printf("Finished finding diag lengths for %i diags\n",ndiag);
	// Allocate each line in each diag
	for(int i=0;i<ndiag;i++)
	{
		diag_lines[i] = (int3*)malloc(sizeof(int3)*diag_len[i]);
		diag_start[i] += nPoints;
		nPoints += diag_len[i];
	}

	printf("Finished allocating temp space for diags lines\n");

	// Populate the lines
	for(int k=0;k<nz_l;k++)
		for(int j=0;j<ny_l;j++)
			for(int i=0;i<nx_l;i++)
			{
				int nn = i+j+k;
				int ing = indx[nn];
				indx[nn] += 1;

				diag_lines[nn][ing].x = i;
				diag_lines[nn][ing].y = j;
				diag_lines[nn][ing].z = k;
			}

	printf("Finished filling diag lines, we have %i points\n",nPoints);
	int* diag_lines_ic = (int*)malloc(nPoints*sizeof(int));
	int* diag_lines_j = (int*)malloc(nPoints*sizeof(int));
	int* diag_lines_k = (int*)malloc(nPoints*sizeof(int));

	size_t iPoint = 0;
	// Put the line information in a form that I can apply some legion abstractions to
	for(int i=0;i<ndiag;i++)
	{
		for(int j=0;j<diag_len[i];j++)
		{
			diag_lines_ic[iPoint] = diag_lines[i][j].x;
			diag_lines_j[iPoint] = diag_lines[i][j].y;
			diag_lines_k[iPoint] = diag_lines[i][j].z;

			iPoint++;
		}
	}

	printf("Finished transforming diag lines\n");

	for(int i=0;i<ndiag;i++)
		free(diag_lines[i]);

	printf("Creating %i diags with %i total points\n",ndiag,nPoints);

	diags.create(ctx,runtime,"diagonals",{ndiag},0,int(),1,int());

	lines.create(ctx,runtime,"lines",{nPoints},0,int(),1,int(),2,int());


	// Fill the diag information
	LegionHelper lrhelper(ctx,runtime);

	lrhelper.set(diags.lr,FID_DiagLen,diag_len,ndiag);
	lrhelper.set(diags.lr,FID_DiagStart,diag_start,ndiag);
	lrhelper.set(lines.lr,FID_ic,diag_lines_ic,nPoints);
	lrhelper.set(lines.lr,FID_j,diag_lines_j,nPoints);
	lrhelper.set(lines.lr,FID_k,diag_lines_k,nPoints);

	free(diag_lines_ic);
	free(diag_lines_j);
	free(diag_lines_k);

	return nPoints;

}

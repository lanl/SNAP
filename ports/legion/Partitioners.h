/*
 * Partitioners.h
 *
 *  Created on: Jan 29, 2015
 *      Author: payne
 */

#ifndef PARTITIONERS_H_
#define PARTITIONERS_H_

#include "Utils.h"
#include <legion.h>
#include <tuple>
//#include "cuda.h"
//#include "cuda_runtime.h"
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;





class SimpleSubPart : public PVecItem
{
public:
	int nTotal,nSub;
	SimpleSubPart(Context ctx, HighLevelRuntime *runtime,LogicalRegion lr_in,int _nSub);

	const int partID;
	int PartitionID()const{return partID;};

	SimpleSubPart(void) : partID(2) {};
};

class SinglePart : public PVecItem
{
public:
	SinglePart(Context ctx,HighLevelRuntime *runtime,LogicalRegion lr_in);
private:
	const int partID;
	int PartitionID()const{return partID;};

	SinglePart(void) : partID(1) {};
};


class CopiedPart : public PVecItem
{
public:
	CopiedPart(Context ctx,HighLevelRuntime *runtime,LogicalRegion lr_in,int nDim);
private:
	const int partID;
	int PartitionID()const{return partID;};

	CopiedPart(void) : partID(3) {};
};

class ColoredPart : public PVecItem
{
public:
	ColoredPart(Context ctx, HighLevelRuntime *runtime,LogicalRegion lr_in,Coloring _coloring);

	const int partID;
	int PartitionID()const{return partID;};

	ColoredPart() : partID(4) {};
};

int nTotal_from_region(Context ctx, HighLevelRuntime *runtime,LogicalRegion lr_in);





#endif /* PARTITIONERS_H_ */

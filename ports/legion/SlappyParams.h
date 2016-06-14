/*
 * SlappyParams.h
 *
 *  Created on: Jun 21, 2015
 *      Author: payne
 */

#ifndef SLAPPYPARAMS_H_
#define SLAPPYPARAMS_H_

#include <legion.h>
#include <tuple>
#include <iostream>
#include <sstream>
#include <string>
#include <LegionMatrix.h>
#include <LRWrapper.h>
#include <MustEpochKernelLauncher.h>

using namespace Dragon;


// Each Chunk needs a separate copy of these arrays
namespace FIDs
{
struct Array
{
	enum
	{
		angFluxes,scalFluxes,savedFlux,sigmaT,sigmaS,mat,bci,bcj,bck
	};
};

struct AngFlux
{
	enum
	{
		ptr_in,ptr_out
	};
};

struct ScalFlux
{
	enum
	{
		q2grp,qtot,flux
	};
};

struct SavedFlux
{
	enum
	{
		fluxpo,fluxpi
	};
};

struct SigmaT
{
	enum
	{
		t_xs,a_xs,qi
	};
};

struct SigmaS
{
	enum
	{
		s_xs
	};
};

struct Mat
{
	enum
	{
		mat
	};
};

struct BCi
{
	enum
	{
		ib_out
	};
};

struct BCj
{
	enum
	{
		jb_out
	};
};

struct BCk
{
	enum
	{
		kb_out
	};
};
struct Diag{
enum {len,start};
enum {ic,j,k};
};
}

class SlappyParams
{
public:
	int nx,ny,nz;
	int nx_l,ny_l,nz_l;
	int nCx,nCy,nCz;
	int ng,nang,nmom;
	int noct;
	int nmat;
	int cmom;
	int nDiag,nPoints;
	double dt;
	double dx,dy,dz;
	bool fixup;
	int ndimen;

	SlappyParams(const InputArgs &command_args);

	void print();
};

void tokenize(const std::string& str, std::vector<int>& tokens, const std::string& delimiters = ",");


namespace lli {
  class CommaSeparatedVector
  {
      public:
	  CommaSeparatedVector(const std::initializer_list<int> _vals);

	  CommaSeparatedVector();

        // comma separated values list
        std::vector<int> values;

      operator std::string();
//      {
//    	  std::stringstream sstream;
//
//    	  for(auto val : values)
//    	  {
//    		  sstream << val << ",";
//    	  }
//
//    	  std::string res = sstream.str().substr(0,sstream.str().size()-1);
//
//
//    	  return res;
//      }

      friend std::ostream& operator<<(std::ostream& out,const CommaSeparatedVector &value);
//      {
//    	  std::stringstream sstream;
//
//    	  for(auto val : value.values)
//    	  {
//    		  sstream << val << ",";
//    	  }
//
//    	  std::string res = sstream.str().substr(0,sstream.str().size()-1);
//
//    	  out << res;
//    	  return out;
//      }
  };

  // mapper for "lli::CommaSeparatedVector"
  std::istream& operator>>(std::istream& in, CommaSeparatedVector &value);
//  {
//      std::string token;
//      in >> token;
//
//      tokenize(token, value.values);
//
//      return in;
//  }


}

std::map<std::pair<FieldID,FieldID>,PrivilegeMode> propagate_nested_regions_privs(Context ctx,HighLevelRuntime* rt,
                                                        LRWrapper parent,std::vector<FieldPrivlages> parent_privs);

std::vector<RegionRequirement> propagate_nested_regions(Context ctx,HighLevelRuntime* rt,
                                                        LRWrapper parent,
                                                        std::map<std::pair<FieldID,FieldID>,PrivilegeMode> child_privs);

std::map<Color,std::vector<RegionRequirement>> distribute_nested_regions(Context ctx,HighLevelRuntime* rt,
                                                        LRWrapper parent,
                                                        std::map<std::pair<FieldID,FieldID>,PrivilegeMode> child_privs);

#endif /* SLAPPYPARAMS_H_ */

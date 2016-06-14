/*
 * SlappyParams.cc
 *
 *  Created on: Jun 21, 2015
 *      Author: payne
 */

#include <stdio.h>
#include <stdlib.h>
#include "SlappyParams.h"
#include <boost/tokenizer.hpp>
#include <boost/program_options.hpp>
#include <boost/foreach.hpp>
#include <boost/algorithm/string.hpp>
#include "legion_tasks.h"

namespace po = boost::program_options;


void tokenize(const std::string& str, std::vector<int>& tokens, const std::string& delimiters)
{
  // Skip delimiters at beginning.
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);

  // Find first non-delimiter.
  std::string::size_type pos = str.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos) {
    // Found a token, add it to the vector.

	  std::istringstream tstream(str.substr(lastPos, pos - lastPos));
	  int tmp;
	  tstream >> tmp;
    tokens.push_back(tmp);

    // Skip delimiters.
    lastPos = str.find_first_not_of(delimiters, pos);

    // Find next non-delimiter.
    pos = str.find_first_of(delimiters, lastPos);
  }
}

namespace lli {
	CommaSeparatedVector::CommaSeparatedVector(const std::initializer_list<int> _vals) : values(_vals) {}

	CommaSeparatedVector::CommaSeparatedVector() {}

	// comma separated values list
	std::vector<int> values;

	CommaSeparatedVector::operator std::string()
	{
	  std::stringstream sstream;

	  for(auto val : values)
	  {
		  sstream << val << ",";
	  }

	  std::string res = sstream.str().substr(0,sstream.str().size()-1);


	  return res;
	}

	std::ostream& operator<<(std::ostream& out,const CommaSeparatedVector &value)
	{
	  std::stringstream sstream;

	  for(auto val : value.values)
	  {
		  sstream << val << ",";
	  }

	  std::string res = sstream.str().substr(0,sstream.str().size()-1);

	  out << res;
	  return out;
	}


	// mapper for "lli::CommaSeparatedVector"
	std::istream& operator>>(std::istream& in, CommaSeparatedVector &value)
	{
	std::string token;
	in >> token;

	tokenize(token, value.values);

	return in;
	}


}


SlappyParams::SlappyParams(const InputArgs &command_args) : noct(8)
{
	using namespace std;
	lli::CommaSeparatedVector global_dims({32,32,32});


	lli::CommaSeparatedVector domain_dims({1,1,1});


	// General Options
	po::options_description gen("General Options");
	gen.add_options()
		("help", "produce help message")
	;

	po::options_description psize("Problem Size Options");
	psize.add_options()
		("global-dims", po::value<lli::CommaSeparatedVector>()->multitoken()->default_value(global_dims),
				"global number of cells for each direction\nEx: --global-dims=nx,ny,nz")
		("nx", po::value<int>(&nx)->default_value(128), "global number of cells in the x-direction")
		("ny", po::value<int>(&ny)->default_value(128), "global number of cells in the y-direction")
		("nz", po::value<int>(&nz)->default_value(128), "global number of cells in the z-direction")

		("ng", po::value<int>(&ng)->default_value(16), "total number of velocity groups")
		("nang", po::value<int>(&nang)->default_value(16), "total number of angles")
		("nmom", po::value<int>(&nmom)->default_value(4), "total number of moments")
		("nmat", po::value<int>(&nmat)->default_value(2), "total number of materials")
	;

	po::options_description dsize("Domain Decomposition Options");
	dsize.add_options()
		("domain-dims", po::value<lli::CommaSeparatedVector>()->multitoken()->default_value(domain_dims),
				"number of domain chunks for each direction\nEx: --domain-dims=nCx,nCy,nCz")
		("nCx", po::value<int>(&nCx)->default_value(1), "number of domain chunks in the x-direction")
		("nCy", po::value<int>(&nCy)->default_value(1), "number of domain chunks in the y-direction")
		("nCz", po::value<int>(&nCz)->default_value(1), "number of domain chunks in the z-direction")
	;

	// Declare an options description instance which will include
	// all the options
	po::options_description all("Allowed options");
	all.add(gen).add(psize).add(dsize);

	// Declare an options description instance which will be shown
	// to the user
	po::options_description visible("Allowed options");
	visible.add(gen).add(psize).add(dsize);

	po::variables_map vm;
	po::command_line_parser parser(command_args.argc, command_args.argv);
	parser = parser.options(all).style(po::command_line_style::unix_style
		 | po::command_line_style::allow_long_disguise
		 | po::command_line_style::long_allow_adjacent).allow_unregistered();
	po::store(parser.run(), vm);
	po::notify(vm);

	if (vm.count("help")) {

		 std::stringstream stream;
					stream << visible;
					string helpMsg = stream.str ();
					boost::algorithm::replace_all (helpMsg, "--", "-");
					cout << helpMsg << endl;
		return;
	}

	if(vm.count("global-dims"))
	{
		if(vm.count("nx") || vm.count("ny") || vm.count("nz"))
			printf("-nx, -ny, or -nz used with -global-dims, -global-dims takes precedent\n");

		std::vector<int> gdims = vm["global-dims"].as<lli::CommaSeparatedVector>().values;
		switch(gdims.size())
		{
			case 0: printf("Warning, no global dims specified, using defaults\n");break;
			case 1: nx = gdims[0]; printf("Warning, only nx specified, using defaults for ny and nz\n"); break;
			case 2: nx = gdims[0];ny = gdims[1];printf("Warning, only nx and ny specified, using defaults for nz\n"); break;
			case 3: nx = gdims[0];ny = gdims[1];nz = gdims[2];break;
			default: break;
		}
	}

	if(vm.count("domain-dims"))
	{
		std::vector<int> gdims = vm["domain-dims"].as<lli::CommaSeparatedVector>().values;
		switch(gdims.size())
		{
			case 0: printf("Warning, no domain dims specified, using defaults\n");break;
			case 1: nCx = gdims[0]; printf("Warning, only nCx specified, using defaults for nCy and nCz\n"); break;
			case 2: nCx = gdims[0];nCy = gdims[1];printf("Warning, only nCx and nCy specified, using defaults for nCz\n"); break;
			case 3: nCx = gdims[0];nCy = gdims[1];nCz = gdims[2];break;
			default: break;
		}
	}

	assert(nx > 0);
	assert(ny > 0);
	assert(nz > 0);
	assert(nCx > 0);
	assert(nCy > 0);
	assert(nCz > 0);
	// Make sure all chunks are the same size
	assert(nx%nCx == 0);
	assert(ny%nCy == 0);
	assert(nz%nCz == 0);

	nx_l = nx/nCx;
	ny_l = ny/nCy;
	nz_l = nz/nCz;

	cmom=nmom*nmom;
	nDiag = nx_l + ny_l + nz_l - 2;

	dt = 1.0;
	fixup = true;
	dx = 1.0/nx;
	dy = 1.0/ny;
	dz = 1.0/nz;
	ndimen = 3;


	print();
}

void SlappyParams::print()
{
	printf("====== Global Spatial Dimensions ======\n");
	printf("nx: %i    ny: %i    nz: %i\n",nx,ny,nz);
	printf("========== Physics Parameters =========\n");
	printf("ng: %i    nang: %i    nmom: %i    nmat: %i\n",ng,nang,nmom,nmat);
	printf("======= Domain Decomp Dimensions ======\n");
	printf("nCx: %i    nCy: %i    nCz: %i\n",nCx,nCy,nCz);
	printf("======= Local Spatial Dimensions ======\n");
	printf("nx_l: %i    ny_l: %i    nz_l: %i\n",nx_l,ny_l,nz_l);
}


std::map<std::pair<FieldID,FieldID>,PrivilegeMode> propagate_nested_regions_privs(Context ctx,HighLevelRuntime* rt,
                                                        LRWrapper parent,std::vector<FieldPrivlages> parent_privs)
{

	std::map<std::pair<FieldID,FieldID>,PrivilegeMode> res;
	std::vector<FieldID> fids;
	for(auto fp : parent_privs)
	{
		fids.push_back(fp.fid);
	}

	PhysicalRegion pr;
	bool mapped_pr = 1;


	const Task* task = ctx->as_mappable_task();

	for(auto fp : parent_privs)
	{
		bool mapped_pr = 1;

		for(int i=0;i<task->regions.size();i++)
						{
					if(task->regions[i].region == parent.lr && task->regions[i].has_field_privilege(fp.fid))
							{
								pr = ctx->get_physical_region(i);
								mapped_pr=0;
								break;
							}
						}


				if(mapped_pr)
				{
				RegionRequirement rr(parent.lr,READ_ONLY,SIMULTANEOUS,parent.lr);
						rr.add_fields(fids);


						pr = rt->map_region(ctx,rr);
				}

		LegionMatrix<LRWrapper> wrappers(parent,pr,fp.fid);


		LRWrapper child = wrappers(0).cast();
		std::vector<FieldID> child_fids;
		child_fids.assign(child.fids,child.fids+child.nfields);
		for(auto cfid : child_fids)
			res[std::pair<FieldID,FieldID>(fp.fid,cfid)] = fp.priv;

//		if(mapped_pr)
//		rt->unmap_region(ctx,pr);
	}




	return res;

}


std::vector<RegionRequirement> propagate_nested_regions(Context ctx,HighLevelRuntime* rt,
                                                        LRWrapper parent,
                                                        std::map<std::pair<FieldID,FieldID>,PrivilegeMode> child_privs)
{

	std::vector<FieldID> parent_fields;
	parent_fields.assign(parent.fids,parent.fids+parent.nfields);
	std::vector<RegionRequirement> res;
	std::vector<FieldID> fids;
	for(auto fp : parent_fields)
	{
//		printf("adding field %i\n",fp);
		fids.push_back(fp);
	}

	PhysicalRegion pr;
	bool mapped_pr = 1;


	const Task* task = ctx->as_mappable_task();

	for(auto fp : fids)
	{
		bool mapped_pr = 1;

		for(int i=0;i<task->regions.size();i++)
					{
				if(task->regions[i].region == parent.lr && task->regions[i].has_field_privilege(fp))
						{
							pr = ctx->get_physical_region(i);
							mapped_pr=0;
							printf("Parent region is already mapped!\n");
							break;
						}
					}


			if(mapped_pr)
			{
			RegionRequirement rr(parent.lr,READ_ONLY,SIMULTANEOUS,parent.lr);
					rr.add_fields(fids);


					pr = rt->map_region(ctx,rr);
			}

		LegionMatrix<LRWrapper> wrappers(parent,pr,fp);

		for(int i=0;i<parent.ntotal;i++)
		{
			std::map<PrivilegeMode,std::vector<FieldID>> priv_map;
			LRWrapper child = wrappers(i).cast();

			printf("child of field %i has %i fields\n",fp,child.nfields);
			// We first need to build a map of which privileges get which fields
			for(int it=0;it<child.nfields;it++)
			{
				if(child_privs.find(std::pair<FieldID,FieldID>(fp,child.fids[it])) != child_privs.end())
					priv_map[child_privs[std::pair<FieldID,FieldID>(fp,child.fids[it])]].push_back(child.fids[it]);
			}

			// Create region requirements for all of the field privilege combinations that we need
			for(auto priv : priv_map)
			{
//				printf("Adding region requirement for field %i with %i fields\n",fp,priv.second.size());
//				RegionRequirement child_req(child.lr,priv.first,EXCLUSIVE,child.lr);
//				child_req.add_fields(priv.second);
//				child_req.flags |= NO_ACCESS_FLAG;

				std::vector<FieldID> inst_fields;
				std::set<FieldID> priv_fields(priv.second.begin(),priv.second.end());
				RegionRequirement rr(child.lr,priv.first,EXCLUSIVE,child.lr);
				rr.add_fields(priv.second,false);
				rr.add_flags(NO_ACCESS_FLAG);


				res.push_back(rr);
			}

		}
//		if(mapped_pr)
//		rt->unmap_region(ctx,pr);
	}


	return res;

}

int nTotal_from_region(Context ctx, HighLevelRuntime *runtime,LogicalRegion lr_in);

std::map<Color,std::vector<RegionRequirement>> distribute_nested_regions(Context ctx,HighLevelRuntime* rt,
                                                        LRWrapper parent,
                                                        std::map<std::pair<FieldID,FieldID>,PrivilegeMode> child_privs)
{

	std::vector<FieldID> parent_fields;
	parent_fields.assign(parent.fids,parent.fids+parent.nfields);
	std::map<Color,std::vector<RegionRequirement>> res;
	std::vector<FieldID> fids;
	for(auto fp : parent_fields)
	{
//		printf("adding field %i\n",fp);
		fids.push_back(fp);
	}
	PhysicalRegion pr;


	const Task* task = ctx->as_mappable_task();


	for(auto fp : fids)
	{
		bool mapped_pr = 1;

		for(int i=0;i<task->regions.size();i++)
			{
		if(task->regions[i].region == parent.lr && task->regions[i].has_field_privilege(fp))
				{
					pr = ctx->get_physical_region(i);
					mapped_pr=0;
					printf("Parent region is already mapped!\n");
					break;
				}
			}


			if(mapped_pr)
			{
			RegionRequirement rr(parent.lr,READ_ONLY,SIMULTANEOUS,parent.lr);
					rr.add_fields(fids);


					pr = rt->map_region(ctx,rr);
			}

		LegionMatrix<LRWrapper> wrappers(parent,pr,fp);

		for(int i=0;i<parent.ntotal;i++)
		{
			std::map<PrivilegeMode,std::vector<FieldID>> priv_map;
			LRWrapper child = wrappers(i).cast();


			printf("child of field %i has %i fields\n",fp,child.nfields);
			// We first need to build a map of which privileges get which fields
			for(int it=0;it<child.nfields;it++)
			{
				if(child_privs.find(std::pair<FieldID,FieldID>(fp,child.fids[it])) != child_privs.end())
					priv_map[child_privs[std::pair<FieldID,FieldID>(fp,child.fids[it])]].push_back(child.fids[it]);
			}

			// Create region requirements for all of the field privilege combinations that we need
			for(auto priv : priv_map)
			{

//				RegionRequirement child_req(child.lr,priv.first,EXCLUSIVE,child.lr);
//				child_req.add_fields(priv.second);
//
//				res[i].push_back(child_req);

				std::vector<FieldID> inst_fields;
				std::set<FieldID> priv_fields(priv.second.begin(),priv.second.end());
				RegionRequirement rr(child.lr,priv.first,EXCLUSIVE,child.lr);
				rr.add_fields(priv.second,true);

				const char* lr_name;
				rt->retrieve_name(rr.parent,lr_name);
				printf("Adding region requirement for region %s field %i with %i fields for %i\n",lr_name,fp,priv.second.size(),i);

//				rr.add_flags(NO_ACCESS_FLAG);

				res[i].push_back(rr);
//				(res[i].end()-1)->copy_without_mapping_info(rr);
			}

//			rt->destroy_index_partition(ctx,iPart);

		}
//		if(mapped_pr)
//		rt->unmap_region(ctx,pr);
	}



	return res;

}



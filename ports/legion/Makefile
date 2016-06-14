# Copyright 2014 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

CUDA ?=/usr/local/cuda
ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif
DRAGON_DIR=../../Dragon/src

USE_CUDA=0
#Flags for directing the runtime makefile what to include
DEBUG           ?= 1		# Include debugging symbols
OUTPUT_LEVEL    ?= 2	# Compile time print level
SHARED_LOWLEVEL ?= 0		# Use the shared low level
ALT_MAPPERS     ?= 0		# Compile the alternative mappers
export DEBUG
export USE_CUDA
export OUTPUT_LEVEL
export SHARED_LOWLEVEL
export ALT_MAPPERS
# Put the binary file name here
OUTFILE		?= slappy
# List all the application source files here
GEN_SRC		?= 	Slappy.cc geom.cc SlappyParams.cc Initializer.cc CrossGroupSrc.cc InGroupSource.cc Sweep.cc #SweepMapper.cc
				#Partitioners.cc CrossGroupSrc.cc InGroupSource.cc \
				#Dim3Sweep.cc #dim3_sweep_interface.cc# .cc files
GEN_GPU_SRC	?=  	# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?=   -I$(DRAGON_DIR) -I$(BOOST_INCLUDES) -I$(LG_RT_DIR)#-I/home/payne/Development/FieldKernelOp
CC_FLAGS	?=    -O2  -g  -std=c++11   -DDYNAMIC_TESTS  -DNODE_LOGGING -DORDERED_LOGGING #-DDEBUG_HIGH_LEVEL
NVCC_FLAGS	?=    -O2 -g  -std=c++11  -DDYNAMIC_TESTS     -DNODE_LOGGING -DORDERED_LOGGING #-DDEBUG_HIGH_LEVEL
GASNET_FLAGS	?=
LD_FLAGS	?= -L$(BOOST_LIBS) -lboost_system -lboost_program_options 

export CC_FLAGS
export CUDA







###########################################################################
#
#   Don't change anything below here
#   
###########################################################################

# All these variables will be filled in by the runtime makefile
LOW_RUNTIME_SRC	:=
HIGH_RUNTIME_SRC:=
GPU_RUNTIME_SRC	:=
MAPPER_SRC	:=

include $(LG_RT_DIR)/runtime.mk



# General shell commands
SHELL	:= /bin/sh
SH	:= sh
RM	:= rm -f
LS	:= ls
MKDIR	:= mkdir
MV	:= mv
CP	:= cp
SED	:= sed
ECHO	:= echo
TOUCH	:= touch
MAKE	:= make
ifndef GCC
GCC	:= g++
endif
ifndef NVCC
NVCC	:= $(CUDA)/bin/nvcc
endif
SSH	:= ssh
SCP	:= scp


GEN_OBJS	:= $(GEN_SRC:.cc=.o)
LOW_RUNTIME_OBJS:= $(LOW_RUNTIME_SRC:.cc=.o)
HIGH_RUNTIME_OBJS:=$(HIGH_RUNTIME_SRC:.cc=.o)
MAPPER_OBJS	:= $(MAPPER_SRC:.cc=.o)
# Only compile the gpu objects if we need to 
ifeq ($(strip $(SHARED_LOWLEVEL)),0)
GEN_GPU_OBJS	:= $(GEN_GPU_SRC:.cu=.o)
GPU_RUNTIME_OBJS:= $(GPU_RUNTIME_SRC:.cu=.o)
else
GEN_GPU_OBJS	:= $(GEN_GPU_SRC:.cu=.o)
GPU_RUNTIME_OBJS:=
endif

ALL_OBJS	:= $(GEN_OBJS) $(GEN_GPU_OBJS) $(LOW_RUNTIME_OBJS) $(HIGH_RUNTIME_OBJS) $(GPU_RUNTIME_OBJS) $(MAPPER_OBJS)

.PHONY: all
all: $(OUTFILE)


# If we're using the general low-level runtime we have to link with nvcc
$(OUTFILE) : $(ALL_OBJS) $(DRAGON_DIR)/libDragon.a
	@echo "---> Linking objects into one binary: $(OUTFILE)"
ifeq ($(strip $(SHARED_LOWLEVEL)),1)
	$(GCC) -o $(OUTFILE) $(ALL_OBJS) $(LD_FLAGS) $(GASNET_FLAGS) -L$(DRAGON_DIR) -lDragon
else
	$(NVCC) -o $(OUTFILE)  $(ALL_OBJS) $(LD_FLAGS) $(GASNET_FLAGS) -L$(DRAGON_DIR) -lDragon
endif



$(DRAGON_DIR)/libDragon.a : $(DRAGON_DIR)/Makefile $(DRAGON_DIR)/*.cc
	cd $(DRAGON_DIR) && $(MAKE) $(MFLAGS)

$(GEN_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)
	
%.o : %.cpp
	$(CXX) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)	

$(LOW_RUNTIME_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(HIGH_RUNTIME_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(MAPPER_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

ifeq ($(strip $(SHARED_LOWLEVEL)),0)
$(GEN_GPU_OBJS) : %.o : %.cu
	$(NVCC) -o $@ -c $< $(INC_FLAGS) $(NVCC_FLAGS) -DUSE_CUDA_FKOP
else
$(GEN_GPU_OBJS) : %.o : %.cu
	$(GCC) -o $@ -x c++ -c $<  $(INC_FLAGS) $(CC_FLAGS)
endif

$(GPU_RUNTIME_OBJS): %.o : %.cu
	$(NVCC) -o $@ -c $< $(INC_FLAGS) $(NVCC_FLAGS)

clean:
	@$(RM) -rf $(ALL_OBJS) $(OUTFILE)
	cd $(DRAGON_DIR) && make clean

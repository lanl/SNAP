# Copyright 2017 NVIDIA Corporation
#
# The U.S. Department of Energy funded the development of this software
# under subcontract B609478 with Lawrence Livermore National Security, LLC
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


ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

# Flags for directing the runtime makefile what to include
DEBUG           ?= 1		# Include debugging symbols
OUTPUT_LEVEL    ?= LEVEL_DEBUG	# Compile time logging level
SHARED_LOWLEVEL ?= 0		# Use shared-memory runtime (not recommended)
USE_CUDA        ?= 0		# Include CUDA support (requires CUDA)
USE_GASNET      ?= 0		# Include GASNet support (requires GASNet)
USE_HDF         ?= 0		# Include HDF5 support (requires HDF5)
ALT_MAPPERS     ?= 0		# Include alternative mappers (not recommended)

# Put the binary file name here
OUTFILE		?= snap
# List all the application source files here
GEN_SRC		?= main.cc  \
		   snap.cc  \
		   init.cc  \
		   outer.cc \
		   inner.cc \
		   expxs.cc \
		   sweep.cc \
		   mms.cc   \
		   mapper.cc\
		   convergence.cc # .cc files
GEN_GPU_SRC	?= gpu_outer.cu \
		   gpu_inner.cu	\
		   gpu_sweep.cu \
		   gpu_expxs.cu # .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?=
CC_FLAGS	?=
CC_FLAGS	+= -DSNAP_USE_RELAXED_COHERENCE # Do this until Realm supports multi-field reduction instances
CC_FLAGS	+= -std=c++11 # Need this to deal with linkage issues
NVCC_FLAGS	?=
GASNET_FLAGS	?=
LD_FLAGS	?=

ifeq ($(strip $(USE_CUDA)),1)
CC_FLAGS	+= -DUSE_GPU_KERNELS
GPU_ARCH	= kepler
endif

###########################################################################
#
#   Don't change anything below here
#   
###########################################################################

include $(LG_RT_DIR)/runtime.mk


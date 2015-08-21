
#ifeq ($(CUDA), yes)
#CC = nvcc
#CCFLAGS = -O3 -arch=sm_35  
#LINK = g++
#LINKFLAGS = 
#EXE = $(SRC:.cpp=.cuda)
#else
#CC = g++
#CCFLAGS = -O3
#LINK = g++
#LINKFLAGS =  
#EXE = $(SRC:.cpp=.host)
#endif

TRILINOS_INSTALL_DIR=$(TRILINOS_DIR)
include $(TRILINOS_INSTALL_DIR)/include/Makefile.export.Kokkos

LDFLAGS = -L$(TRILINOS_INSTALL_DIR)/lib
LDFLAGS += $(Kokkos_LIBRARIES) $(Kokkos_TPL_LIBRARIES)
LDFLAGS += $(Teuchos_LIBRARIES)
INCFLAGS = -I. -I$(TRILINOS_INSTALL_DIR)/include
INCFLAGS += $(Kokkos_TPL_INCLUDE_DIRS)
CXX = $(Kokkos_CXX_COMPILER)
CXXFLAGS = $(Kokkos_CXX_FLAGS)
#CXXFLAGS = $(Kokkos_CXX_FLAGS) -v -std=c++11

#vtune
#CXXFLAGS += -I$(VTUNE_AMPLIFIER_XE_2015_DIR)/include
#LDFLAGS += -L$(VTUNE_AMPLIFIER_XE_2015_DIR)/lib64 -littnotify

all: snappy.o

snappy.o: main.cpp snappy.hpp 
	$(CXX) $(CXXFLAGS) $(INCFLAGS) main.cpp $(LDFLAGS) -o snappy.x



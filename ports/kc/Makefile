#KOKKOS_PATH = /home/${USER}/kg-cuda-uvm-gnu
#KOKKOS_PATH = /home/${USER}/kg-cuda-7.5-gnu-4.9.3
KOKKOS_PATH = /home/${USER}/kg-cuda-uvm-gnu_5.3.0-release
#KOKKOS_PATH = /home/${USER}/kg-cuda-uvm-intel
#KOKKOS_PATH = /home/${USER}/kg-serial-intel
#KOKKOS_PATH = /home/${USER}/kg-openmp-intel

SRC = $(wildcard *.cpp)

default: build
	echo "Start Build"

CXX = $(KOKKOS_PATH)/config/nvcc_wrapper
#CXX = icpc
CXXFLAGS = -O3
LINK = ${CXX}
LINKFLAGS = 
EXE = snappy.cuda

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean 
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

A_PREFIX=snap
A_SUFFIX=_mkl_vml
A_TARGET=$(A_PREFIX)$(A_SUFFIX)

CC=mpiicc			# C compiler command
CFLAGS=-mkl -O3 -xAVX -std=c99 -fopenmp -parallel -DUSEMKL -DUSEVML	# C compiler flags
#CC=mpicc
#CFLAGS=-O3 -std=c99 -fopenmp -lm #-wall	# C compiler flags
LDFLAGS=			# Linker flags, e.g. -L<lib dir> if you have libraries in a nonstandard directory <lib dir>
LIBS=				# Libraries to pass to the linker, e.g. -l<library>
CPPFLAGS=			# (Objective) C/C++ preprocessor flags, e.g. -I<include dir> if you have headers in a nonstandard directory <include dir>
CXX=icc				# C++ compiler command
CXXFLAGS=-O3 -std=c++11 -openmp	# C++ compiler flags
CXXCPP=				# C++ preprocessor flags

A_FILES := $(wildcard *.cpp) $(wildcard *.c)
A_OBJS := $(sort $(patsubst %.c, %.o, $(patsubst %.cpp, %.o, $(A_FILES))))

INPUT_FILE=scale_test_8_4.inp
OUTPUT_PREFIX=snap_test
OUTPUT_SUFFIX=.out
OUTPUT_FILE=$(OUTPUT_PREFIX)$(OUTPUT_SUFFIX)

SRCS = snap.h snap_main.c utils.c version.c plib.c geom.c \
       sn.c data.c control.c input.c setup.c dealloc.c \
       translv.c solvar.c outer.c expxs.c inner.c sweep.c \
       octsweep.c dim1_sweep.c dim3_sweep.c output.c time.c mms.c


all: $(A_TARGET) 			# Build all targets

vars:					# Show Makefile variable values
	@echo "       CC "$(CC)
	@echo " A_TARGET "$(A_TARGET)
	@echo "   CFLAGS "$(CFLAGS)
	@echo "  LDFLAGS "$(LDFLAGS)
	@echo "     LIBS "$(LIBS)@egrep "[a-z]\w+:" Makefile | sort

clean:					# Remove make build object and target files
	-rm -f *{.o,~} $(A_TARGET)

cleanout: clean
	-rm -f $(OUTPUT_FILE)

.PHONY:	clean vars help install rebuild

rebuild: clean $(A_TARGET)		# Clean and build target

install: 				# Build and install

run: $(A_TARGET)
	./$(A_TARGET) --fi $(INPUT_FILE) --fo $(OUTPUT_FILE)

$(A_TARGET): $(A_OBJS)			# Build target and its required object files
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CXXCPP) -c $< -o $@

%: %.cpp
	$(CXX) $(CXXFLAGS) $(CXXCPP) -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

%: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $<

#
# Count lines of code
#
count:
	rm -f Lines
	for file in $(SRCS); do ./LineCount $$file Lines; done
	gawk -f ./LineReport < Lines >> Lines
	cat Lines

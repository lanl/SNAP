#
# Build SNAP
#
OPT = yes
FFLAGS =
FFLAG2 =
DEFS =
PP =

MPI = yes
#MPI = cray
OPENMP = yes

MPIFLAGS = 

FORTRAN = mpif90
#FORTRAN = mpifort
#FORTRAN = ftn
#FORTRAN = mpiifort

TARGET = gsnap
#TARGET = isnap
#TARGET = ksnap
#TARGET = xsnap

#HASWELL = yes

ifeq ($(MPI),cray)
  MPI = yes
  MPIFLAGS = -I$(MPICH_DIR)/include -L$(MPICH_DIR)/lib -lmpich
  ifeq ($(TARGET),gsnap)
    FORTRAN = gfortran
  else ifeq ($(TARGET),xsnap)
    FORTRAN = ifx
  else
    FORTRAN = ifort
  endif
endif

ifeq ($(MPI),yes)
  DEFS := $(DEFS) MPI
else
  ifeq ($(TARGET),gsnap)
    FORTRAN = gfortran
  else ifeq ($(TARGET),xsnap)
    FORTRAN = ifx
  else
    FORTRAN = ifort
  endif
endif

ifeq ($(OPENMP),yes)
  DEFS := $(DEFS) OPENMP
  ifeq ($(TARGET),gsnap)
    OMPFLAG = -fopenmp
  else ifeq ($(TARGET),xsnap)
    OMPFLAG = -fiopenmp
  else
    OMPFLAG = -qopenmp
  endif
else
  OMPFLAG =
endif

PPFLAGS = $(foreach def,$(DEFS),-D$(def))

ifeq ($(TARGET),gsnap)
#  FFLAGS = -Ofast -funroll-loops -march=native $(OMPFLAG)
  FFLAGS = -O3 $(OMPFLAG)
  FFLAG2 =
  PP = cpp
  ifeq ($(OPT),no)
    TARGET = gsnap_db
    FFLAGS = -O0 $(OMPFLAG) -g -fbounds-check -fbacktrace
    FFLAG2 =
  endif
endif

ifeq ($(TARGET),isnap)
  FFLAGS = -O3 $(OMPFLAG) -ip -align array32byte -qno-opt-dynamic-align -fno-fnalias -fp-model fast -fp-speculation fast
  ifeq ($(HASWELL),yes)
    FFLAGS := $(FFLAGS) -xcore-avx2
  else
    FFLAGS := $(FFLAGS) -xHost
  endif
  FFLAG2 =
  PP = fpp
  ifeq ($(OPT),no)
    TARGET = isnap_db
    FFLAGS = -O0 $(OMPFLAG) -g -check bounds -traceback -warn unused
    FFLAG2 =
  endif
  FFLAGS := $(FFLAGS) $(MPIFLAGS)
endif

ifeq ($(TARGET),ksnap)
  FFLAGS = -O3 $(OMPFLAG) -xmic-avx512 -ip -align array64byte -qno-opt-dynamic-align -fp-model fast -fp-speculation fast -fno-alias -fno-fnalias
  FFLAG2 =
  PP = fpp
  ifeq ($(OPT),no)
    TARGET = ksnap_db
    FFLAGS = -O0 $(OMPFLAG) -g -check bounds -traceback -warn unused
    FFLAG2 =
  endif
endif

ifeq ($(TARGET),xsnap)
  FFLAGS = -xcore-avx2 -Ofast -funroll-loops -nostandard-realloc-lhs $(OMPFLAG) -align array32byte
#  FFLAGS = -O3 $(OMPFLAG) -ip -align array32byte -qno-opt-dynamic-align -fno-fnalias -fp-model fast -fp-speculation fast
  FFLAGS := $(FFLAGS) -xcore-avx2
  FFLAG2 =
  PP = fpp
  ifeq ($(OPT),no)
    TARGET = xsnap_db
    FFLAGS = -O0 $(OMPFLAG) -g -check bounds -traceback -warn unused
    FFLAG2 =
  endif
  FFLAGS := $(FFLAGS) $(MPIFLAGS)
endif

.SUFFIXES:	.F90 .f90 .o

OBJS = global.o snap_main.o utils.o version.o plib.o geom.o sn.o \
       data.o control.o input.o setup.o dealloc.o translv.o solvar.o \
       outer.o expxs.o inner.o sweep.o octsweep.o dim1_sweep.o \
       dim3_sweep.o output.o time.o mms.o analyze.o thrd_comm.o mkba_sweep.o

SRCS = global.f90 snap_main.f90 utils.f90 version.f90 plib.F90 geom.f90 \
       sn.f90 data.f90 control.f90 input.f90 setup.f90 dealloc.f90 \
       translv.f90 solvar.f90 outer.f90 expxs.f90 inner.f90 sweep.f90 \
       octsweep.f90 dim1_sweep.f90 dim3_sweep.f90 output.f90 \
       time.F90 mms.f90 analyze.f90 thrd_comm.f90 mkba_sweep.f90

$(TARGET) :	$(OBJS)
		$(FORTRAN) $(FFLAGS) $(FFLAG2) -o $@ $(OBJS)

version.o :	global.o
geom.o :	global.o
sn.o :		global.o
data.o :	global.o
control.o :	global.o
time.o :	global.o
plib.o :	global.o time.o
thrd_comm.o:	global.o plib.o
expxs.o :	global.o geom.o sn.o data.o
solvar.o :	global.o plib.o geom.o sn.o data.o control.o
mms.o:		global.o geom.o sn.o data.o control.o plib.o
dealloc.o :	global.o sn.o data.o mms.o geom.o solvar.o control.o
utils.o :	global.o dealloc.o plib.o control.o
analyze.o:	global.o control.o geom.o data.o solvar.o plib.o
input.o :	global.o plib.o geom.o sn.o data.o control.o utils.o time.o
setup.o :	global.o geom.o sn.o data.o control.o mms.o utils.o time.o \
		plib.o
output.o :	global.o geom.o sn.o data.o control.o utils.o solvar.o \
		mms.o time.o plib.o
snap_main.o :	global.o utils.o version.o input.o setup.o output.o time.o \
		dealloc.o plib.o control.o
dim1_sweep.o :	global.o geom.o sn.o data.o control.o plib.o
dim3_sweep.o :	global.o plib.o geom.o sn.o data.o control.o thrd_comm.o
mkba_sweep.o :	global.o plib.o geom.o sn.o data.o control.o thrd_comm.o
octsweep.o :	global.o geom.o sn.o data.o control.o solvar.o \
		thrd_comm.o dim1_sweep.o dim3_sweep.o mkba_sweep.o
sweep.o :	global.o data.o geom.o sn.o control.o octsweep.o solvar.o \
		plib.o thrd_comm.o
inner.o :	global.o geom.o sn.o data.o control.o solvar.o sweep.o \
		time.o plib.o thrd_comm.o
outer.o :	global.o geom.o sn.o data.o solvar.o control.o inner.o \
		time.o plib.o expxs.o thrd_comm.o
translv.o :	global.o plib.o geom.o sn.o data.o control.o utils.o \
		solvar.o expxs.o outer.o time.o thrd_comm.o analyze.o

#
# Fortran rules
#
%.o:	%.F90
	$(PP) -P $(PPFLAGS) $< >$*.f90
	$(FORTRAN) $(FFLAGS) -c $*.f90

%.o:	%.f90
	$(FORTRAN) $(FFLAGS) -c $<

#
# Cleanup
#
clean:
	rm -f *.o *.mod *.bc

#
# Count lines of code
#
count:
	rm -f Lines
	for file in $(SRCS); do ./LineCount $$file Lines; done
	awk -f ./LineReport < Lines >> Lines
	cat Lines

#
# Link compiled files only. No recompile.
#
link:
	$(FORTRAN) $(FFLAGS) $(FFLAG2) -o $(TARGET) $(OBJS)



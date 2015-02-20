#
# Build SNAP
#
OPT = yes
FFLAGS =
FFLAG2 =

FORTRAN = mpif90
#COMPILER = gfortran
COMPILER = ifort

ifeq ($(COMPILER),gfortran)
  TARGET = snap
  FFLAGS = -O3 -fopenmp
  FFLAG2 =
  ifeq ($(OPT),no)
    TARGET = snap_db
    FFLAGS = -O0 -fopenmp -g -fbounds-check -fbacktrace
    FFLAG2 =
  endif
endif

ifeq ($(COMPILER),ifort)
  TARGET = isnap
  FFLAGS = -O3 -qopenmp -align array32byte -fp-model fast -fp-speculation fast -xHost
  FFLAG2 =
  ifeq ($(OPT),no)
    TARGET = isnap_db
    FFLAGS = -O0 -qopenmp -g -check bounds -traceback -warn unused
    FFLAG2 =
  endif
endif

.SUFFIXES:	.f90 .o

OBJS = global.o snap_main.o utils.o version.o plib.o geom.o sn.o \
       data.o control.o input.o setup.o dealloc.o translv.o solvar.o \
       outer.o expxs.o inner.o sweep.o octsweep.o dim1_sweep.o dim3_sweep.o \
       output.o time.o mms.o analyze.o thrd_comm.o mkba_sweep.o

SRCS = global.f90 snap_main.f90 utils.f90 version.f90 plib.f90 geom.f90 \
       sn.f90 data.f90 control.f90 input.f90 setup.f90 dealloc.f90 \
       translv.f90 solvar.f90 outer.f90 expxs.f90 inner.f90 sweep.f90 \
       octsweep.f90 dim1_sweep.f90 dim3_sweep.f90 output.f90 time.f90 mms.f90 \
       analyze.f90 thrd_comm.f90 mkba_sweep.f90

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
sweep.o :	global.o geom.o data.o control.o octsweep.o solvar.o plib.o \
		thrd_comm.o
inner.o :	global.o geom.o sn.o data.o control.o solvar.o sweep.o \
		time.o plib.o thrd_comm.o
outer.o :	global.o geom.o sn.o data.o solvar.o control.o inner.o \
		time.o plib.o expxs.o thrd_comm.o
translv.o :	global.o plib.o geom.o sn.o data.o control.o utils.o \
		solvar.o expxs.o outer.o time.o analyze.o

#
# Fortran rule
#
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



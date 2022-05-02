################################################################################
#
# CMakeLists.txt
# 
# Author: Anthony Cabrera:
# Contact: cabreraam AT ieee DOT org
# Description: CMakeLists.txt replacement for SNAP Makefile
#
################################################################################

################################################################################
#
# Options and Variables
#
################################################################################


option(MPI "Use MPI" OFF)
option(OPT "Turn on optimizations" ON)
option(OPENMP "Enable OpenMP" OFF)
option(KNL "Intel KNL target" OFF)
option(HASWELL "Does the CPU use the Haswell u-arch" OFF)
option(CRAY "Is the target system a Cray?" OFF)

if(MPI)
	find_package(MPI REQUIRED)
	message(STATUS "MPI_Fortran_COMPILER - ${MPI_Fortran_COMPILER}")
	message(STATUS "MPI_Fortran_COMPILE_OPTIONS - ${MPI_Fortran_COMPILE_OPTIONS}")
	message(STATUS "MPI_Fortran_COMPILE_DEFINITIONS - ${MPI_Fortran_COMPILE_DEFINITIONS}")
	message(STATUS "MPI_Fortran_INCLUDE_DIRS - ${MPI_Fortran_INCLUDE_DIRS}")
	message(STATUS "MPI_Fortran_LINK_FLAGS - ${MPI_Fortran_LINK_FLAGS}")
	message(STATUS "MPI_Fortran_LIB_NAMES - ${MPI_Fortran_LIB_NAMES}")
	message(STATUS "MPI_Fortran_LIBRARY - ${MPI_Fortran_LIBRARY}")
	message(STATUS "MPI_Fortran_LIBRARIES - ${MPI_Fortran_LIBRARIES}")
endif()

if(OPENMP)
	find_package(OpenMP REQUIRED)
	message(STATUS "OpenMP_VERSION - ${OpenMP_VERSION}")
	message(STATUS "OpenMP_Fortran_FOUND - ${OpenMP_Fortran_COMPILE_OPTIONS}")
	message(STATUS "OpenMP_Fortran_FLAGS - ${OpenMP_Fortran_COMPILE_DEFINITIONS}")
	message(STATUS "OpenMP_Fortran_INCLUDE_DIRS - ${OpenMP_Fortran_INCLUDE_DIRS}")
	message(STATUS "OpenMP_Fortran_LINK_FLAGS - ${OpenMP_Fortran_LINK_FLAGS}")
	message(STATUS "OpenMP_Fortran_LIB_NAMES - ${OpenMP_Fortran_LIB_NAMES}")
	message(STATUS "OpenMP_Fortran_LIBRARY - ${OpenMP_Fortran_LIBRARY}")
	message(STATUS "OpenMP_Fortran_LIBRARIES - ${OpenMP_Fortran_LIBRARIES}")
	message(STATUS "OpenMP_HOME - ${OpenMP_HOME}")
endif()

# Compiler Preprocessor Definitions
# These should be passed in at the command line as a semi-colon-separated list
# If the developer wants MPI, it would be set here
# e.g., MPI + other pound define --> -DUSER_DEFS="MPI;OTHER_VAR;OTHER_VAR2=1"
set(USER_DEFS "" CACHE STRING "Semi-colon separted preprocessor pound defines")


################################################################################
#
# Compile Options for Different Compilers
#
################################################################################

# Compile Options for GNU
set(SNAP_COMPILER_OPTIONS_GNU_OPT
	"-O3" 
	CACHE 
	STRING 
	"GNU compiler options when OPT=ON"
)
set(SNAP_COMPILER_OPTIONS_GNU_NOT_OPT
	"-O0"
	"-g"
	"-fbounds-check"
	"-fbacktrace"
	CACHE 
	STRING 
	"GNU compiler options when OPT=OFF"
)
set(SNAP_COMPILER_OPTIONS_GNU
	$<$<BOOL:${OPENMP}>:-fopenmp>
	$<IF:$<BOOL:${OPT}>,${SNAP_COMPILER_OPTIONS_GNU_OPT},${SNAP_COMPILER_OPTIONS_GNU_NOT_OPT}>
	CACHE
	STRING
	"Compiler options when GNU compiler detected"
)

# Compile Options for ifort 
set(SNAP_COMPILER_OPTIONS_IFORT_OPT
	"-O3"
	"-ip"
	"-align array32byte"
	"-qno-opt-dynamic-align"
	"-fno-fnalias"
	"-fp-model fast"
	"-fp-speculation fast"
	CACHE 
	STRING 
	"ifort compiler options when OPT=ON"
)
set(SNAP_COMPILER_OPTIONS_INTEL_NOT_OPT
	"-O0"
	"-g "
	"-check bounds"
	"-traceback"
	"-warn unused"
	CACHE 
	STRING 
	"ifort compiler options when OPT=OFF"
)
set(SNAP_COMPILER_OPTIONS_IFORT
	$<$<BOOL:${OPENMP}>:-qopenmp>
	$<IF:$<BOOL:${OPT}>,${SNAP_COMPILER_OPTIONS_IFORT_OPT},${SNAP_COMPILER_OPTIONS_INTEL_NOT_OPT}>
	$<IF:$<BOOL:${HASWELL}>,"-xcore-avx2","-xHost">
	CACHE
	STRING
	"Compiler options when ifort compiler detected"
)

# Compile Options for ifort knl
set(SNAP_COMPILER_OPTIONS_IFORT_KNL_OPT
	"-O3"
	"-xmic-avx512"
	"-ip"
	"-align array64byte"
	"-qno-opt-dynamic-align"
	"-fp-model fast"
	"-fp-speculation fast"
	"-fno-alias -fno-fnalias"
	CACHE 
	STRING 
	"ifort compiler options when OPT=ON and KNL=ON"
)
# This is the same as SNAP_COMPILER_OPTIONS_INTEL_NOT_OPT
#set(SNAP_COMPILER_OPTIONS_IFORT_KNL_NOT_OPT
#)
set(SNAP_COMPILER_OPTIONS_IFORT_KNL
	$<$<BOOL:${OPENMP}>:-qopenmp>
	$<IF:$<BOOL:${OPT}>,${SNAP_COMPILER_OPTIONS_IFORT_KNL_OPT},${SNAP_COMPILER_OPTIONS_INTEL_NOT_OPT}>
	CACHE
	STRING
	"Compiler options when ifort + knl compiler detected"
)

# Compile Options for intel ifx
set(SNAP_COMPILER_OPTIONS_IFX_OPT
	"-xcore-avx2"
	"-Ofast"
	"-funroll-loops"
	"-nostandard-realloc-lhs"
	"-align array32byte"
	CACHE 
	STRING 
	"ifx compiler options when OPT=ON"
)
# This is the same as SNAP_COMPILER_OPTIONS_INTEL_NOT_OPT
#set(SNAP_COMPILER_OPTIONS_IFX_OPT
#)
set(SNAP_COMPILER_OPTIONS_IFX
	$<$<BOOL:${OPENMP}>:-fiopenmp>
	$<IF:$<BOOL:${OPT}>,${SNAP_COMPILER_OPTIONS_IFX_OPT},${SNAP_COMPILER_OPTIONS_INTEL_NOT_OPT}>
	CACHE
	STRING
	"Compiler options when ifx compiler detected"
)

################################################################################
#
# Setting up build target
#
################################################################################

set(SNAP_SOURCE_FILES
	global.f90
	snap_main.f90
	utils.f90
	version.f90
	plib.F90
	geom.f90
	sn.f90
	data.f90
	control.f90
	input.f90
	setup.f90
	dealloc.f90
	translv.f90
	solvar.f90
	outer.f90
	expxs.f90
	inner.f90
	sweep.f90
	octsweep.f90
	dim1_sweep.f90
	dim3_sweep.f90
	output.f90
	time.F90
	mms.f90
	analyze.f90
	thrd_comm.f90
	mkba_sweep.f90
	CACHE
	STRING
	"List of SNAP source files"
)

# Handle files that need to be proprocessed, i.e., files with .F90 and NOT .f90
set(SNAP_SOURCE_FILES_TO_PREPROC ${SNAP_SOURCE_FILES})
list(FILTER SNAP_SOURCE_FILES_TO_PREPROC INCLUDE REGEX .*\.F90)
set(SNAP_SOURCE_FILES_TO_PREPROC 
	${SNAP_SOURCE_FILES_TO_PREPROC}
	CACHE
	STRING
	"List of SNAP source files to preprocess"
)
set_source_files_properties(
	${SNAP_SOURCE_FILES_TO_PREPROC}
	PROPERTIES
		Fortran_PREPROCESS ON
)

add_executable(snap)
target_sources(snap
	PRIVATE
		${SNAP_SOURCE_FILES}
)

target_compile_options(snap 
	PRIVATE
		$<$<Fortran_COMPILER_ID:GNU>:${SNAP_COMPILER_OPTIONS_GNU}>
		$<$<AND:$<Fortran_COMPILER_ID:Intel>,$<NOT:$<BOOL:${KNL}>>>:${SNAP_COMPILER_OPTIONS_IFORT}>
		$<$<AND:$<Fortran_COMPILER_ID:Intel>,$<BOOL:${KNL}>>:${SNAP_COMPILER_OPTIONS_IFORT_KNL}>
		$<$<Fortran_COMPILER_ID:IntelLLVM>:${SNAP_COMPILER_OPTIONS_IFX}>
)
target_compile_definitions(snap
	PRIVATE
		${USER_DEFS}
		$<$<AND:$<BOOL:${MPI}>,$<BOOL:${CRAY}>>:MPI>
		$<$<BOOL:${OPENMP}>:OPENMP>
)
# -I$(MPICH_DIR)/include
target_include_directories(snap
	PRIVATE
		$<$<AND:$<BOOL:${MPI}>,$<BOOL:${CRAY}>>:${MPI_Fortran_INCLUDE_DIRS}>
		$<$<BOOL:${OPENMP}>:${OpenMP_Fortran_INCLUDE_DIRS}>
)
target_link_libraries(snap
	PRIVATE
		$<$<AND:$<BOOL:${MPI}>,$<BOOL:${CRAY}>>:MPI::MPI_Fortran>
		$<$<BOOL:${OPENMP}>:${OpenMP_Fortran_LIBRARIES}>
)

# create line counting functionality
add_custom_target(
	count
	DEPENDS
		Lines
)

add_custom_command(
	OUTPUT
		src_list.txt
	DEPENDS
		${SNAP_SOURCE_FILES}
	COMMAND
		cmake -E touch ${CMAKE_CURRENT_BINARY_DIR}/src_list.txt
	COMMAND
		echo ${SNAP_SOURCE_FILES} > ${CMAKE_CURRENT_BINARY_DIR}/src_list.txt
	COMMENT
		"Generate source file list lines of code"
	VERBATIM
)
add_custom_command(
	OUTPUT
		Lines
	DEPENDS
		src_list.txt
	COMMAND
		cmake -E copy 
			${CMAKE_CURRENT_SOURCE_DIR}/LineCount	
			${CMAKE_CURRENT_BINARY_DIR}/LineCount
	COMMAND
		cmake -E copy 
			${CMAKE_CURRENT_SOURCE_DIR}/LineReport	
			${CMAKE_CURRENT_BINARY_DIR}/LineReport
	COMMAND
		cmake -E remove ${CMAKE_CURRENT_BINARY_DIR}/Lines
	COMMAND
		cmake -E touch ${CMAKE_CURRENT_BINARY_DIR}/Lines
	COMMAND
		cmake -E cat ${CMAKE_CURRENT_BINARY_DIR}/src_list.txt
	COMMAND
		bash -c "while read line; do \
			for file in $line; do \
				${CMAKE_CURRENT_BINARY_DIR}/LineCount ${CMAKE_CURRENT_SOURCE_DIR}/$file	${CMAKE_CURRENT_BINARY_DIR}/Lines; \
			done; \
		done < '${CMAKE_CURRENT_BINARY_DIR}/src_list.txt'" 
	COMMAND
		awk -f ${CMAKE_CURRENT_BINARY_DIR}/LineReport < ${CMAKE_CURRENT_BINARY_DIR}/Lines >> ${CMAKE_CURRENT_BINARY_DIR}/Lines
	COMMAND
		cmake -E cat ${CMAKE_CURRENT_BINARY_DIR}/Lines
	COMMAND
		cmake -E remove ${CMAKE_CURRENT_BINARY_DIR}/src_list.txt
	VERBATIM
)

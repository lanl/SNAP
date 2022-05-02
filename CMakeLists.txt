################################################################################
#
# CMakeLists.txt
# 
# Author: Anthony Cabrera:
# Contact: cabreraam AT ieee DOT org
# Description: CMakeLists.txt replacement for SNAP Makefile
#
################################################################################

cmake_minimum_required(VERSION 3.18)

project(snap_test)

# Direct CMake to a specific MPI implementation. Specify
# -DMPI_EXECUTABLE_SUFFIX=.mpich for MPICH MPI implementation 
set(MPI_EXECUTABLE_SUFFIX "" CACHE STRING "Direct CMake to a specific MPI implementation")

enable_language(Fortran)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)

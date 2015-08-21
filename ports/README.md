snap-c C port of SNAP 1.0 produced by Intel Corporation
kokkos-direct "direct" Kokkos port of <code>dim3_sweep</code> produced by Geoff Womeldorff
kokkos-hp hierarchical parallelism Kokkos port of <code>dim3_sweep</code> produced by Geoff Womeldorff

Both Kokkos versions are set to build with Kokkos from Trilinos, and use an environment variable TRILINOS_DIR to identify its location. To build with the GitHub version of Kokkos, please substitute the correct headers into snappy.hpp and main.cpp.

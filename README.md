CloverLeaf
====

WIP

CloverLeaf implementation in a wide range of parallel programming models.
This implementation has support for building with and without MPI.
When MPI is enabled, all models will adjust accordingly for asynchronous MPI send/recv.

This is a consolidation of the following independent ports with a shared driver and working MPI
paths:

- <https://github.com/UoB-HPC/cloverleaf_sycl/>
- <https://github.com/UoB-HPC/cloverleaf_kokkos/>
- <https://github.com/UoB-HPC/cloverleaf_stdpar/>
- <https://github.com/UoB-HPC/cloverleaf_openmp_target/>
- <https://github.com/UoB-HPC/cloverleaf_HIP/>
- <https://github.com/UoB-HPC/cloverleaf_tbb>

## Programming Models

CloverLeaf is currently implemented in the following parallel programming models, listed in no
particular order:

- OpenMP 3 and 4.5
- C++ Parallel STL
- Kokkos
- SYCL and SYCL 2020

TODO:

- CUDA
- HIP
- OpenACC
- RAJA
- TBB
- Thrust (via CUDA or HIP)

## Building

Drivers, compiler and software applicable to whichever implementation you would like to build
against is required.

### CMake

The project supports building with CMake >= 3.13.0, which can be installed without root via
the [official script](https://cmake.org/download/).

Each BabelStream implementation (programming model) is built as follows:

```shell
$ cd CloverLeaf

# configure the build, build type defaults to Release
# The -DMODEL flag is required
$ cmake -Bbuild -H. -DMODEL=<model> -DENABLE_MPI=ON <model specific flags prefixed with -D...>

# compile
$ cmake --build build

# run executables in ./build
$ ./build/<model>-cloverleaf
```

The `MODEL` option selects one implementation of BabelStream to build.
The source for each model's implementations are located in `./src/<model>`.


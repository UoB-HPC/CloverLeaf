CloverLeaf
====

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

- CUDA
- HIP
- OpenMP
- OpenMP target
- C++ Parallel STL (StdPar)
- Kokkos >= 4
- SYCL and SYCL 2020
- OpenACC (special thanks to @pranav-sivaraman's contribution)

Planned:

- RAJA
- TBB
- Thrust (via CUDA or HIP)

## Building

Drivers, compiler and software applicable to whichever implementation you would like to build
against is required.

### CMake

The project supports building with CMake >= 3.13.0, which can be installed without root via
the [official script](https://cmake.org/download/).

Each implementation (programming model) is built as follows:

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

The `MODEL` option selects one implementation of CloverLeaf to build.
The source for each model's implementations are located in `./src/<model>`.

## Running

CloverLeaf supports the following options:

```
Usage: --help [OPTIONS]

Options:
  -h  --help                             Print this message
      --list                             List available devices with index and exit
      --device           <INDEX|NAME>    Use device at INDEX from output of --list or substring match iff INDEX is not an id
      --file,--in              <FILE>    Custom clover.in file FILE (defaults to clover.in if unspecified)
      --out                    <FILE>    Custom clover.out file FILE (defaults to clover.out if unspecified)
      --dump                    <DIR>    Dumps all field data in ASCII to ./DIR for debugging, DIR is created if missing
      --profile                          Enables kernel profiling, this takes precedence over the profiler_on in clover.in
      --staging-buffer <true|false|auto> If true, use a host staging buffer for device-host MPI halo exchange.
                                         If false, use device pointers directly for MPI halo exchange.
                                         Defaults to auto which elides the buffer if a device-aware (i.e CUDA-aware) is used.
                                         This option is no-op for CPU-only models.
                                         Setting this to false on an MPI that is not device-aware may cause a segfault.


```

For example

The output on stdout is machine-readable in YAML format where the `Output` key contains CloverLeaf
1.3's output format.
For example, here's the output
of `mpirun -np 3 kokkos_cloverleaf --device 0 --file InputDecks/clover_bm_short.in --profile true`:

```yaml
---
Devices:
  0: N6Kokkos4CudaE
CloverLeaf:
  - Ver.: 2.000
  - Deck: InputDecks/clover_bm_short.in
  - Out: clover.out
  - Profiler: true
MPI:
  - Enabled: true
  - Total ranks: 3
  - Header device-awareness (CUDA-awareness): true
  - Runtime device-awareness (CUDA-awareness): true
  - Host-Device halo exchange staging buffer: false
Model:
  - Name: Kokkos 4.0.1
  - Execution: Offload (device)
  - Backend space: N6Kokkos4CudaE
  - Backend host space: N6Kokkos6SerialE
# ---- 
Output: |+1
 Output file clover.out opened. All output will go there.
 Args: --device 0 --file InputDecks/clover_bm_short.in --profile true
 Using input: `InputDecks/clover_bm_short.in`
 Problem initialised and generated
 Launching hydro
 Step 1 time 0 control sound timestep  0.00616258 1,1 x 0 y 0
 Wall clock 0.0259612
 ...... 
 Step 86 time 0.491277 control sound timestep  0.00584781 1,1 x 0 y 0
 Wall clock 1.42524
 Average time per cell 1.79824e-08
  Step time per cell    1.69889e-08
 Step 87 time 0.497124 control sound timestep  0.005848 1,1 x 0 y 0
 Test problem 2 is within 1.17018e-11% of the expected solution
 This test is considered PASSED
 Wall clock 1.44286
 First step overhead 0

 Profiler Output        Time     Percentage
 Timestep              :0.110086 7.629754
 Ideal Gas             :0.000370 0.025662
 Viscosity             :0.001094 0.075812
 PdV                   :0.058765 4.072801
 Revert                :0.000815 0.056463
 Acceleration          :0.001175 0.081414
 Fluxes                :0.001452 0.100665
 Cell Advection        :0.001999 0.138538
 Momentum Advection    :0.003294 0.228296
 Reset                 :0.002566 0.177848
 Summary               :0.014976 1.037959
 Visit                 :0.000000 0.000000
 Tile Halo Exchange    :0.000016 0.001107
 Self Halo Exchange    :0.009350 0.648008
 MPI Halo Exchange     :1.236754 85.715627
 Total                 :1.442712 99.989953
 The Rest              :0.000145 0.010047

Result:
  - Problem: 2
  - Outcome: PASSED
```

# Licence

```
Crown Copyright 2012 AWE.
Copyright (c) 2019-24 Wei-Chen Lin, Tom Deakin, Simon McIntosh-Smith.


CloverLeaf is free software: you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the 
Free Software Foundation, either version 3 of the License, or (at your option) 
any later version.

CloverLeaf is distributed in the hope that it will be useful, but 
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
details.

You should have received a copy of the GNU General Public License along with
CloverLeaf. If not, see http://www.gnu.org/licenses/.
 ```
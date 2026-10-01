CloverLeaf
====

[![DOI](https://zenodo.org/badge/670167269.svg)](https://doi.org/10.5281/zenodo.23091924) [![CI](https://github.com/UoB-HPC/CloverLeaf/actions/workflows/linux.yml/badge.svg)](https://github.com/UoB-HPC/CloverLeaf/actions/workflows/linux.yml)

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
- TBB

Planned:
- RAJA
- Thrust (via CUDA or HIP)

## Building

Drivers, compiler and software applicable to whichever implementation you would like to build
against is required.

### CMake

The project supports building with CMake >= 3.14.0, which can be installed without root via
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

### Static builds

Build the serial model statically with:

```shell
$ cmake -Bbuild-static-serial -H. -DMODEL=serial -DENABLE_MPI=OFF \
    -DCMAKE_EXE_LINKER_FLAGS=-static
$ cmake --build build-static-serial
```

For a static GCC OpenMP build, select GCC's static OpenMP runtime explicitly:

```shell
$ CXX=g++
$ cmake -Bbuild-static-omp -H. -DMODEL=omp -DENABLE_MPI=OFF \
    -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_EXE_LINKER_FLAGS=-static \
    -DOpenMP_gomp_LIBRARY="$("$CXX" -print-file-name=libgomp.a)"
$ cmake --build build-static-omp
```

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

Environment:
  CLOVERLEAF_DEVICE                      Default for --device; the command line takes precedence.

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

## Testing

The `test_problem` option selects a reference solution check. If omitted, the check reports `SKIPPED`.
All runs also check field values, domain volume and timesteps. Failed checks return a nonzero exit status.

Run the 87-step `clover_bm16_short.in` reference case and regression tests after building:

```shell
$ ctest --test-dir build --output-on-failure
```

## Citing this work

Please cite the CloverLeaf software release in all work that uses this repository:

```bibtex
@software{lin_2026_23091925,
  author       = {Lin, Wei-Chen and
                  Deakin, Tom and
                  McIntosh-Smith, Simon and
                  Sivaraman, Pranav and
                  Alpay, Aksel},
  title        = {UoB-HPC/CloverLeaf: CloverLeaf 1.0},
  month        = oct,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.23091925},
  url          = {https://doi.org/10.5281/zenodo.23091925}
}
```

If the work depends on a particular model, also cite the corresponding paper below:

| Use | Citation |
| --- | --- |
| Any use | [CloverLeaf software release](https://doi.org/10.5281/zenodo.23091925) |
| OpenMP/MPI | [On the Performance Portability of Structured Grid Codes on Many-Core Computer Architectures](https://doi.org/10.1007/978-3-319-07518-1_4) |
| SYCL | [On measuring the maturity of SYCL implementations by tracking historical performance improvements](https://doi.org/10.1145/3456669.3456701) |
| C++ Parallel STL (StdPar) | [Evaluating ISO C++ Parallel Algorithms on Heterogeneous HPC Systems](https://doi.org/10.1109/PMBS56514.2022.00009) |
| OpenMP target | [Tracking Performance Portability on the Yellow Brick Road to Exascale](https://doi.org/10.1109/P3HPC51967.2020.00006) |
| Kokkos | [Performance Portability across Diverse Computer Architectures](https://doi.org/10.1109/P3HPC49587.2019.00006) |
| OpenACC | [Taking GPU Programming Models to Task for Performance Portability](https://doi.org/10.1145/3721145.3730423) |
| AdaptiveCpp PCUDA | [AdaptiveCpp Portable CUDA: A SYCL-Compatible CUDA Compiler for CPUs and GPUs from Multiple Vendors](https://doi.org/10.1145/3811257.3811262) |


```bibtex
@inproceedings{Lin_2024,
  author    = {Lin, Wei-Chen and Deakin, Tom and McIntosh-Smith, Simon},
  title     = {A Metric for HPC Programming Model Productivity},
  booktitle = {SC24-W: Workshops of the International Conference for High Performance Computing,
               Networking, Storage and Analysis},
  publisher = {IEEE},
  year      = {2024},
  pages     = {1192--1205},
  doi       = {10.1109/SCW63240.2024.00160}
}

@inbook{McIntosh_Smith_2014,
  author    = {McIntosh-Smith, Simon and Boulton, Michael and Curran, Dan and Price, James},
  title     = {On the Performance Portability of Structured Grid Codes on Many-Core Computer
               Architectures},
  booktitle = {Supercomputing},
  publisher = {Springer International Publishing},
  year      = {2014},
  pages     = {53--75},
  doi       = {10.1007/978-3-319-07518-1_4}
}

@inproceedings{Lin_2021,
  author    = {Lin, Wei-Chen and Deakin, Tom and McIntosh-Smith, Simon},
  title     = {On measuring the maturity of {SYCL} implementations by tracking historical
               performance improvements},
  booktitle = {International Workshop on OpenCL},
  publisher = {ACM},
  year      = {2021},
  pages     = {1--13},
  doi       = {10.1145/3456669.3456701}
}

@inproceedings{Lin_2022,
  author    = {Lin, Wei-Chen and Deakin, Tom and McIntosh-Smith, Simon},
  title     = {Evaluating {ISO C++} Parallel Algorithms on Heterogeneous {HPC} Systems},
  booktitle = {2022 IEEE/ACM International Workshop on Performance Modeling, Benchmarking and
               Simulation of High Performance Computer Systems (PMBS)},
  publisher = {IEEE},
  year      = {2022},
  pages     = {36--47},
  doi       = {10.1109/PMBS56514.2022.00009}
}

@inproceedings{Deakin_2020,
  author    = {Deakin, Tom and Poenaru, Andrei and Lin, Tom and McIntosh-Smith, Simon},
  title     = {Tracking Performance Portability on the Yellow Brick Road to Exascale},
  booktitle = {2020 IEEE/ACM International Workshop on Performance, Portability and Productivity
               in HPC (P3HPC)},
  publisher = {IEEE},
  year      = {2020},
  pages     = {1--13},
  doi       = {10.1109/P3HPC51967.2020.00006}
}

@inproceedings{Deakin_2019,
  author    = {Deakin, Tom and McIntosh-Smith, Simon and Price, James and Poenaru, Andrei and
               Atkinson, Patrick and Popa, Codrin and Salmon, Justin},
  title     = {Performance Portability across Diverse Computer Architectures},
  booktitle = {2019 IEEE/ACM International Workshop on Performance, Portability and Productivity
               in HPC (P3HPC)},
  publisher = {IEEE},
  year      = {2019},
  pages     = {1--13},
  doi       = {10.1109/P3HPC49587.2019.00006}
}

@inproceedings{Davis_2025,
  author    = {Davis, Joshua Hoke and Sivaraman, Pranav and Kitson, Joy and Parasyris, Konstantinos
               and Menon, Harshitha and Minn, Isaac and Georgakoudis, Giorgis and Bhatele, Abhinav},
  title     = {Taking {GPU} Programming Models to Task for Performance Portability},
  booktitle = {Proceedings of the 39th ACM International Conference on Supercomputing},
  publisher = {ACM},
  year      = {2025},
  pages     = {776--791},
  doi       = {10.1145/3721145.3730423}
}

@inproceedings{Alpay_2026,
  author    = {Alpay, Aksel and Heuveline, Vincent},
  title     = {{AdaptiveCpp Portable CUDA}: A {SYCL}-Compatible {CUDA} Compiler for {CPUs} and
               {GPUs} from Multiple Vendors},
  booktitle = {Proceedings of the International Workshop on OpenCL and SYCL},
  publisher = {ACM},
  year      = {2026},
  pages     = {1--12},
  doi       = {10.1145/3811257.3811262}
}
```

# Licence

```
Crown Copyright 2012 AWE.
Copyright (c) 2019-26 Wei-Chen Lin, Tom Deakin, Simon McIntosh-Smith.


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

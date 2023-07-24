#!/usr/bin/env bash

set -eu

export NVHPC_DIR="/opt/nvidia/hpc_sdk/Linux_x86_64/23.5"
export CUDA_DIR="$NVHPC_DIR/cuda/"
export PATH=$NVHPC_DIR/compilers/bin/:${PATH:-}
export KOKKOS_DIR="/home/tom/Downloads/kokkos-4.0.01/"

export CPU_RANKS=16
export GPU_RANKS=2

VERBOSE="ON"

function test() {
  rm -rf build # /CMakeCache.txt
  echo "${@:3}"
  cmake -DCMAKE_BUILD_TYPE=Release -S. -B build -GNinja -DCMAKE_VERBOSE_MAKEFILE=$VERBOSE -DENABLE_MPI=ON -DENABLE_PROFILING=ON "${@:3}" # &>/dev/null
  cmake --build build                                                                                                                    # &>/dev/null
  export OMP_PLACES=cores
  export OMP_PROC_BIND=true
  export OMP_NUM_THREADS=1
  export OMP_TARGET_OFFLOAD=MANDATORY
  # | grep -i -e "This run" -e "Timestep *" #
  export ASAN_OPTIONS=detect_leaks=0
  mkdir -p out

  #  which mpirun
  mpirun -np "$1" --tag-output -bind-to core -map-by core sh -c " build/*-cloverleaf --device "$2" --file InputDecks/clover_bm16_short.in"
  #  build/*-cloverleaf --device "$2" --file tea.in --out out/tea.out --problems tea.problems
  # konsole -e gdb -ex run --args
}

function test_nompi() {
  rm -rf build
  echo "${@:2}"
  cmake -DCMAKE_BUILD_TYPE=Release -S. -B build -GNinja -DCMAKE_VERBOSE_MAKEFILE=$VERBOSE -DENABLE_MPI=OFF -DENABLE_PROFILING=ON "${@:2}" # &>/dev/null
  cmake --build build                                                                                                                     # &>/dev/null

  export OMP_TARGET_OFFLOAD=MANDATORY
  export OMP_PLACES=cores
  export OMP_PROC_BIND=true
  export OMP_NUM_THREADS=$(nproc)
  # | grep -i -e "This run" -e "Timestep *" #
  export ASAN_OPTIONS=detect_leaks=0
  mkdir -p out
  build/*-cloverleaf --device "$1" --file InputDecks/clover_bm16_short.in
}

(
  test_nompi "0" -DMODEL=std-indices -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCXX_EXTRA_FLAGS="-Ofast" -DUSE_TBB=ON
  test_nompi "0" -DMODEL=std-indices -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCXX_EXTRA_FLAGS="-Ofast" -DUSE_ONEDPL=OPENMP
  test_nompi "0" -DMODEL=std-indices -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCXX_EXTRA_FLAGS="-Ofast" -DUSE_TBB=ON
  test_nompi "0" -DMODEL=std-indices -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCXX_EXTRA_FLAGS="-Ofast" -DUSE_ONEDPL=OPENMP
  test_nompi "0" -DMODEL=std-indices -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ -DCXX_EXTRA_FLAGS="-Ofast;-stdpar;-target=multicore"
  test_nompi "0" -DMODEL=std-indices -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ -DCXX_EXTRA_FLAGS="-Ofast;-stdpar;-gpu=cc61"
  #
  #    test_nompi "0" -DMODEL=serial -DCXX_EXTRA_FLAGS="-Ofast" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
  #    test_nompi "0" -DMODEL=serial -DCXX_EXTRA_FLAGS="-Ofast" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++

  test_nompi "0" -DMODEL=omp -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCXX_EXTRA_FLAGS="-Ofast"
  test_nompi "0" -DMODEL=omp -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ # clang fails with -Ofast, solutions if off by 0.008%
  test_nompi "0" -DMODEL=omp-target -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DOFFLOAD=NVIDIA:sm_60 -DCXX_EXTRA_FLAGS="-Ofast;--cuda-path=$CUDA_DIR"
  test_nompi "0" -DMODEL=omp-target -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ -DCXX_EXTRA_FLAGS="-Ofast"
  #
  test_nompi "0" -DMODEL=kokkos -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DKOKKOS_IN_TREE="$KOKKOS_DIR" -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_ZEN3=ON -DCXX_EXTRA_FLAGS="-Ofast"
  test_nompi "0" -DMODEL=kokkos -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DKOKKOS_IN_TREE="$KOKKOS_DIR" -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_ZEN3=ON -DCXX_EXTRA_FLAGS="-Ofast"
  test_nompi "0" -DMODEL=kokkos -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=$KOKKOS_DIR/bin/nvcc_wrapper -DKOKKOS_IN_TREE="$KOKKOS_DIR" -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_PASCAL61=ON
  test_nompi "0" -DMODEL=kokkos -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ -DKOKKOS_IN_TREE="$KOKKOS_DIR" -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_PASCAL61=ON
  #
  #  test_nompi "0" -DMODEL=cuda -DCXX_EXTRA_FLAGS="-Ofast" -DCMAKE_CUDA_COMPILER=nvcc -DCUDA_ARCH=sm_60
  #   test_nompi "0" -DCXX_EXTRA_FLAGS="-Ofast" -DMODEL=hip -DCMAKE_CXX_COMPILER=/usr/lib/aomp_17.0-1/bin/hipcc

  echo ""
)

(
  module load mpi

  test $CPU_RANKS "0" -DMODEL=std-indices -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DUSE_TBB=ON
  test $CPU_RANKS "0" -DMODEL=std-indices -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DUSE_ONEDPL=OPENMP
  test $CPU_RANKS "0" -DMODEL=std-indices -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DUSE_TBB=ON
  test $CPU_RANKS "0" -DMODEL=std-indices -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DUSE_ONEDPL=OPENMP
  test $CPU_RANKS "0" -DMODEL=std-indices -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ -DCXX_EXTRA_FLAGS="-Ofast;-stdpar;-target=multicore"
  test $GPU_RANKS "0" -DMODEL=std-indices -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ -DCXX_EXTRA_FLAGS="-Ofast;-stdpar;-gpu=cc61;--restrict"

  #  #  test $CPU_RANKS "0" -DMODEL=serial -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
  #  #  test $CPU_RANKS "0" -DMODEL=serial -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++

  test $CPU_RANKS "0" -DMODEL=omp -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
  test $CPU_RANKS "0" -DMODEL=omp -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
  test $GPU_RANKS "0" -DMODEL=omp-target -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DOFFLOAD=NVIDIA:sm_60 -DCXX_EXTRA_FLAGS="-Ofast;--cuda-path=$CUDA_DIR"
  test $CPU_RANKS "0" -DMODEL=omp-target -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ -DCXX_EXTRA_FLAGS="-Ofast" -DOFFLOAD=ON -DOFFLOAD_FLAGS="-mp;-gpu=cc60"
  #
  test $CPU_RANKS "0" -DMODEL=kokkos -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DKOKKOS_IN_TREE="$KOKKOS_DIR" -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_ZEN3=ON
  test $CPU_RANKS "0" -DMODEL=kokkos -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DKOKKOS_IN_TREE="$KOKKOS_DIR" -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_ZEN3=ON
  test $GPU_RANKS "0" -DMODEL=kokkos -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=$KOKKOS_DIR/bin/nvcc_wrapper -DKOKKOS_IN_TREE="$KOKKOS_DIR" -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_PASCAL61=ON
  test $GPU_RANKS "0" -DMODEL=kokkos -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ -DKOKKOS_IN_TREE="$KOKKOS_DIR" -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_PASCAL61=ON

  #  test $GPU_RANKS "0" -DMODEL=cuda -DCMAKE_CUDA_COMPILER=nvcc -DCUDA_ARCH=sm_60
  #  test $GPU_RANKS "0" -DMODEL=hip -DCMAKE_CXX_COMPILER=/usr/lib/aomp_17.0-1/bin/hipcc

  #  module use /opt/nvidia/hpc_sdk/modulefiles
  #  module load nvhpc-nompi
  #  export LD_LIBRARY_PATH=${NVHPC_DIR}/cuda/lib64:${LD_LIBRARY_PATH:-}

)
##
(
  set +eu
  #  #  source /opt/intel/oneapi/setvars.sh
  #  #  source  /opt/intel/oneapi/mpi/2021.9.0/env/vars.sh
  module load mpi
  source /opt/intel/oneapi/tbb/2021.10.0/env/vars.sh
  source /opt/intel/oneapi/compiler/2023.2.0/env/vars.sh
  set -eu
  export DPCPP_CPU_NUM_CUS=1
  export DPCPP_CPU_SCHEDULE=static
  test $CPU_RANKS "AMD" -DMODEL=sycl-acc -DSYCL_COMPILER=ONEAPI-ICPX
  test $CPU_RANKS "AMD" -DMODEL=sycl-usm -DSYCL_COMPILER=ONEAPI-ICPX
)

(
  set +eu
  #source /opt/intel/oneapi/setvars.sh --include-intel-llvm
  module load mpi
  source /opt/intel/oneapi/tbb/2021.10.0/env/vars.sh
  source /opt/intel/oneapi/compiler/2023.2.0/env/vars.sh --include-intel-llvm
  set -eu

  test $GPU_RANKS "0" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DMODEL=std-indices -DUSE_ONEDPL=DPCPP -DCXX_EXTRA_FLAGS="-fsycl;-fsycl-targets=nvptx64-nvidia-cuda;-Xsycl-target-backend;--cuda-gpu-arch=sm_60;--cuda-path=$CUDA_DIR"

  cuda_sycl_flags="-fsycl-targets=nvptx64-nvidia-cuda;--cuda-path=$CUDA_DIR;-Xsycl-target-backend;--cuda-gpu-arch=sm_60"
  test $GPU_RANKS "NVIDIA" -DMODEL=sycl-acc -DSYCL_COMPILER=ONEAPI-Clang -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2023.1.0/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$cuda_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$cuda_sycl_flags"
  test $GPU_RANKS "NVIDIA" -DMODEL=sycl-usm -DSYCL_COMPILER=ONEAPI-Clang -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2023.1.0/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$cuda_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$cuda_sycl_flags"

  export LD_LIBRARY_PATH=/opt/rocm-5.4.3/lib:${LD_LIBRARY_PATH:-}
  hip_sycl_flags="-fsycl;-fsycl-targets=amdgcn-amd-amdhsa;-Xsycl-target-backend;--offload-arch=gfx1012"
  test $GPU_RANKS "Radeon" -DMODEL=sycl-acc -DSYCL_COMPILER=ONEAPI-Clang -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2023.1.0/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$hip_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$hip_sycl_flags"
  test $GPU_RANKS "Radeon" -DMODEL=sycl-usm -DSYCL_COMPILER=ONEAPI-Clang -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2023.1.0/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$hip_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$hip_sycl_flags"
)

#!/usr/bin/env bash
set -euo pipefail

apt-get update
apt-get install -y --no-install-recommends cmake ninja-build g++ curl ca-certificates gpg

if [[ -n ${CI_APT_SOURCE:-} ]]; then
  curl --fail --silent --show-error --location --retry 3 --retry-connrefused "$CI_APT_KEY" |
    gpg --dearmor --yes -o /usr/share/keyrings/cloverleaf-toolchain.gpg
  printf 'deb [signed-by=/usr/share/keyrings/cloverleaf-toolchain.gpg] %s\n' \
    "$CI_APT_SOURCE" > /etc/apt/sources.list.d/cloverleaf-toolchain.list
  # XXX Keep vendor packages together (Ubuntu's hipcc has a higher version number)
  repo_host=${CI_APT_SOURCE#*://}
  repo_host=${repo_host%%/*}
  printf 'Package: *\nPin: origin %s\nPin-Priority: 600\n' "$repo_host" \
    > /etc/apt/preferences.d/cloverleaf-toolchain
  apt-get update
fi

read -r -a packages <<< "$CI_PACKAGES"
if ((${#packages[@]})); then
  apt-get install -y --no-install-recommends "${packages[@]}"
fi

if [[ -n ${CI_MPICH_SOURCE_URL:-} ]]; then
  # XXX Ubuntu's PMIx library and Hydra launcher can silently run as singletons
  mpi_build=$(mktemp -d)
  curl --fail --silent --show-error --location --retry 3 \
    "$CI_MPICH_SOURCE_URL" \
    -o "$mpi_build/mpich.tar.gz"
  echo "$CI_MPICH_SHA256" \
    " $mpi_build/mpich.tar.gz" | sha256sum --check
  tar -xzf "$mpi_build/mpich.tar.gz" -C "$mpi_build" --strip-components=1
  (
    cd "$mpi_build"
    CC=gcc CXX=g++ ./configure --prefix=/opt/mpich --with-device=ch3:sock \
      --with-pm=hydra --without-hwloc --disable-fortran --disable-cxx --disable-romio
    make -j "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
    make install
  )
  rm -rf "$mpi_build"
fi

if [[ -n ${CI_KOKKOS_REF:-} ]]; then
  git clone --depth 1 --branch "$CI_KOKKOS_REF" https://github.com/kokkos/kokkos.git /opt/kokkos
fi

if [[ -n ${CI_ADAPTIVECPP_SOURCE_URL:-} ]]; then
  acpp_build=$(mktemp -d)
  curl --fail --silent --show-error --location --retry 3 \
    "$CI_ADAPTIVECPP_SOURCE_URL" -o "$acpp_build/source.tar.gz"
  echo "$CI_ADAPTIVECPP_SHA256" " $acpp_build/source.tar.gz" | sha256sum --check
  tar -xzf "$acpp_build/source.tar.gz" -C "$acpp_build" --strip-components=1
  cmake -S "$acpp_build" -B "$acpp_build/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/adaptivecpp \
    "-DCMAKE_C_COMPILER=$CC" "-DCMAKE_CXX_COMPILER=$CXX" \
    -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm -DCLANG_EXECUTABLE_PATH=/usr/bin/clang++-18 \
    -DWITH_CUDA_BACKEND=OFF -DWITH_ROCM_BACKEND=OFF \
    -DWITH_OPENCL_BACKEND=OFF -DWITH_LEVEL_ZERO_BACKEND=OFF
  cmake --build "$acpp_build/build" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
  cmake --install "$acpp_build/build"
  rm -rf "$acpp_build"
fi

"$CC" --version
"$CXX" --version
cmake --version

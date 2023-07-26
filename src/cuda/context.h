/*
 Crown Copyright 2012 AWE.

 This file is part of CloverLeaf.

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
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "shared.h"

#define CLOVER_DEFAULT_BLOCK_SIZE (256)
#define DEVICE_KERNEL __host__ __device__

// #define CLOVER_SYNC_ALL_KERNELS
// #define CLOVER_MANAGED_ALLOC

#ifdef CLOVER_MANAGED_ALLOC
  #define CLOVER_MEMCPY_KIND_D2H (cudaMemcpyDefault)
  #define CLOVER_MEMCPY_KIND_H2D (cudaMemcpyDefault)
#else
  #define CLOVER_MEMCPY_KIND_D2H (cudaMemcpyDeviceToHost)
  #define CLOVER_MEMCPY_KIND_H2D (cudaMemcpyHostToDevice)
#endif

#ifdef CLOVER_SYNC_ALL_KERNELS
  #define CLOVER_BUILTIN_FILE __builtin_FILE()
  #define CLOVER_BUILTIN_LINE __builtin_LINE()
#else
  #define CLOVER_BUILTIN_FILE ("")
  #define CLOVER_BUILTIN_LINE (0)
#endif

namespace clover {

struct chunk_context {};
struct context {};

// Generic error checking for when callsite is async or unimportant
static inline void checkError(const cudaError_t err = cudaGetLastError()) {
  if (err != cudaSuccess) {
    std::cerr << std::string(cudaGetErrorName(err)) + ": " + std::string(cudaGetErrorString(err)) << std::endl;
    std::abort();
  }
}

template <typename T> static inline T *alloc(size_t count) {
  void *p{};
#ifdef CLOVER_MANAGED_ALLOC
  auto result = cudaMallocManaged(&p, count * sizeof(T));
#else
  auto result = cudaMalloc(&p, count * sizeof(T));
#endif
  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate memory of " << count << " bytes: " << cudaGetErrorString(result) << std::endl;
    std::abort();
  }
  return static_cast<T *>(p);
}

static inline void dealloc(void *p) {
  if (auto result = cudaFree(p); result != cudaSuccess) {
    std::cerr << "Failed to deallocate " << p << ": " << cudaGetErrorString(result) << std::endl;
    std::abort();
  }
}

template <typename T> struct Buffer1D {
  size_t size;
  T *data;
  Buffer1D(context &, size_t size) : size(size), data(alloc<T>(size)) {}
  Buffer1D(context &, size_t size, T *host_init) : size(size), data(alloc<T>(size)) {
    if (auto result = cudaMemcpy(data, host_init, (sizeof(T) * size), CLOVER_MEMCPY_KIND_H2D); result != cudaSuccess) {
      std::cerr << "Buffer1D cudaMemcpy failed:"
                << ": " << cudaGetErrorString(result) << std::endl;
      std::abort();
    }
  }
  // XXX the following ctors break oneDPL, making this class not device_copyable
  //  Buffer1D(Buffer1D &&other) noexcept : size(other.size), data(std::exchange(other.data, nullptr)) {}
  //  Buffer1D(const Buffer1D<T> &that) : size(that.size), data(that.data) {}
  // XXX this model captures this class by value, the dtor is called for each lambda scope which is wrong!
  //  ~Buffer1D() { std::free(data); }

  void release() { dealloc(data); }

  __host__ __device__ T &operator[](size_t i) const { return data[i]; }
  T *actual() { return data; }

  template <size_t D> [[nodiscard]] size_t extent() const {
    static_assert(D < 1);
    return size;
  }

  std::vector<T> mirrored() const {
    std::vector<T> buffer(size);
    if (auto result = cudaMemcpy(buffer.data(), data, buffer.size() * sizeof(T), CLOVER_MEMCPY_KIND_D2H); result != cudaSuccess) {
      std::cerr << "cudaMemcpy failed:"
                << ": " << cudaGetErrorString(result) << std::endl;
      std::abort();
    }
    return buffer;
  }
};

template <typename T> struct Buffer2D {
  size_t sizeX, sizeY;
  T *data;
  Buffer2D(context &, size_t sizeX, size_t sizeY) : sizeX(sizeX), sizeY(sizeY), data(alloc<T>(sizeX * sizeY)) {}
  // XXX the following ctors break oneDPL, making this class not device_copyable
  //  Buffer2D(Buffer2D &&other) noexcept : sizeX(other.sizeX), sizeY(other.sizeY), data(std::exchange(other.data, nullptr)) {}
  //  Buffer2D(const Buffer2D<T> &that) : sizeX(that.sizeX), sizeY(that.sizeY), data(that.data) {}
  // XXX this model captures this class by value, the dtor is called for each lambda scope which is wrong!
  // ~Buffer2D() { std::free(data); }

  void release() { dealloc(data); }

  __host__ __device__ T &operator()(size_t i, size_t j) const { return data[i + j * sizeX]; }
  T *actual() { return data; }

  template <size_t D> [[nodiscard]] size_t extent() const {
    if constexpr (D == 0) {
      return sizeX;
    } else if (D == 1) {
      return sizeY;
    } else {
      static_assert(D < 2);
    }
  }

  std::vector<T> mirrored() const {
    std::vector<T> buffer(sizeX * sizeY);
    if (auto result = cudaMemcpy(buffer.data(), data, buffer.size() * sizeof(T), CLOVER_MEMCPY_KIND_D2H); result != cudaSuccess) {
      std::cerr << "cudaMemcpy failed:"
                << ": " << cudaGetErrorString(result) << std::endl;
      std::abort();
    }
    return buffer;
  }
  clover::BufferMirror2D<T> mirrored2() { return {mirrored(), extent<0>(), extent<1>()}; }
};
template <typename T> using StagingBuffer1D = T*;

template <typename F> __global__ void par_reduce_kernel(F functor) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  functor(gid);
}

template <size_t THREADS, size_t BLOCK, typename F>
static void par_reduce(const F functor, const char *file = CLOVER_BUILTIN_FILE, int loc = CLOVER_BUILTIN_LINE) {
  par_reduce_kernel<F><<<THREADS, BLOCK>>>(functor);
#ifdef CLOVER_SYNC_ALL_KERNELS
  if (auto result = cudaDeviceSynchronize(); result != cudaSuccess) {
    std::cerr << "Reduce kernel at " << file << ":" << loc << " failed: " << cudaGetErrorString(result) << std::endl;
  }
#endif
}

template <typename F> __global__ void par_ranged1d_kernel(Range1d r, F functor) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= r.size) return;
  functor(r.from + gid);
}

template <size_t BLOCK = CLOVER_DEFAULT_BLOCK_SIZE, typename F>
static void par_ranged1(const Range1d &r, const F functor, const char *file = CLOVER_BUILTIN_FILE, int loc = CLOVER_BUILTIN_LINE) {
  int blocks = r.size < BLOCK ? 1 : BLOCK;
  int threads = std::ceil(static_cast<double>(r.size) / blocks);
  par_ranged1d_kernel<F><<<threads, blocks>>>(r, functor);
#ifdef CLOVER_SYNC_ALL_KERNELS
  if (auto result = cudaDeviceSynchronize(); result != cudaSuccess) {
    std::cerr << "1D kernel at " << file << ":" << loc << " failed: " << cudaGetErrorString(result) << std::endl;
  }
#endif
}

template <typename F> __global__ void par_ranged2d_kernel(Range2d r, F functor) {
  // linearise because of limits (65536) on the second and third dimension
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= r.sizeX * r.sizeY) return;
  const auto x = r.fromX + (gid % r.sizeX);
  const auto y = r.fromY + (gid / r.sizeX);
  functor(x, y);
}

template <size_t BLOCK = CLOVER_DEFAULT_BLOCK_SIZE, typename F>
static void par_ranged2(const Range2d &r, F functor, const char *file = CLOVER_BUILTIN_FILE, int loc = CLOVER_BUILTIN_LINE) {
  int blocks = r.sizeX * r.sizeY < BLOCK ? 1 : BLOCK;
  int threads = std::ceil(static_cast<double>(r.sizeX * r.sizeY) / blocks);
  par_ranged2d_kernel<F><<<threads, blocks>>>(r, functor);
#ifdef CLOVER_SYNC_ALL_KERNELS
  if (auto result = cudaDeviceSynchronize(); result != cudaSuccess) {
    std::cerr << "2D kernel at " << file << ":" << loc << " failed: " << cudaGetErrorString(result) << std::endl;
  }
#endif
}

template <typename T, int offset> struct reduce {
  __device__ inline static void run(T *array, T *out, T (*func)(T, T)) {
    if (offset > 16) __syncthreads(); // only need to sync if not working within a warp
    if (threadIdx.x < offset) {       // only continue if it's in the lower half
      array[threadIdx.x] = func(array[threadIdx.x], array[threadIdx.x + offset]);
      reduce<T, offset / 2>::run(array, out, func);
    }
  }
};

template <typename T> struct reduce<T, 0> {
  __device__ inline static void run(T *array, T *out, T (*)(T, T)) { out[blockIdx.x] = array[0]; }
};

} // namespace clover

using clover::Range1d;
using clover::Range2d;

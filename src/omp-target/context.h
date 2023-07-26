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

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "shared.h"

#define SYCL_DEBUG   // enable for debugging SYCL related things, also syncs kernel calls
#define SYNC_KERNELS // enable for fully synchronous (e.g queue.wait_and_throw()) kernel calls
#define USE_COND_TARGET
#ifdef USE_COND_TARGET
  #define clover_use_target(cond) if (target : (cond))
#else
  #define clover_use_target(cond) /*no-op*/
#endif

namespace clover {

struct context {
  bool use_target;
};

template <typename T> static inline T *alloc(size_t count) { return static_cast<T *>(std::malloc(count * sizeof(T))); }

template <typename T> struct Buffer1D {
  size_t size;
  T *data;
  Buffer1D(context &, size_t size) : size(size), data(alloc<T>(size)) {}
  Buffer1D(Buffer1D &&other) noexcept : size(other.size), data(std::exchange(other.data, nullptr)) {}
  Buffer1D(const Buffer1D<T> &that) : size(that.size), data(that.data) {}
  ~Buffer1D() { std::free(data); }

  T &operator[](size_t i) const { return data[i]; }
  [[nodiscard]] constexpr size_t N() const { return size; }
  T *actual() { return data; }

  // alternatively, replace both assignment operators with
  Buffer1D<T> &operator=(Buffer1D<T> other) noexcept {
    std::swap(data, other.data);
    std::swap(size, other.size);
    return *this;
  }

  //  Buffer1D &operator=(Buffer1D &&other) noexcept {
  //    size = other.size;
  //    std::swap(data, other.data);
  //    return *this;
  //  }
  //  Buffer1D<T> &operator=(const Buffer1D<T> &other) {
  //    if (this != &other) {
  //      delete[] data;
  //      std::copy(other.data, other.data + size, data);
  //      size = other.size;
  //    }
  //    return *this;
  //  }

  template <size_t D> [[nodiscard]] size_t extent() const {
    static_assert(D < 1);
    return size;
  }

  std::vector<T> mirrored() const {
    std::vector<T> buffer(size);
    std::copy(data, data + buffer.size(), buffer.begin());
    return buffer;
  }
};

template <typename T> struct Buffer2D {
  size_t sizeX, sizeY;
  T *data;
  Buffer2D(context &, size_t sizeX, size_t sizeY) : sizeX(sizeX), sizeY(sizeY), data(alloc<T>(sizeX * sizeY)) {}
  Buffer2D(Buffer2D &&other) noexcept : sizeX(other.sizeX), sizeY(other.sizeY), data(std::exchange(other.data, nullptr)) {}
  Buffer2D(const Buffer2D<T> &that) : sizeX(that.sizeX), sizeY(that.sizeY), data(that.data) {}
  ~Buffer2D() { std::free(data); }

  T &operator()(size_t i, size_t j) const { return data[j + i * sizeY]; }
  [[nodiscard]] constexpr size_t N() const { return sizeX * sizeY; }
  [[nodiscard]] constexpr size_t nX() const { return sizeX; }
  [[nodiscard]] constexpr size_t nY() const { return sizeY; }
  T *actual() { return data; }

  Buffer2D<T> &operator=(Buffer2D<T> other) noexcept {
    std::swap(data, other.data);
    std::swap(sizeX, other.sizeX);
    std::swap(sizeY, other.sizeY);
    return *this;
  }

  //  Buffer2D<T> &operator=(const Buffer2D<T> &other) {
  //    if (this != &other) {
  //      return *this = Buffer2D(other);
  //    }
  //  }
  //
  //  Buffer2D &operator=(Buffer2D &&other) noexcept {
  //    sizeX = other.sizeX;
  //    sizeY = other.sizeY;
  //    std::swap(data, other.data);
  //    return *this;
  //  }

  template <size_t D> [[nodiscard]] size_t extent() const {
    if constexpr (D == 0) {
      return sizeX;
    } else if (D == 1) {
      return sizeY;
    } else {
      static_assert(D < 2);
      return 0;
    }
  }

  std::vector<T> mirrored() const {
    std::vector<T> buffer(sizeX * sizeY);
    std::copy(data, data + buffer.size(), buffer.begin());
    return buffer;
  }
  clover::BufferMirror2D<T> mirrored2() { return {mirrored(), extent<0>(), extent<1>()}; }
};
template <typename T> using StagingBuffer1D = T *;

struct chunk_context {};

} // namespace clover

using clover::Range1d;
using clover::Range2d;

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

namespace clover {

struct context {};

template <typename T> static inline T *alloc(size_t count) { return static_cast<T *>(std::malloc(count * sizeof(T))); }

template <typename T> struct Buffer1D {
  size_t size;
  T *data;
  Buffer1D(context &, size_t size) : size(size), data(alloc<T>(size)) {}
  Buffer1D(Buffer1D &&other) noexcept : size(other.size), data(std::exchange(other.data, nullptr)) {}
  Buffer1D(const Buffer1D<T> &that) : size(that.size), data(that.data) {}
  ~Buffer1D() { std::free(data); }

  T &operator[](size_t i) const { return data[i]; }
  T *actual() { return data; }

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

  T &operator()(size_t i, size_t j) const { return data[i + j * sizeX]; }
  T *actual() { return data; }

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
template <typename T> using StagingBuffer1D = Buffer1D<T> &;

struct chunk_context {};

} // namespace clover

using clover::Range1d;
using clover::Range2d;

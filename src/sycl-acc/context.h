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

#include <CL/sycl.hpp>
#include <vector>

#include "shared.h"

using namespace cl;

using sycl::accessor;
using sycl::buffer;
using sycl::handler;
using sycl::id;
using sycl::queue;
using sycl::range;

constexpr sycl::access::mode R = sycl::access::mode::read;
constexpr sycl::access::mode W = sycl::access::mode::write;
constexpr sycl::access::mode RW = sycl::access::mode::read_write;

// #define SYCL_DEBUG // enable for debugging SYCL related things, also syncs kernel calls
// #define SYNC_KERNELS // enable for fully synchronous (e.g queue.wait_and_throw()) kernel calls

namespace clover {

// abstracts away sycl::accessor
template <typename T, int N, sycl::access::mode mode> struct Accessor {
  typedef sycl::accessor<T, N, mode, sycl::access::target::device> Type;
  //  typedef sycl::accessor<T, N, mode, sycl::access::target::host_buffer> HostType;

  inline static Type from(sycl::buffer<T, N> &b, sycl::handler &cgh) {
    return b.template get_access<mode, sycl::access::target::device>(cgh);
  }

  inline static Type from(sycl::buffer<T, N> &b, sycl::handler &cgh, sycl::range<N> accessRange, sycl::id<N> accessOffset) {
    return b.template get_access<mode, sycl::access::target::device>(cgh, accessRange, accessOffset);
  }

  inline static auto access_host(sycl::buffer<T, N> &b) {
    return b.get_host_access();

    //    return b.template get_access<mode>();
  }
};

struct chunk_context {};
struct context {
  sycl::queue queue;
};

template <typename T> struct Buffer1D {

  sycl::buffer<T, 1> buffer;

  // delegates to the corresponding buffer constructor
  explicit Buffer1D(clover::context, size_t x) : buffer(sycl::range{x}) {}

  explicit Buffer1D(T *src, sycl::range<1> range) : buffer(src, range) {}

  // delegates to the corresponding buffer constructor
  template <typename Iterator> explicit Buffer1D(Iterator begin, Iterator end) : buffer(begin, end) {}

  // delegates to accessor.get_access<mode>(handler)
  template <sycl::access::mode mode> inline typename Accessor<T, 1, mode>::Type access(sycl::handler &cgh) {
    return Accessor<T, 1, mode>::from(buffer, cgh);
  }

  // delegates to accessor.get_access<mode>(handler)
  template <sycl::access::mode mode>
  inline typename Accessor<T, 1, mode>::Type access(sycl::handler &cgh, sycl::range<1> accessRange, sycl::id<1> accessOffset) {
    return Accessor<T, 1, mode>::from(buffer, cgh, accessRange, accessOffset);
  }

  // delegates to accessor.get_access<mode>()
  // **for host buffers only**
  inline auto access() { return buffer.get_host_access(); }

  template <sycl::access::mode mode> inline auto access_ptr(size_t count) {

    return sycl::host_accessor<T, 1, mode>{buffer, count}.get_pointer();

//    return buffer.get_host_access(count).get_pointer();
//    return buffer.template get_access<mode>(count).get_pointer();
  }

  template <size_t D> [[nodiscard]] size_t extent() const {
    static_assert(D < 1);
    return buffer.size();
  }

  std::vector<T> mirrored() {
    std::vector<T> out(buffer.size());
    auto data = buffer.get_host_access().get_pointer();
    std::copy(data, data + out.size(), out.begin());
    return out;
  }

};

template <typename T> struct Buffer2D {

  sycl::buffer<T, 2> buffer;

  // delegates to the corresponding buffer constructor
  explicit Buffer2D(clover::context, size_t x, size_t y) : buffer(sycl::range{x, y}) {}

  explicit Buffer2D(T *src, sycl::range<2> range) : buffer(src, range) {}

  // delegates to the corresponding buffer constructor
  template <typename Iterator> explicit Buffer2D(Iterator begin, Iterator end) : buffer(begin, end) {}

  // delegates to accessor.get_access<mode>(handler)
  template <sycl::access::mode mode> inline typename Accessor<T, 2, mode>::Type access(sycl::handler &cgh) {
    return Accessor<T, 2, mode>::from(buffer, cgh);
  }

  // delegates to accessor.get_access<mode>(handler)
  template <sycl::access::mode mode>
  inline typename Accessor<T, 2, mode>::Type access(sycl::handler &cgh, sycl::range<2> accessRange, sycl::id<2> accessOffset) {
    return Accessor<T, 2, mode>::from(buffer, cgh, accessRange, accessOffset);
  }

  // delegates to accessor.get_access<mode>()
  // **for host buffers only**
  inline auto access() { return buffer.get_host_access(); }

  template <size_t D> [[nodiscard]] size_t extent() const {
    if constexpr (D == 0) {
      return buffer.get_range()[0];
    } else if (D == 1) {
      return buffer.get_range()[1];
    } else {
      static_assert(D < 2);
    }
  }

  std::vector<T> mirrored() {
    std::vector<T> out(buffer.size());
    auto data = buffer.get_host_access().get_pointer();
    std::copy(data, data + out.size(), out.begin());
    return out;
  }
  clover::BufferMirror2D<T> mirrored2() { return {mirrored(), extent<0>(), extent<1>()}; }
};
template <typename T> using StagingBuffer1D = Buffer1D<T>&;

// safely offset an id<2> by j and k
static inline sycl::id<2> offset(const sycl::id<2> idx, const int j, const int k) {
  int jj = static_cast<int>(idx[0]) + j;
  int kk = static_cast<int>(idx[1]) + k;
#ifdef SYCL_DEBUG
  // XXX only use on runtime that provides assertions, eg: CPU
  assert(jj >= 0);
  assert(kk >= 0);
#endif
  return sycl::id<2>(jj, kk);
}

// delegates to parallel_for, handles flipping if enabled
template <typename nameT, class functorT> static inline void par_ranged(sycl::handler &cgh, const Range1d &range, functorT functor) {
  cgh.parallel_for<nameT>(sycl::range<1>(range.size), [=](sycl::id<1> idx) {
    idx = sycl::id<1>(idx.get(0) + range.from);
    functor(idx);
  });
}

// delegates to parallel_for, handles flipping if enabled
template <typename nameT, class functorT> static inline void par_ranged(sycl::handler &cgh, const Range2d &range, functorT functor) {

#define RANGE2D_NORMAL 0x01
#define RANGE2D_LINEAR 0x02
#define RANGE2D_ROUND 0x04

#ifndef RANGE2D_MODE
  #error "RANGE2D_MODE not set"
#endif

#if RANGE2D_MODE == RANGE2D_NORMAL
  cgh.parallel_for<nameT>(sycl::range<2>(range.sizeX, range.sizeY), [=](sycl::id<2> idx) {
    idx = sycl::id<2>(idx[0] + range.fromX, idx[1] + range.fromY);
    functor(idx);
  });
#elif RANGE2D_MODE == RANGE2D_LINEAR
  cgh.parallel_for<nameT>(sycl::range<1>(range.sizeX * range.sizeY), [=](sycl::id<1> id) {
    const auto x = (id[0] / range.sizeY) + range.fromX;
    const auto y = (id[0] % range.sizeY) + range.fromY;
    functor(sycl::id<2>(x, y));
  });
#elif RANGE2D_MODE == RANGE2D_ROUND
  const size_t minBlockSize = 32;
  const size_t roundedX = range.sizeX % minBlockSize == 0 ? range.sizeX //
                                                          : ((range.sizeX + minBlockSize - 1) / minBlockSize) * minBlockSize;
  const size_t roundedY = range.sizeY % minBlockSize == 0 ? range.sizeY //
                                                          : ((range.sizeY + minBlockSize - 1) / minBlockSize) * minBlockSize;
  cgh.parallel_for<nameT>(sycl::range<2>(roundedX, roundedY), [=](sycl::id<2> idx) {
    if (idx[0] >= range.sizeX) return;
    if (idx[1] >= range.sizeY) return;
    idx = sycl::id<2>(idx[0] + range.fromX, idx[1] + range.fromY);
    functor(idx);
  });
#else
  #error "Unsupported RANGE2D_MODE"
#endif
}

// delegates to queue.submit(cgf), handles blocking submission if enable
template <typename T> static void execute(sycl::queue &queue, T cgf) {
  try {
    queue.submit(cgf);
#if defined(SYCL_DEBUG) || defined(SYNC_KERNELS)
    queue.wait_and_throw();
#endif
  } catch (sycl::exception &e) {
    std::cerr << "[SYCL] Exception : `" << e.what() << "`" << std::endl;
    throw e;
  }
}
} // namespace clover

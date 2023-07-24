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

#include "context.h"
#include <CL/sycl.hpp>
#include <iostream>
#include <utility>

using namespace cl;

namespace clover {

template <typename T, typename U, typename C> struct local_reducer {

  sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local> local;

  C actual;
  sycl::accessor<U, 1, sycl::access::mode::read_write, sycl::access::target::device> result;

  local_reducer(sycl::handler &h, size_t size, C actual, sycl::buffer<U, 1> &b)
      : local(sycl::range<1>(size), h), actual(actual), result(b.template get_access<sycl::access::mode::read_write>(h)) {}

  inline void drain(sycl::id<1> lid, sycl::id<1> gid) const { local[lid] = result[gid]; }
};

template <typename nameT, size_t dimension, class RangeTpe, class LocalType, class LocalAllocator = std::nullptr_t,
          class Empty = std::nullptr_t, class Functor = std::nullptr_t, class BinaryOp = std::nullptr_t, class Finaliser = std::nullptr_t,
          class RangeLengthFn = std::nullptr_t, class RangeIdFn = std::nullptr_t>
static inline void par_reduce_nd_impl(sycl::queue &q, RangeTpe range, RangeLengthFn lengthFn, RangeIdFn rangeIdFn, LocalAllocator allocator,
                                      Empty empty, Functor functor, BinaryOp combiner, Finaliser finaliser) {

  auto dev = q.get_device();

  size_t dot_num_groups;
  size_t dot_wgsize;
  if (dev.is_cpu()) {
    dot_num_groups = dev.get_info<sycl::info::device::max_compute_units>();
    dot_wgsize = dev.get_info<sycl::info::device::native_vector_width_double>() * 2;

  } else {
    dot_num_groups = dev.get_info<sycl::info::device::max_compute_units>() * 4;
    // TODO max wg size is 1024 on NV but it fails with an CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES with used directly.
    //  The same works on HIP so something isn't quite right here. Halving it seems to workaround this but not ideal.
    //  This issue *appears* to be independent of the local memory size. Might be a bug in CUDA PI.
    dot_wgsize = dev.get_info<sycl::info::device::max_work_group_size>() / 2;
  }

  size_t N = lengthFn(range);
  dot_num_groups = std::min(N, dot_num_groups);

#ifdef SYCL_DEBUG
  std::cout << "RD: dot_wgsize=" << dot_wgsize << " dot_num_groups:" << dot_num_groups << " N=" << N << "\n";
#endif

  q.submit([=](sycl::handler &h) mutable {
    auto ctx = allocator(h, dot_wgsize);
    h.parallel_for<nameT>(sycl::nd_range<1>(dot_num_groups * dot_wgsize, dot_wgsize), [=](sycl::nd_item<1> item) {
      size_t i = item.get_global_id(0);
      size_t li = item.get_local_id(0);
      size_t global_size = item.get_global_range()[0];

      empty(ctx, sycl::id<1>(li));
      for (; i < N; i += global_size) {
        functor(ctx, sycl::id<1>(li), rangeIdFn(sycl::id<1>(i), range));
      }

      size_t local_size = item.get_local_range()[0]; // 8
      for (size_t offset = local_size / 2; offset > 0; offset /= 2) {
        item.barrier(sycl::access::fence_space::local_space);
        if (li < offset) {
          combiner(ctx, sycl::id<1>(li), sycl::id<1>(li + offset));
        }
      }

      if (li == 0) {
        finaliser(ctx, item.get_group(0), sycl::id<1>(0));
      }
    });
  });

  q.submit([=](sycl::handler &h) mutable {
    auto ctx = allocator(h, dot_num_groups);
    h.parallel_for<class final_reduction>(sycl::nd_range<1>(1, 1), [=](auto) {
      auto zero = sycl::id<1>(0);
      empty(ctx, zero); // local[0] = empty
      for (size_t i = 0; i < dot_num_groups; ++i) {
        ctx.drain(sycl::id<1>(i), sycl::id<1>(i));
      }
      for (size_t i = 1; i < dot_num_groups; ++i) {
        combiner(ctx, zero, sycl::id<1>(i)); // local[0] = local[0] |+| xs[i]
      }
      finaliser(ctx, 0, zero); // xs[0] = local[0]
    });
  });
#ifdef SYNC_KERNELS
  q.wait_and_throw();
#endif
#ifdef SYCL_DEBUG
  std::cout << "RD: done= " << N << "\n";
#endif
}

template <typename nameT, class LocalType, class LocalAllocator = std::nullptr_t, class Empty = std::nullptr_t,
          class Functor = std::nullptr_t, class BinaryOp = std::nullptr_t, class Finaliser = std::nullptr_t>
static inline void par_reduce_2d(sycl::queue &q, const clover::Range2d &range, LocalAllocator allocator, Empty empty, Functor functor,
                                 BinaryOp combiner, Finaliser finaliser) {
#ifdef SYCL_DEBUG
  std::cout << "par_reduce_2d " << range << "\n";
#endif
  par_reduce_nd_impl<nameT, 2, clover::Range2d, LocalType, LocalAllocator, Empty, Functor, BinaryOp, Finaliser>(
      q, range, [](clover::Range2d r) { return r.sizeX * r.sizeY; },
      [](sycl::id<1> gid, clover::Range2d r) {
        const auto x = (gid[0] / r.sizeY) + r.fromX;
        const auto y = (gid[0] % r.sizeY) + r.fromY;
        return sycl::id<2>(x, y);
      },
      allocator, empty, functor, combiner, finaliser);
}

// applies a 1d reduction
template <typename nameT, class LocalType, class LocalAllocator = std::nullptr_t, class Empty = std::nullptr_t,
          class Functor = std::nullptr_t, class BinaryOp = std::nullptr_t, class Finaliser = std::nullptr_t>
static inline void par_reduce_1d(sycl::queue &q, const clover::Range1d &range, LocalAllocator allocator, Empty empty, Functor functor,
                                 BinaryOp combiner, Finaliser finaliser) {
#ifdef SYCL_DEBUG
  std::cout << "par_reduce_1d " << range << "\n";
#endif
  par_reduce_nd_impl<nameT, 1, clover::Range1d, LocalType, LocalAllocator, Empty, Functor, BinaryOp, Finaliser>(
      q, range, [](clover::Range1d r) { return r.size; }, [](sycl::id<1> gid, clover::Range1d r) { return r.from + gid[0]; }, allocator,
      empty, functor, combiner, finaliser);
}

} // namespace clover

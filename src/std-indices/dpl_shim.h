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

#include <cstddef>
#include <cstdlib>

#ifdef USE_ONEDPL

// oneDPL C++17 PSTL

  #include <oneapi/dpl/algorithm>
  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/numeric>

  #if ONEDPL_USE_DPCPP_BACKEND

    #include <CL/sycl.hpp>

const static auto EXEC_POLICY =
    oneapi::dpl::execution::device_policy<>{oneapi::dpl::execution::make_device_policy(cl::sycl::device{cl::sycl::default_selector_v})};

template <typename T> T *alloc_raw(size_t size) { return sycl::malloc_shared<T>(size, EXEC_POLICY.queue()); }

template <typename T> void dealloc_raw(T *ptr) { sycl::free(ptr, EXEC_POLICY.queue()); }

  #else

// auto exe_policy = dpl::execution::seq;
// auto exe_policy = dpl::execution::par;
static constexpr auto EXEC_POLICY = dpl::execution::par_unseq;
    #define USE_STD_PTR_ALLOC_DEALLOC

  #endif

#else

// Normal C++17 PSTL

  #include <algorithm>
  #include <execution>
  #include <numeric>

// auto exe_policy = std::execution::seq;
// auto exe_policy = std::execution::par;
static constexpr auto EXEC_POLICY = std::execution::par_unseq;
  #define USE_STD_PTR_ALLOC_DEALLOC

#endif

#ifdef USE_STD_PTR_ALLOC_DEALLOC

template <typename T> T *alloc_raw(size_t size) { return static_cast<T *>(std::malloc(size * sizeof(T))); }
template <typename T> void dealloc_raw(T *ptr) { std::free(ptr); }

#endif

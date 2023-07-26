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

#include <Kokkos_Core.hpp>
#include <iostream>
#include <utility>

#include "shared.h"

namespace clover {

struct context {};

template <typename T> struct Buffer1D {
  Kokkos::View<T *> view;

  explicit Buffer1D(clover::context, size_t x) : view(Kokkos::ViewAllocateWithoutInitializing(""), x) {}

  template <size_t D> [[nodiscard]] size_t extent() const {
    static_assert(D < 1);
    return view.extent(D);
  }

  std::vector<T> mirrored() {
    std::vector<T> out(view.size());
    auto data = view.data();
    std::copy(data, data + out.size(), out.begin());
    return out;
  }
};

template <typename T> struct Buffer2D {
  Kokkos::View<T **> view;

  explicit Buffer2D(clover::context, size_t x, size_t y) : view(Kokkos::ViewAllocateWithoutInitializing(""), x, y) {}

  template <size_t D> [[nodiscard]] size_t extent() const {
    static_assert(D < 2);
    return view.extent(D);
  }

  std::vector<T> mirrored() {
    std::vector<T> out(view.size());
    auto data = view.data();
    std::copy(data, data + out.size(), out.begin());
    return out;
  }
  clover::BufferMirror2D<T> mirrored2() { return {mirrored(), extent<0>(), extent<1>()}; }
};

template <typename T> using StagingBuffer1D = T *;

struct chunk_context {
  //  Kokkos::View<double*>::HostMirror hm_left_rcv_buffer, hm_right_rcv_buffer, hm_bottom_rcv_buffer, hm_top_rcv_buffer;
  //  Kokkos::View<double*>::HostMirror hm_left_snd_buffer, hm_right_snd_buffer, hm_bottom_snd_buffer, hm_top_snd_buffer;
};

// template <typename T> using Buffer1D = Kokkos::View<T *>;
// template <typename T> using Buffer2D = Kokkos::View<T **>;
// #define BUFFER_NO_FORWARD_DECL
} // namespace clover

using clover::Range1d;
using clover::Range2d;

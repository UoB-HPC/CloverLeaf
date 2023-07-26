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
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "dpl_shim.h"
#include "shared.h"

namespace clover {

struct chunk_context {};
struct context {};

template <typename T> struct Buffer1D {
  size_t size;
  T *data;
  Buffer1D(context &, size_t size) : size(size), data(alloc_raw<T>(size)) {}
  // XXX the following ctors break oneDPL, making this class not device_copyable
  //  Buffer1D(Buffer1D &&other) noexcept : size(other.size), data(std::exchange(other.data, nullptr)) {}
  //  Buffer1D(const Buffer1D<T> &that) : size(that.size), data(that.data) {}
  // XXX this model captures this class by value, the dtor is called for each lambda scope which is wrong!
  //  ~Buffer1D() { std::free(data); }

  void release() { dealloc_raw(data); }

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
  Buffer2D(context &, size_t sizeX, size_t sizeY) : sizeX(sizeX), sizeY(sizeY), data(alloc_raw<T>(sizeX * sizeY)) {}
  // XXX the following ctors break oneDPL, making this class not device_copyable
  //  Buffer2D(Buffer2D &&other) noexcept : sizeX(other.sizeX), sizeY(other.sizeY), data(std::exchange(other.data, nullptr)) {}
  //  Buffer2D(const Buffer2D<T> &that) : sizeX(that.sizeX), sizeY(that.sizeY), data(that.data) {}
  // XXX this model captures this class by value, the dtor is called for each lambda scope which is wrong!
  // ~Buffer2D() { std::free(data); }

  void release() { dealloc_raw(data); }

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

template <typename N> class range {
public:
  class iterator {
    friend class range;

  public:
    using difference_type = typename std::make_signed_t<N>;
    using value_type = N;
    using pointer = const N *;
    using reference = N;
    using iterator_category = std::random_access_iterator_tag;

    // XXX This is not part of the iterator spec, it gets picked up by oneDPL if enabled.
    // Without this, the DPL SYCL backend collects the iterator data on the host and copies to the device.
    // This type is unused for any other STL impl.
    using is_passed_directly = std::true_type;

    reference operator*() const { return i_; }
    iterator &operator++() {
      ++i_;
      return *this;
    }
    iterator operator++(int) {
      iterator copy(*this);
      ++i_;
      return copy;
    }

    iterator &operator--() {
      --i_;
      return *this;
    }
    iterator operator--(int) {
      iterator copy(*this);
      --i_;
      return copy;
    }

    iterator &operator+=(N by) {
      i_ += by;
      return *this;
    }

    value_type operator[](const difference_type &i) const { return i_ + i; }

    difference_type operator-(const iterator &it) const { return i_ - it.i_; }
    iterator operator+(const value_type v) const { return iterator(i_ + v); }

    bool operator==(const iterator &other) const { return i_ == other.i_; }
    bool operator!=(const iterator &other) const { return i_ != other.i_; }
    bool operator<(const iterator &other) const { return i_ < other.i_; }

  protected:
    explicit iterator(N start) : i_(start) {}

  private:
    N i_;
  };

  [[nodiscard]] iterator begin() const { return begin_; }
  [[nodiscard]] iterator end() const { return end_; }
  range(N begin, N end) : begin_(begin), end_(end) {}

private:
  iterator begin_;
  iterator end_;
};

template <typename F> static void par_ranged1(const Range1d &r, const F &functor) {
  auto groups = range<int>(r.from, r.to);
  std::for_each(EXEC_POLICY, groups.begin(), groups.end(), [functor](int i) { functor(i); });
  // for (size_t i = r.from; i < r.to; i++) {
  //	functor(i);
  // }
}

template <typename F> static void par_ranged2(const Range2d &r, const F &functor) {
  auto xy = range<int>(0, r.sizeX * r.sizeY);
  std::for_each(EXEC_POLICY, xy.begin(), xy.end(), [=](int v) {
    const auto x = r.fromX + (v % r.sizeX);
    const auto y = r.fromY + (v / r.sizeX);
    functor(x, y);
  });
  // for (size_t j = r.fromY; j < r.toY; j++) {
  //     for (size_t i = r.fromX; i < r.toX; i++) {
  //         functor(i, j);
  //     }
  // }
}

} // namespace clover

using clover::Range1d;
using clover::Range2d;

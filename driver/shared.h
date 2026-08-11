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

#include <cassert>
#include <cstddef>
#include <ostream>

struct global_variables;

namespace clover {

template <typename T> struct BufferMirror2D {
  size_t sizeX, sizeY;
  std::vector<T> actual;
  BufferMirror2D(std::vector<T> actual, size_t sizeX, size_t sizeY) : sizeX(sizeX), sizeY(sizeY), actual(actual) {
    if (sizeX * sizeY != actual.size()) throw std::logic_error("Bad mirror size");
  }

  BufferMirror2D(T *actual_data, size_t sizeX, size_t sizeY) : sizeX(sizeX), sizeY(sizeY) {
    actual.assign(actual_data, actual_data + sizeX*sizeY);
  }

  T &operator()(size_t i, size_t j)   { return actual[j + i * sizeY]; }
};


struct Range1d {
  const size_t from, to;
  const size_t size;
  template <typename A, typename B> Range1d(A from, B to) : from(from), to(to), size(to - from) {
    assert(from < to);
    assert(size != 0);
  }
  friend std::ostream &operator<<(std::ostream &os, const Range1d &d);
};

struct Range2d {
  const size_t fromX, toX;
  const size_t fromY, toY;
  const size_t sizeX, sizeY;
  template <typename A, typename B, typename C, typename D>
  Range2d(A fromX, B fromY, C toX, D toY) : fromX(fromX), toX(toX), fromY(fromY), toY(toY), sizeX(toX - fromX), sizeY(toY - fromY) {
    assert(fromX < toX);
    assert(fromY < toY);
    assert(sizeX != 0);
    assert(sizeY != 0);
  }
  friend std::ostream &operator<<(std::ostream &os, const Range2d &d);
};

void dump(global_variables &g, const std::string &filename);

} // namespace clover

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

#include "reset_field.h"
#include "context.h"
#include "timer.h"

//  @brief Fortran reset field kernel.
//  @author Wayne Gaudin
//  @details Copies all of the final end of step filed data to the begining of
//  step data, ready for the next timestep.
void reset_field_kernel(int x_min, int x_max, int y_min, int y_max, clover::Buffer2D<double> &density0, clover::Buffer2D<double> &density1,
                        clover::Buffer2D<double> &energy0, clover::Buffer2D<double> &energy1, clover::Buffer2D<double> &xvel0,
                        clover::Buffer2D<double> &xvel1, clover::Buffer2D<double> &yvel0, clover::Buffer2D<double> &yvel1) {

  // DO k=y_min,y_max
  //   DO j=x_min,x_max
  //  clover::par_ranged2(Range2d{x_min + 1, y_min + 1, x_max + 2, y_max + 2}, [=] DEVICE_KERNEL(const int i, const int j) {
  const RAJA::TypedRangeSegment<int> row_Range(y_min + 1,  y_max + 2);
  const RAJA::TypedRangeSegment<int> col_Range(x_min + 1,  x_max + 2);
  RAJA::kernel<KERNEL_EXEC_POL>(RAJA::make_tuple(col_Range, row_Range),
      [=] RAJA_HOST_DEVICE (const int i, const int j) {
    density0(i, j) = density1(i, j);
    energy0(i, j) = energy1(i, j);
  });

  // DO k=y_min,y_max+1
  //   DO j=x_min,x_max+1
  // clover::par_ranged2(Range2d{x_min + 1, y_min + 1, x_max + 1 + 2, y_max + 1 + 2}, [=] DEVICE_KERNEL(const int i, const int j) {
  const RAJA::TypedRangeSegment<int> row_Range1(y_min + 1,  y_max + 1 + 2);
  const RAJA::TypedRangeSegment<int> col_Range1(x_min + 1,  x_max + 1 + 2);
  RAJA::kernel<KERNEL_EXEC_POL>(RAJA::make_tuple(col_Range1, row_Range1),
      [=] RAJA_HOST_DEVICE (const int i, const int j) {
    xvel0(i, j) = xvel1(i, j);
    yvel0(i, j) = yvel1(i, j);
  });
}

//  @brief Reset field driver
//  @author Wayne Gaudin
//  @details Invokes the user specified field reset kernel.
void reset_field(global_variables &globals) {

  double kernel_time = 0;
  if (globals.profiler_on) kernel_time = timer();

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {

    tile_type &t = globals.chunk.tiles[tile];
    reset_field_kernel(t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax,

                       t.field.density0, t.field.density1, t.field.energy0, t.field.energy1, t.field.xvel0, t.field.xvel1, t.field.yvel0,
                       t.field.yvel1);
  }

  if (globals.profiler_on) {
    clover::checkError(rajaDeviceSynchronize());
    globals.profiler.reset += timer() - kernel_time;
  }
}

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

#include "accelerate.h"
#include "context.h"
#include "timer.h"

//#define par_ranged2m(rg, f) \
//{"KERNEL2D_START";                        \
//                           \
//    Range2d range = rg;                    \
//    for (size_t j = range.fromY; j < range.toY; j++) {        \
//        for (size_t i = range.fromX; i < range.toX; i++)  {"KERNEL2D_A";f"KERNEL2D_B";}                           \
//    }                                                      \
//"KERNEL2D_END";}

// @brief Fortran acceleration kernel
// @author Wayne Gaudin
// @details The pressure and viscosity gradients are used to update the
// velocity field.
void accelerate_kernel(int x_min, int x_max, int y_min, int y_max, double dt, clover::Buffer2D<double> &xarea,
                       clover::Buffer2D<double> &yarea, clover::Buffer2D<double> &volume, clover::Buffer2D<double> &density0,
                       clover::Buffer2D<double> &pressure, clover::Buffer2D<double> &viscosity, clover::Buffer2D<double> &xvel0,
                       clover::Buffer2D<double> &yvel0, clover::Buffer2D<double> &xvel1, clover::Buffer2D<double> &yvel1) {

  double halfdt = 0.5 * dt;

  // DO k=y_min,y_max+1
  //   DO j=x_min,x_max+1
  //	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_min + 1, y_min + 1},
  //	                                               {x_max + 1 + 2, y_max + 1 + 2});

  clover::par_ranged2(Range2d{x_min + 1, y_min + 1, x_max + 1 + 2, y_max + 1 + 2}, [=] DEVICE_KERNEL(const int i, const int j) {
    double stepbymass_s = halfdt / ((density0(i - 1, j - 1) * volume(i - 1, j - 1) + density0(i - 1, j + 0) * volume(i - 1, j + 0) +
                                     density0(i, j) * volume(i, j) + density0(i + 0, j - 1) * volume(i + 0, j - 1)) *
                                    0.25);

    xvel1(i, j) = xvel0(i, j) - stepbymass_s * (xarea(i, j) * (pressure(i, j) - pressure(i - 1, j + 0)) +
                                                xarea(i + 0, j - 1) * (pressure(i + 0, j - 1) - pressure(i - 1, j - 1)));
    yvel1(i, j) = yvel0(i, j) - stepbymass_s * (yarea(i, j) * (pressure(i, j) - pressure(i + 0, j - 1)) +
                                                yarea(i - 1, j + 0) * (pressure(i - 1, j + 0) - pressure(i - 1, j - 1)));
    xvel1(i, j) = xvel1(i, j) - stepbymass_s * (xarea(i, j) * (viscosity(i, j) - viscosity(i - 1, j + 0)) +
                                                xarea(i + 0, j - 1) * (viscosity(i + 0, j - 1) - viscosity(i - 1, j - 1)));
    yvel1(i, j) = yvel1(i, j) - stepbymass_s * (yarea(i, j) * (viscosity(i, j) - viscosity(i + 0, j - 1)) +
                                                yarea(i - 1, j + 0) * (viscosity(i - 1, j + 0) - viscosity(i - 1, j - 1)));
  });
}

//  @brief Driver for the acceleration kernels
//  @author Wayne Gaudin
//  @details Calls user requested kernel
void accelerate(global_variables &globals) {

  double kernel_time = 0;
  if (globals.profiler_on) kernel_time = timer();

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    tile_type &t = globals.chunk.tiles[tile];

    accelerate_kernel(t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, globals.dt, t.field.xarea, t.field.yarea, t.field.volume,
                      t.field.density0, t.field.pressure, t.field.viscosity, t.field.xvel0, t.field.yvel0, t.field.xvel1, t.field.yvel1);
  }

  if (globals.profiler_on) {
    clover::checkError(cudaDeviceSynchronize());
    globals.profiler.acceleration += timer() - kernel_time;
  }
}

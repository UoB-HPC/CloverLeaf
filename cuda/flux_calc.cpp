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

#include "flux_calc.h"
#include "context.h"
#include "timer.h"

//  @brief Fortran flux kernel.
//  @author Wayne Gaudin
//  @details The edge volume fluxes are calculated based on the velocity fields.
void flux_calc_kernel(int x_min, int x_max, int y_min, int y_max, double dt, clover::Buffer2D<double> &xarea,
                      clover::Buffer2D<double> &yarea, clover::Buffer2D<double> &xvel0, clover::Buffer2D<double> &yvel0,
                      clover::Buffer2D<double> &xvel1, clover::Buffer2D<double> &yvel1, clover::Buffer2D<double> &vol_flux_x,
                      clover::Buffer2D<double> &vol_flux_y) {

  // DO k=y_min,y_max+1
  //   DO j=x_min,x_max+1
  // Note that the loops calculate one extra flux than required, but this
  // allows loop fusion that improves performance
  clover::par_ranged2(Range2d{x_min + 1, y_min + 1, x_max + 1 + 2, y_max + 1 + 2}, [=] DEVICE_KERNEL(const int i, const int j) {
    vol_flux_x(i, j) = 0.25 * dt * xarea(i, j) * (xvel0(i, j) + xvel0(i + 0, j + 1) + xvel1(i, j) + xvel1(i + 0, j + 1));
    vol_flux_y(i, j) = 0.25 * dt * yarea(i, j) * (yvel0(i, j) + yvel0(i + 1, j + 0) + yvel1(i, j) + yvel1(i + 1, j + 0));
  });
}

// @brief Driver for the flux kernels
// @author Wayne Gaudin
// @details Invokes the used specified flux kernel
void flux_calc(global_variables &globals) {

  double kernel_time = 0;
  if (globals.profiler_on) kernel_time = timer();

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {

    tile_type &t = globals.chunk.tiles[tile];
    flux_calc_kernel(t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, globals.dt, t.field.xarea, t.field.yarea, t.field.xvel0,
                     t.field.yvel0, t.field.xvel1, t.field.yvel1, t.field.vol_flux_x, t.field.vol_flux_y);
  }

  if (globals.profiler_on) {
    clover::checkError(cudaDeviceSynchronize());
    globals.profiler.flux += timer() - kernel_time;
  }
}

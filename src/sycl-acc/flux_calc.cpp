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
void flux_calc_kernel(handler &h, int x_min, int x_max, int y_min, int y_max, double dt, clover::Accessor<double, 2, R>::Type xarea,
                      clover::Accessor<double, 2, R>::Type yarea, clover::Accessor<double, 2, R>::Type xvel0,
                      clover::Accessor<double, 2, R>::Type yvel0, clover::Accessor<double, 2, R>::Type xvel1,
                      clover::Accessor<double, 2, R>::Type yvel1, clover::Accessor<double, 2, W>::Type vol_flux_x,
                      clover::Accessor<double, 2, W>::Type vol_flux_y) {

  // DO k=y_min,y_max+1
  //   DO j=x_min,x_max+1
  // Note that the loops calculate one extra flux than required, but this
  // allows loop fusion that improves performance
  clover::par_ranged<class flux_calc>(h, {x_min + 1, y_min + 1, x_max + 1 + 2, y_max + 1 + 2}, [=](id<2> idx) {
    vol_flux_x[idx] =
        0.25 * dt * xarea[idx] * (xvel0[idx] + xvel0[clover::offset(idx, 0, 1)] + xvel1[idx] + xvel1[clover::offset(idx, 0, 1)]);
    vol_flux_y[idx] =
        0.25 * dt * yarea[idx] * (yvel0[idx] + yvel0[clover::offset(idx, 1, 0)] + yvel1[idx] + yvel1[clover::offset(idx, 1, 0)]);
  });
}

// @brief Driver for the flux kernels
// @author Wayne Gaudin
// @details Invokes the used specified flux kernel
void flux_calc(global_variables &globals) {

  double kernel_time = 0;
  if (globals.profiler_on) kernel_time = timer();

  clover::execute(globals.context.queue, [&](handler &h) {
    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {

      tile_type &t = globals.chunk.tiles[tile];
      flux_calc_kernel(h, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, globals.dt,
                       t.field.xarea.access<R>(h),      //
                       t.field.yarea.access<R>(h),      //
                       t.field.xvel0.access<R>(h),      //
                       t.field.yvel0.access<R>(h),      //
                       t.field.xvel1.access<R>(h),      //
                       t.field.yvel1.access<R>(h),      //
                       t.field.vol_flux_x.access<W>(h), //
                       t.field.vol_flux_y.access<W>(h)  //
      );
    }
  });

  if (globals.profiler_on) globals.profiler.flux += timer() - kernel_time;
}

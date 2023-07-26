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
#include "timer.h"

//  @brief Fortran flux kernel.
//  @author Wayne Gaudin
//  @details The edge volume fluxes are calculated based on the velocity fields.
void flux_calc_kernel(bool use_target, int x_min, int x_max, int y_min, int y_max, double dt, field_type &field) {

  // DO k=y_min,y_max+1
  //   DO j=x_min,x_max+1
  // Note that the loops calculate one extra flux than required, but this
  // allows loop fusion that improves performance

  const int flux_x_stride = field.flux_x_stride;
  const int flux_y_stride = field.flux_y_stride;
  const int vels_wk_stride = field.vels_wk_stride;

  double *xarea = field.xarea.data;
  double *yarea = field.yarea.data;
  double *xvel0 = field.xvel0.data;
  double *yvel0 = field.yvel0.data;
  double *xvel1 = field.xvel1.data;
  double *yvel1 = field.yvel1.data;
  double *vol_flux_x = field.vol_flux_x.data;
  double *vol_flux_y = field.vol_flux_y.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
  for (int j = (y_min + 1); j < (y_max + 1 + 2); j++) {
    for (int i = (x_min + 1); i < (x_max + 1 + 2); i++) {
      vol_flux_x[i + j * flux_x_stride] = 0.25 * dt * xarea[i + j * flux_x_stride] *
                                          (xvel0[i + j * vels_wk_stride] + xvel0[(i + 0) + (j + 1) * vels_wk_stride] +
                                           xvel1[i + j * vels_wk_stride] + xvel1[(i + 0) + (j + 1) * vels_wk_stride]);
      vol_flux_y[i + j * flux_y_stride] = 0.25 * dt * yarea[i + j * flux_y_stride] *
                                          (yvel0[i + j * vels_wk_stride] + yvel0[(i + 1) + (j + 0) * vels_wk_stride] +
                                           yvel1[i + j * vels_wk_stride] + yvel1[(i + 1) + (j + 0) * vels_wk_stride]);
    }
  }
}

// @brief Driver for the flux kernels
// @author Wayne Gaudin
// @details Invokes the used specified flux kernel
void flux_calc(global_variables &globals) {

  double kernel_time = 0;
  if (globals.profiler_on) kernel_time = timer();

#if SYNC_BUFFERS
  globals.hostToDevice();
#endif

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {

    tile_type &t = globals.chunk.tiles[tile];
    flux_calc_kernel(globals.context.use_target, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, globals.dt, t.field);
  }

#if SYNC_BUFFERS
  globals.deviceToHost();
#endif

  if (globals.profiler_on) globals.profiler.flux += timer() - kernel_time;
}

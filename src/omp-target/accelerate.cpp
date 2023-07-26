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
#include "timer.h"

// @brief Fortran acceleration kernel
// @author Wayne Gaudin
// @details The pressure and viscosity gradients are used to update the
// velocity field.
void accelerate_kernel(bool use_target, int x_min, int x_max, int y_min, int y_max, double dt, field_type &field) {

  double halfdt = 0.5 * dt;

  // DO k=y_min,y_max+1
  //   DO j=x_min,x_max+1
  //	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_min + 1, y_min + 1},
  //	                                               {x_max + 1 + 2, y_max + 1 + 2});

  // for(int j = )

  const int xarea_sizex = field.flux_x_stride;
  const int yarea_sizex = field.flux_y_stride;
  const int base_stride = field.base_stride;
  const int vels_wk_stride = field.vels_wk_stride;

  double *xarea = field.xarea.data;
  double *yarea = field.yarea.data;
  double *volume = field.volume.data;
  double *density0 = field.density0.data;
  double *pressure = field.pressure.data;
  double *viscosity = field.viscosity.data;
  double *xvel0 = field.xvel0.data;
  double *yvel0 = field.yvel0.data;
  double *xvel1 = field.xvel1.data;
  double *yvel1 = field.yvel1.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
  for (int j = (y_min + 1); j < (y_max + 1 + 2); j++) {
    for (int i = (x_min + 1); i < (x_max + 1 + 2); i++) {
      double stepbymass_s = halfdt / ((density0[(i - 1) + (j - 1) * base_stride] * volume[(i - 1) + (j - 1) * base_stride] +
                                       density0[(i - 1) + (j + 0) * base_stride] * volume[(i - 1) + (j + 0) * base_stride] +
                                       density0[i + j * base_stride] * volume[i + j * base_stride] +
                                       density0[(i + 0) + (j - 1) * base_stride] * volume[(i + 0) + (j - 1) * base_stride]) *
                                      0.25);
      xvel1[i + j * vels_wk_stride] =
          xvel0[i + j * vels_wk_stride] -
          stepbymass_s * (xarea[i + j * xarea_sizex] * (pressure[i + j * base_stride] - pressure[(i - 1) + (j + 0) * base_stride]) +
                          xarea[(i + 0) + (j - 1) * xarea_sizex] *
                              (pressure[(i + 0) + (j - 1) * base_stride] - pressure[(i - 1) + (j - 1) * base_stride]));
      yvel1[i + j * vels_wk_stride] =
          yvel0[i + j * vels_wk_stride] -
          stepbymass_s * (yarea[i + j * yarea_sizex] * (pressure[i + j * base_stride] - pressure[(i + 0) + (j - 1) * base_stride]) +
                          yarea[(i - 1) + (j + 0) * yarea_sizex] *
                              (pressure[(i - 1) + (j + 0) * base_stride] - pressure[(i - 1) + (j - 1) * base_stride]));
      xvel1[i + j * vels_wk_stride] =
          xvel1[i + j * vels_wk_stride] -
          stepbymass_s * (xarea[i + j * xarea_sizex] * (viscosity[i + j * base_stride] - viscosity[(i - 1) + (j + 0) * base_stride]) +
                          xarea[(i + 0) + (j - 1) * xarea_sizex] *
                              (viscosity[(i + 0) + (j - 1) * base_stride] - viscosity[(i - 1) + (j - 1) * base_stride]));
      yvel1[i + j * vels_wk_stride] =
          yvel1[i + j * vels_wk_stride] -
          stepbymass_s * (yarea[i + j * yarea_sizex] * (viscosity[i + j * base_stride] - viscosity[(i + 0) + (j - 1) * base_stride]) +
                          yarea[(i - 1) + (j + 0) * yarea_sizex] *
                              (viscosity[(i - 1) + (j + 0) * base_stride] - viscosity[(i - 1) + (j - 1) * base_stride]));
    }
  }
}

//  @brief Driver for the acceleration kernels
//  @author Wayne Gaudin
//  @details Calls user requested kernel
void accelerate(global_variables &globals) {

  double kernel_time = 0;
  if (globals.profiler_on) kernel_time = timer();

#if SYNC_BUFFERS
  globals.hostToDevice();
#endif

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    tile_type &t = globals.chunk.tiles[tile];

    accelerate_kernel(globals.context.use_target, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, globals.dt, t.field);
  }

#if SYNC_BUFFERS
  globals.deviceToHost();
#endif

  if (globals.profiler_on) globals.profiler.acceleration += timer() - kernel_time;
}

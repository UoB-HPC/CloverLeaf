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
void accelerate_kernel(int x_min, int x_max, int y_min, int y_max, double dt, Kokkos::View<double **> &xarea,
                       Kokkos::View<double **> &yarea, Kokkos::View<double **> &volume, Kokkos::View<double **> &density0,
                       Kokkos::View<double **> &pressure, Kokkos::View<double **> &viscosity, Kokkos::View<double **> &xvel0,
                       Kokkos::View<double **> &yvel0, Kokkos::View<double **> &xvel1, Kokkos::View<double **> &yvel1) {

  double halfdt = 0.5 * dt;

  // DO k=y_min,y_max+1
  //   DO j=x_min,x_max+1
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({x_min + 1, y_min + 1}, {x_max + 1 + 2, y_max + 1 + 2});
  Kokkos::parallel_for(
      "accelerate", policy, KOKKOS_LAMBDA(const int j, const int k) {
        double stepbymass_s = halfdt / ((density0(j - 1, k - 1) * volume(j - 1, k - 1) + density0(j, k - 1) * volume(j, k - 1) +
                                         density0(j, k) * volume(j, k) + density0(j - 1, k) * volume(j - 1, k)) *
                                        0.25);

        xvel1(j, k) = xvel0(j, k) - stepbymass_s * (xarea(j, k) * (pressure(j, k) - pressure(j - 1, k)) +
                                                    xarea(j, k - 1) * (pressure(j, k - 1) - pressure(j - 1, k - 1)));
        yvel1(j, k) = yvel0(j, k) - stepbymass_s * (yarea(j, k) * (pressure(j, k) - pressure(j, k - 1)) +
                                                    yarea(j - 1, k) * (pressure(j - 1, k) - pressure(j - 1, k - 1)));
        xvel1(j, k) = xvel1(j, k) - stepbymass_s * (xarea(j, k) * (viscosity(j, k) - viscosity(j - 1, k)) +
                                                    xarea(j, k - 1) * (viscosity(j, k - 1) - viscosity(j - 1, k - 1)));
        yvel1(j, k) = yvel1(j, k) - stepbymass_s * (yarea(j, k) * (viscosity(j, k) - viscosity(j, k - 1)) +
                                                    yarea(j - 1, k) * (viscosity(j - 1, k) - viscosity(j - 1, k - 1)));
      });
}

//  @brief Driver for the acceleration kernels
//  @author Wayne Gaudin
//  @details Calls user requested kernel
void accelerate(global_variables &globals) {

  double kernel_time;
  if (globals.profiler_on) kernel_time = timer();

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {

    accelerate_kernel(globals.chunk.tiles[tile].info.t_xmin, globals.chunk.tiles[tile].info.t_xmax, globals.chunk.tiles[tile].info.t_ymin,
                      globals.chunk.tiles[tile].info.t_ymax, globals.dt, globals.chunk.tiles[tile].field.xarea.view,
                      globals.chunk.tiles[tile].field.yarea.view, globals.chunk.tiles[tile].field.volume.view,
                      globals.chunk.tiles[tile].field.density0.view, globals.chunk.tiles[tile].field.pressure.view,
                      globals.chunk.tiles[tile].field.viscosity.view, globals.chunk.tiles[tile].field.xvel0.view,
                      globals.chunk.tiles[tile].field.yvel0.view, globals.chunk.tiles[tile].field.xvel1.view,
                      globals.chunk.tiles[tile].field.yvel1.view);
  }

  if (globals.profiler_on) globals.profiler.acceleration += timer() - kernel_time;
}

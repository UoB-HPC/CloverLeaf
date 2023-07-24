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
void flux_calc_kernel(int x_min, int x_max, int y_min, int y_max, double dt, Kokkos::View<double **> &xarea, Kokkos::View<double **> &yarea,
                      Kokkos::View<double **> &xvel0, Kokkos::View<double **> &yvel0, Kokkos::View<double **> &xvel1,
                      Kokkos::View<double **> &yvel1, Kokkos::View<double **> &vol_flux_x, Kokkos::View<double **> &vol_flux_y) {

  // DO k=y_min,y_max+1
  //   DO j=x_min,x_max+1
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({x_min + 1, y_min + 1}, {x_max + 1 + 2, y_max + 1 + 2});

  // Note that the loops calculate one extra flux than required, but this
  // allows loop fusion that improves performance
  Kokkos::parallel_for(
      "flux_calc", policy, KOKKOS_LAMBDA(const int j, const int k) {
        vol_flux_x(j, k) = 0.25 * dt * xarea(j, k) * (xvel0(j, k) + xvel0(j, k + 1) + xvel1(j, k) + xvel1(j, k + 1));
        vol_flux_y(j, k) = 0.25 * dt * yarea(j, k) * (yvel0(j, k) + yvel0(j + 1, k) + yvel1(j, k) + yvel1(j + 1, k));
      });
}

// @brief Driver for the flux kernels
// @author Wayne Gaudin
// @details Invokes the used specified flux kernel
void flux_calc(global_variables &globals) {

  double kernel_time;
  if (globals.profiler_on) kernel_time = timer();

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {

    flux_calc_kernel(globals.chunk.tiles[tile].info.t_xmin, globals.chunk.tiles[tile].info.t_xmax, globals.chunk.tiles[tile].info.t_ymin,
                     globals.chunk.tiles[tile].info.t_ymax, globals.dt, globals.chunk.tiles[tile].field.xarea.view,
                     globals.chunk.tiles[tile].field.yarea.view, globals.chunk.tiles[tile].field.xvel0.view,
                     globals.chunk.tiles[tile].field.yvel0.view, globals.chunk.tiles[tile].field.xvel1.view,
                     globals.chunk.tiles[tile].field.yvel1.view, globals.chunk.tiles[tile].field.vol_flux_x.view,
                     globals.chunk.tiles[tile].field.vol_flux_y.view);
  }

  if (globals.profiler_on) globals.profiler.flux += timer() - kernel_time;
}

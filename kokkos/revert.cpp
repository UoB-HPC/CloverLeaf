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

#include "revert.h"

//  @brief Fortran revert kernel.
//  @author Wayne Gaudin
//  @details Takes the half step field data used in the predictor and reverts
//  it to the start of step data, ready for the corrector.
//  Note that this does not seem necessary in this proxy-app but should be
//  left in to remain relevant to the full method.
void revert_kernel(int x_min, int x_max, int y_min, int y_max, Kokkos::View<double **> &density0, Kokkos::View<double **> &density1,
                   Kokkos::View<double **> &energy0, Kokkos::View<double **> &energy1) {

  // DO k=y_min,y_max
  //   DO j=x_min,x_max
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({x_min + 1, y_min + 1}, {x_max + 2, y_max + 2});

  Kokkos::parallel_for(
      "revert", policy, KOKKOS_LAMBDA(const int j, const int k) {
        density1(j, k) = density0(j, k);
        energy1(j, k) = energy0(j, k);
      });
}

//  @brief Driver routine for the revert kernels.
//  @author Wayne Gaudin
//  @details Invokes the user specified revert kernel.
void revert(global_variables &globals) {
  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    tile_type &type = globals.chunk.tiles[tile];
    revert_kernel(type.info.t_xmin, type.info.t_xmax, type.info.t_ymin, type.info.t_ymax, type.field.density0.view,
                  type.field.density1.view, type.field.energy0.view, type.field.energy1.view);
  }
}

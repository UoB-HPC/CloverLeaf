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

#include "viscosity.h"

//  @brief Fortran viscosity kernel.
//  @author Wayne Gaudin
//  @details Calculates an artificial viscosity using the Wilkin's method to
//  smooth out shock front and prevent oscillations around discontinuities.
//  Only cells in compression will have a non-zero value.

void viscosity_kernel(int x_min, int x_max, int y_min, int y_max, Kokkos::View<double *> &celldx, Kokkos::View<double *> &celldy,
                      Kokkos::View<double **> &density0, Kokkos::View<double **> &pressure, Kokkos::View<double **> &viscosity,
                      Kokkos::View<double **> &xvel0, Kokkos::View<double **> &yvel0) {

  // DO k=y_min,y_max
  //   DO j=x_min,x_max
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({x_min + 1, y_min + 1}, {x_max + 2, y_max + 2});
  Kokkos::parallel_for(
      "viscosity", policy, KOKKOS_LAMBDA(const int j, const int k) {
        double ugrad = (xvel0(j + 1, k) + xvel0(j + 1, k + 1)) - (xvel0(j, k) + xvel0(j, k + 1));

        double vgrad = (yvel0(j, k + 1) + yvel0(j + 1, k + 1)) - (yvel0(j, k) + yvel0(j + 1, k));

        double div = (celldx(j) * (ugrad) + celldy(k) * (vgrad));

        double strain2 = 0.5 * (xvel0(j, k + 1) + xvel0(j + 1, k + 1) - xvel0(j, k) - xvel0(j + 1, k)) / celldy(k) +
                         0.5 * (yvel0(j + 1, k) + yvel0(j + 1, k + 1) - yvel0(j, k) - yvel0(j, k + 1)) / celldx(j);

        double pgradx = (pressure(j + 1, k) - pressure(j - 1, k)) / (celldx(j) + celldx(j + 1));
        double pgrady = (pressure(j, k + 1) - pressure(j, k - 1)) / (celldy(k) + celldy(k + 1));

        double pgradx2 = pgradx * pgradx;
        double pgrady2 = pgrady * pgrady;

        double limiter = ((0.5 * (ugrad) / celldx(j)) * pgradx2 + (0.5 * (vgrad) / celldy(k)) * pgrady2 + strain2 * pgradx * pgrady) /
                         Kokkos::max(pgradx2 + pgrady2, 1.0e-16);

        if ((limiter > 0.0) || (div >= 0.0)) {
          viscosity(j, k) = 0.0;
        } else {
          double dirx = 1.0;
          if (pgradx < 0.0) dirx = -1.0;
          pgradx = dirx * Kokkos::max(1.0e-16, fabs(pgradx));
          double diry = 1.0;
          if (pgradx < 0.0) diry = -1.0;
          pgrady = diry * Kokkos::max(1.0e-16, fabs(pgrady));
          double pgrad = Kokkos::sqrt(pgradx * pgradx + pgrady * pgrady);
          double xgrad = Kokkos::fabs(celldx(j) * pgrad / pgradx);
          double ygrad = Kokkos::fabs(celldy(k) * pgrad / pgrady);
          double grad = Kokkos::min(xgrad, ygrad);
          double grad2 = grad * grad;

          viscosity(j, k) = 2.0 * density0(j, k) * grad2 * limiter * limiter;
        }
      });
}

//  @brief Driver for the viscosity kernels
//  @author Wayne Gaudin
//  @details Selects the user specified kernel to caluclate the artificial
//  viscosity.
void viscosity(global_variables &globals) {
  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    tile_type &type = globals.chunk.tiles[tile];
    viscosity_kernel(type.info.t_xmin, type.info.t_xmax, //
                     type.info.t_ymin, type.info.t_ymax, //
                     type.field.celldx.view,             //
                     type.field.celldy.view,             //
                     type.field.density0.view,           //
                     type.field.pressure.view,           //
                     type.field.viscosity.view,          //
                     type.field.xvel0.view,              //
                     type.field.yvel0.view);
  }
}

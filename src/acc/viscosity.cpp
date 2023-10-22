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
#include "context.h"
#include <cmath>

//  @brief Fortran viscosity kernel.
//  @author Wayne Gaudin
//  @details Calculates an artificial viscosity using the Wilkin's method to
//  smooth out shock front and prevent oscillations around discontinuities.
//  Only cells in compression will have a non-zero value.

void viscosity_kernel(bool use_target, int x_min, int x_max, int y_min, int y_max, field_type &field) {

  // DO k=y_min,y_max
  //   DO j=x_min,x_max

  const int base_stride = field.base_stride;
  const int vels_wk_stride = field.vels_wk_stride;

  double *celldx = field.celldx.data;
  double *celldy = field.celldy.data;
  double *density0 = field.density0.data;
  double *pressure = field.pressure.data;
  double *viscosity = field.viscosity.data;
  double *xvel0 = field.xvel0.data;
  double *yvel0 = field.yvel0.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
  for (int j = (y_min + 1); j < (y_max + 2); j++) {
    for (int i = (x_min + 1); i < (x_max + 2); i++) {
      double ugrad = (xvel0[(i + 1) + (j + 0) * vels_wk_stride] + xvel0[(i + 1) + (j + 1) * vels_wk_stride]) -
                     (xvel0[i + j * vels_wk_stride] + xvel0[(i + 0) + (j + 1) * vels_wk_stride]);
      double vgrad = (yvel0[(i + 0) + (j + 1) * vels_wk_stride] + yvel0[(i + 1) + (j + 1) * vels_wk_stride]) -
                     (yvel0[i + j * vels_wk_stride] + yvel0[(i + 1) + (j + 0) * vels_wk_stride]);
      double div = (celldx[i] * (ugrad) + celldy[j] * (vgrad));
      double strain2 = 0.5 *
                           (xvel0[(i + 0) + (j + 1) * vels_wk_stride] + xvel0[(i + 1) + (j + 1) * vels_wk_stride] -
                            xvel0[i + j * vels_wk_stride] - xvel0[(i + 1) + (j + 0) * vels_wk_stride]) /
                           celldy[j] +
                       0.5 *
                           (yvel0[(i + 1) + (j + 0) * vels_wk_stride] + yvel0[(i + 1) + (j + 1) * vels_wk_stride] -
                            yvel0[i + j * vels_wk_stride] - yvel0[(i + 0) + (j + 1) * vels_wk_stride]) /
                           celldx[i];
      double pgradx = (pressure[(i + 1) + (j + 0) * base_stride] - pressure[(i - 1) + (j + 0) * base_stride]) / (celldx[i] + celldx[i + 1]);
      double pgrady = (pressure[(i + 0) + (j + 1) * base_stride] - pressure[(i + 0) + (j - 1) * base_stride]) / (celldy[j] + celldy[j + 2]);
      double pgradx2 = pgradx * pgradx;
      double pgrady2 = pgrady * pgrady;
      double limiter = ((0.5 * (ugrad) / celldx[i]) * pgradx2 + (0.5 * (vgrad) / celldy[j]) * pgrady2 + strain2 * pgradx * pgrady) /
                       fmax(pgradx2 + pgrady2, g_small);
      if ((limiter > 0.0) || (div >= 0.0)) {
        viscosity[i + j * base_stride] = 0.0;
      } else {
        double dirx = 1.0;
        if (pgradx < 0.0) dirx = -1.0;
        pgradx = dirx * fmax(g_small, fabs(pgradx));
        double diry = 1.0;
        if (pgradx < 0.0) diry = -1.0;
        pgrady = diry * fmax(g_small, fabs(pgrady));
        double pgrad = sqrt(pgradx * pgradx + pgrady * pgrady);
        double xgrad = fabs(celldx[i] * pgrad / pgradx);
        double ygrad = fabs(celldy[j] * pgrad / pgrady);
        double grad = fmin(xgrad, ygrad);
        double grad2 = grad * grad;
        viscosity[i + j * base_stride] = 2.0 * density0[i + j * base_stride] * grad2 * limiter * limiter;
      }
    }
  }
}

//  @brief Driver for the viscosity kernels
//  @author Wayne Gaudin
//  @details Selects the user specified kernel to caluclate the artificial
//  viscosity.
void viscosity(global_variables &globals) {

#if SYNC_BUFFERS
  globals.hostToDevice();
#endif

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    tile_type &t = globals.chunk.tiles[tile];
    viscosity_kernel(globals.context.use_target, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field);
  }

#if SYNC_BUFFERS
  globals.deviceToHost();
#endif
}

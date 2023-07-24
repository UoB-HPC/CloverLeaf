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

//  @brief Fortran viscosity kernel.
//  @author Wayne Gaudin
//  @details Calculates an artificial viscosity using the Wilkin's method to
//  smooth out shock front and prevent oscillations around discontinuities.
//  Only cells in compression will have a non-zero value.

void viscosity_kernel(handler &h, int x_min, int x_max, int y_min, int y_max, clover::Accessor<double, 1, R>::Type celldx,
                      clover::Accessor<double, 1, R>::Type celldy, clover::Accessor<double, 2, R>::Type density0,
                      clover::Accessor<double, 2, R>::Type pressure, clover::Accessor<double, 2, W>::Type viscosity,
                      clover::Accessor<double, 2, R>::Type xvel0, clover::Accessor<double, 2, R>::Type yvel0) {

  // DO k=y_min,y_max
  //   DO j=x_min,x_max
  clover::par_ranged<class viscosity_>(h, {x_min + 1, y_min + 1, x_max + 2, y_max + 2}, [=](id<2> idx) {
    double ugrad = (xvel0[clover::offset(idx, 1, 0)] + xvel0[clover::offset(idx, 1, 1)]) - (xvel0[idx] + xvel0[clover::offset(idx, 0, 1)]);
    double vgrad = (yvel0[clover::offset(idx, 0, 1)] + yvel0[clover::offset(idx, 1, 1)]) - (yvel0[idx] + yvel0[clover::offset(idx, 1, 0)]);
    double div = (celldx[idx[0]] * (ugrad) + celldy[idx[1]] * (vgrad));
    double strain2 =
        0.5 * (xvel0[clover::offset(idx, 0, 1)] + xvel0[clover::offset(idx, 1, 1)] - xvel0[idx] - xvel0[clover::offset(idx, 1, 0)]) /
            celldy[idx[1]] +
        0.5 * (yvel0[clover::offset(idx, 1, 0)] + yvel0[clover::offset(idx, 1, 1)] - yvel0[idx] - yvel0[clover::offset(idx, 0, 1)]) /
            celldx[idx[0]];
    double pgradx = (pressure[clover::offset(idx, 1, 0)] - pressure[clover::offset(idx, -1, 0)]) / (celldx[idx[0]] + celldx[idx[0] + 1]);
    double pgrady = (pressure[clover::offset(idx, 0, 1)] - pressure[clover::offset(idx, 0, -1)]) / (celldy[idx[1]] + celldy[idx[1] + 2]);
    double pgradx2 = pgradx * pgradx;
    double pgrady2 = pgrady * pgrady;
    double limiter = ((0.5 * (ugrad) / celldx[idx[0]]) * pgradx2 + (0.5 * (vgrad) / celldy[idx[1]]) * pgrady2 + strain2 * pgradx * pgrady) /
                     sycl::fmax(pgradx2 + pgrady2, g_small);
    if ((limiter > 0.0) || (div >= 0.0)) {
      viscosity[idx] = 0.0;
    } else {
      double dirx = 1.0;
      if (pgradx < 0.0) dirx = -1.0;
      pgradx = dirx * sycl::fmax(1.0e-16, sycl::fabs(pgradx));
      double diry = 1.0;
      if (pgradx < 0.0) diry = -1.0;
      pgrady = diry * sycl::fmax(1.0e-16, sycl::fabs(pgrady));
      double pgrad = sycl::sqrt(pgradx * pgradx + pgrady * pgrady);
      double xgrad = sycl::fabs(celldx[idx[0]] * pgrad / pgradx);
      double ygrad = sycl::fabs(celldy[idx[1]] * pgrad / pgrady);
      double grad = sycl::fmin(xgrad, ygrad);
      double grad2 = grad * grad;
      viscosity[idx] = 2.0 * density0[idx] * grad2 * limiter * limiter;
    }
  });
}

//  @brief Driver for the viscosity kernels
//  @author Wayne Gaudin
//  @details Selects the user specified kernel to caluclate the artificial
//  viscosity.
void viscosity(global_variables &globals) {

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    tile_type &t = globals.chunk.tiles[tile];
    clover::execute(globals.context.queue, [&](handler &h) {
      viscosity_kernel(h, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax,
                       t.field.celldx.access<R>(h),    //
                       t.field.celldy.access<R>(h),    //
                       t.field.density0.access<R>(h),  //
                       t.field.pressure.access<R>(h),  //
                       t.field.viscosity.access<W>(h), //
                       t.field.xvel0.access<R>(h),     //
                       t.field.yvel0.access<R>(h)      //
      );
    });
  }
}

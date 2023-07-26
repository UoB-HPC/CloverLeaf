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
#include "context.h"

//  @brief Fortran revert kernel.
//  @author Wayne Gaudin
//  @details Takes the half step field data used in the predictor and reverts
//  it to the start of step data, ready for the corrector.
//  Note that this does not seem necessary in this proxy-app but should be
//  left in to remain relevant to the full method.
void revert_kernel(handler &h, int x_min, int x_max, int y_min, int y_max, clover::Accessor<double, 2, R>::Type density0,
                   clover::Accessor<double, 2, W>::Type density1, clover::Accessor<double, 2, R>::Type energy0,
                   clover::Accessor<double, 2, W>::Type energy1) {

  // DO k=y_min,y_max
  //   DO j=x_min,x_max
  clover::par_ranged<class revert>(h, {x_min + 1, y_min + 1, x_max + 2, y_max + 2}, [=](id<2> idx) {
    density1[idx] = density0[idx];
    energy1[idx] = energy0[idx];
  });
}

//  @brief Driver routine for the revert kernels.
//  @author Wayne Gaudin
//  @details Invokes the user specified revert kernel.
void revert(global_variables &globals) {

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {

    clover::execute(globals.context.queue, [&](handler &h) {
      tile_type &t = globals.chunk.tiles[tile];
      revert_kernel(h, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.density0.access<R>(h),
                    t.field.density1.access<W>(h), t.field.energy0.access<R>(h), t.field.energy1.access<W>(h));
    });
  }
}

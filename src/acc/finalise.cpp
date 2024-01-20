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

#include "finalise.h"

void finalise(global_variables &globals) {

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {

    tile_type &t = globals.chunk.tiles[tile];
    field_type &field = t.field;

    double *density0 = field.density0.data;
    double *density1 = field.density1.data;
    double *energy0 = field.energy0.data;
    double *energy1 = field.energy1.data;
    double *pressure = field.pressure.data;
    double *viscosity = field.viscosity.data;
    double *soundspeed = field.soundspeed.data;
    double *yvel0 = field.yvel0.data;
    double *yvel1 = field.yvel1.data;
    double *xvel0 = field.xvel0.data;
    double *xvel1 = field.xvel1.data;
    double *vol_flux_x = field.vol_flux_x.data;
    double *vol_flux_y = field.vol_flux_y.data;
    double *mass_flux_x = field.mass_flux_x.data;
    double *mass_flux_y = field.mass_flux_y.data;
    double *work_array1 = field.work_array1.data;
    double *work_array2 = field.work_array2.data;
    double *work_array3 = field.work_array3.data;
    double *work_array4 = field.work_array4.data;
    double *work_array5 = field.work_array5.data;
    double *work_array6 = field.work_array6.data;
    double *work_array7 = field.work_array7.data;
    double *cellx = field.cellx.data;
    double *celldx = field.celldx.data;
    double *celly = field.celly.data;
    double *celldy = field.celldy.data;
    double *vertexx = field.vertexx.data;
    double *vertexdx = field.vertexdx.data;
    double *vertexy = field.vertexy.data;
    double *vertexdy = field.vertexdy.data;
    double *volume = field.volume.data;
    double *xarea = field.xarea.data;
    double *yarea = field.yarea.data;

#pragma acc exit data delete(density0, density1, energy0, energy1, pressure, viscosity, \
	soundspeed, yvel0, yvel1, xvel0, xvel1, vol_flux_x, vol_flux_y, mass_flux_x, mass_flux_y, \
    work_array1, work_array2, work_array3, work_array4, work_array5, work_array6, work_array7, \
    cellx, celldx, celly, celldy, vertexx, vertexdx, vertexy, vertexdy, volume, xarea, yarea)
  }
}

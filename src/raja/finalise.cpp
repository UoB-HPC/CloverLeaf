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
  for (tile_type &t : globals.chunk.tiles) {
    t.field.density0.release();
    t.field.density1.release();
    t.field.energy0.release();
    t.field.energy1.release();
    t.field.pressure.release();
    t.field.viscosity.release();
    t.field.soundspeed.release();
    t.field.xvel0.release();
    t.field.xvel1.release();
    t.field.yvel0.release();
    t.field.yvel1.release();
    t.field.vol_flux_x.release();
    t.field.mass_flux_x.release();
    t.field.vol_flux_y.release();
    t.field.mass_flux_y.release();
    t.field.work_array1.release();
    t.field.work_array2.release();
    t.field.work_array3.release();
    t.field.work_array4.release();
    t.field.work_array5.release();
    t.field.work_array6.release();
    t.field.work_array7.release();
    t.field.cellx.release();
    t.field.celldx.release();
    t.field.celly.release();
    t.field.celldy.release();
    t.field.vertexx.release();
    t.field.vertexdx.release();
    t.field.vertexy.release();

    t.field.vertexdy.release();
    t.field.volume.release();
    t.field.xarea.release();
    t.field.yarea.release();
  }
}

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
    clover::free(globals.context.queue, t.field.density0, t.field.density1, t.field.energy0, t.field.energy1, t.field.pressure,
                 t.field.viscosity, t.field.soundspeed, t.field.xvel0, t.field.xvel1, t.field.yvel0, t.field.yvel1, t.field.vol_flux_x,
                 t.field.mass_flux_x, t.field.vol_flux_y, t.field.mass_flux_y,
                 t.field.work_array1, // node_flux, stepbymass, volume_change, pre_vol
                 t.field.work_array2, // node_mass_post, post_vol
                 t.field.work_array3, // node_mass_pre,pre_mass
                 t.field.work_array4, // advec_vel, post_mass
                 t.field.work_array5, // mom_flux, advec_vol
                 t.field.work_array6, // pre_vol, post_ener
                 t.field.work_array7, // post_vol, ener_flux
                 t.field.cellx, t.field.celldx, t.field.celly, t.field.celldy, t.field.vertexx, t.field.vertexdx, t.field.vertexy,
                 t.field.vertexdy, t.field.volume, t.field.xarea, t.field.yarea);
  }
}

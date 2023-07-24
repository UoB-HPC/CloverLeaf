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

#include "update_tile_halo.h"
#include "update_tile_halo_kernel.h"

//  @brief Driver for the halo updates
//  @author Wayne Gaudin
//  @details Invokes the kernels for the internal and external halo cells for
//  the fields specified.
void update_tile_halo(global_variables &globals, int fields[NUM_FIELDS], int depth) {

  // Update Top Bottom - Real to Real

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    tile_type &tt = globals.chunk.tiles[tile];
    int t_up = tt.info.tile_neighbours[tile_top];
    int t_down = tt.info.tile_neighbours[tile_bottom];

    if (t_up != external_tile) {
      tile_type &tup = globals.chunk.tiles[t_up];
      update_tile_halo_t_kernel(
          globals, tt.info.t_xmin, tt.info.t_xmax, tt.info.t_ymin, tt.info.t_ymax, tt.field.density0, tt.field.energy0, tt.field.pressure,
          tt.field.viscosity, tt.field.soundspeed, tt.field.density1, tt.field.energy1, tt.field.xvel0, tt.field.yvel0, tt.field.xvel1,
          tt.field.yvel1, tt.field.vol_flux_x, tt.field.vol_flux_y, tt.field.mass_flux_x, tt.field.mass_flux_y, tup.info.t_xmin,
          tup.info.t_xmax, tup.info.t_ymin, tup.info.t_ymax, tup.field.density0, tup.field.energy0, tup.field.pressure, tup.field.viscosity,
          tup.field.soundspeed, tup.field.density1, tup.field.energy1, tup.field.xvel0, tup.field.yvel0, tup.field.xvel1, tup.field.yvel1,
          tup.field.vol_flux_x, tup.field.vol_flux_y, tup.field.mass_flux_x, tup.field.mass_flux_y, fields, depth);
    }

    if (t_down != external_tile) {
      tile_type &tdown = globals.chunk.tiles[t_down];
      update_tile_halo_b_kernel(globals, tt.info.t_xmin, tt.info.t_xmax, tt.info.t_ymin, tt.info.t_ymax, tt.field.density0,
                                tt.field.energy0, tt.field.pressure, tt.field.viscosity, tt.field.soundspeed, tt.field.density1,
                                tt.field.energy1, tt.field.xvel0, tt.field.yvel0, tt.field.xvel1, tt.field.yvel1, tt.field.vol_flux_x,
                                tt.field.vol_flux_y, tt.field.mass_flux_x, tt.field.mass_flux_y, tdown.info.t_xmin, tdown.info.t_xmax,
                                tdown.info.t_ymin, tdown.info.t_ymax, tdown.field.density0, tdown.field.energy0, tdown.field.pressure,
                                tdown.field.viscosity, tdown.field.soundspeed, tdown.field.density1, tdown.field.energy1, tdown.field.xvel0,
                                tdown.field.yvel0, tdown.field.xvel1, tdown.field.yvel1, tdown.field.vol_flux_x, tdown.field.vol_flux_y,
                                tdown.field.mass_flux_x, tdown.field.mass_flux_y, fields, depth);
    }
  }

  // Update Left Right - Ghost, Real, Ghost - > Real

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    tile_type &tt = globals.chunk.tiles[tile];
    int t_left = tt.info.tile_neighbours[tile_left];
    int t_right = tt.info.tile_neighbours[tile_right];

    if (t_left != external_tile) {
      tile_type &tleft = globals.chunk.tiles[t_left];
      update_tile_halo_l_kernel(globals, tt.info.t_xmin, tt.info.t_xmax, tt.info.t_ymin, tt.info.t_ymax, tt.field.density0,
                                tt.field.energy0, tt.field.pressure, tt.field.viscosity, tt.field.soundspeed, tt.field.density1,
                                tt.field.energy1, tt.field.xvel0, tt.field.yvel0, tt.field.xvel1, tt.field.yvel1, tt.field.vol_flux_x,
                                tt.field.vol_flux_y, tt.field.mass_flux_x, tt.field.mass_flux_y, tleft.info.t_xmin, tleft.info.t_xmax,
                                tleft.info.t_ymin, tleft.info.t_ymax, tleft.field.density0, tleft.field.energy0, tleft.field.pressure,
                                tleft.field.viscosity, tleft.field.soundspeed, tleft.field.density1, tleft.field.energy1, tleft.field.xvel0,
                                tleft.field.yvel0, tleft.field.xvel1, tleft.field.yvel1, tleft.field.vol_flux_x, tleft.field.vol_flux_y,
                                tleft.field.mass_flux_x, tleft.field.mass_flux_y, fields, depth);
    }

    if (t_right != external_tile) {
      tile_type &tright = globals.chunk.tiles[t_right];
      update_tile_halo_r_kernel(globals, tt.info.t_xmin, tt.info.t_xmax, tt.info.t_ymin, tt.info.t_ymax, tt.field.density0,
                                tt.field.energy0, tt.field.pressure, tt.field.viscosity, tt.field.soundspeed, tt.field.density1,
                                tt.field.energy1, tt.field.xvel0, tt.field.yvel0, tt.field.xvel1, tt.field.yvel1, tt.field.vol_flux_x,
                                tt.field.vol_flux_y, tt.field.mass_flux_x, tt.field.mass_flux_y, tright.info.t_xmin, tright.info.t_xmax,
                                tright.info.t_ymin, tright.info.t_ymax, tright.field.density0, tright.field.energy0, tright.field.pressure,
                                tright.field.viscosity, tright.field.soundspeed, tright.field.density1, tright.field.energy1,
                                tright.field.xvel0, tright.field.yvel0, tright.field.xvel1, tright.field.yvel1, tright.field.vol_flux_x,
                                tright.field.vol_flux_y, tright.field.mass_flux_x, tright.field.mass_flux_y, fields, depth);
    }
  }
}

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

#include "advection.h"
#include "advec_cell.h"
#include "advec_mom.h"
#include "timer.h"
#include "update_halo.h"

//  @brief Top level advection driver
//  @author Wayne Gaudin
//  @details Controls the advection step and invokes required communications.
void advection(global_variables &globals) {

  int sweep_number = 1;
  int direction;
  if (globals.advect_x) direction = g_xdir;
  if (!globals.advect_x) direction = g_ydir;

  int xvel = g_xdir;
  int yvel = g_ydir;

  int fields[NUM_FIELDS];
  for (int &field : fields)
    field = 0;
  fields[field_energy1] = 1;
  fields[field_density1] = 1;
  fields[field_vol_flux_x] = 1;
  fields[field_vol_flux_y] = 1;
  update_halo(globals, fields, 2);

  double kernel_time = 0;
  if (globals.profiler_on) kernel_time = timer();
  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    advec_cell_driver(globals, tile, sweep_number, direction);
  }

  if (globals.profiler_on) globals.profiler.cell_advection += timer() - kernel_time;

  for (int &field : fields)
    field = 0;
  fields[field_density1] = 1;
  fields[field_energy1] = 1;
  fields[field_xvel1] = 1;
  fields[field_yvel1] = 1;
  fields[field_mass_flux_x] = 1;
  fields[field_mass_flux_y] = 1;
  update_halo(globals, fields, 2);

  if (globals.profiler_on) kernel_time = timer();

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    advec_mom_driver(globals, tile, xvel, direction, sweep_number);
    advec_mom_driver(globals, tile, yvel, direction, sweep_number);
  }

  if (globals.profiler_on) globals.profiler.mom_advection += timer() - kernel_time;

  sweep_number = 2;
  if (globals.advect_x) direction = g_ydir;
  if (!globals.advect_x) direction = g_xdir;

  if (globals.profiler_on) kernel_time = timer();

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    advec_cell_driver(globals, tile, sweep_number, direction);
  }

  if (globals.profiler_on) globals.profiler.cell_advection += timer() - kernel_time;

  for (int &field : fields)
    field = 0;
  fields[field_density1] = 1;
  fields[field_energy1] = 1;
  fields[field_xvel1] = 1;
  fields[field_yvel1] = 1;
  fields[field_mass_flux_x] = 1;
  fields[field_mass_flux_y] = 1;
  update_halo(globals, fields, 2);

  if (globals.profiler_on) kernel_time = timer();

  for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
    advec_mom_driver(globals, tile, xvel, direction, sweep_number);
    advec_mom_driver(globals, tile, yvel, direction, sweep_number);
  }

  if (globals.profiler_on) globals.profiler.mom_advection += timer() - kernel_time;
}

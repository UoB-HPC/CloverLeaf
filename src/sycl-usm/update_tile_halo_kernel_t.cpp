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

#include "context.h"
#include "update_tile_halo_kernel.h"

void update_tile_halo_t_kernel( //
    global_variables &globals, int x_min, int x_max, int y_min, int y_max, clover::Buffer2D<double> &density0,
    clover::Buffer2D<double> &energy0, clover::Buffer2D<double> &pressure, clover::Buffer2D<double> &viscosity,
    clover::Buffer2D<double> &soundspeed, clover::Buffer2D<double> &density1, clover::Buffer2D<double> &energy1,
    clover::Buffer2D<double> &xvel0, clover::Buffer2D<double> &yvel0, clover::Buffer2D<double> &xvel1, clover::Buffer2D<double> &yvel1,
    clover::Buffer2D<double> &vol_flux_x, clover::Buffer2D<double> &vol_flux_y, clover::Buffer2D<double> &mass_flux_x,
    clover::Buffer2D<double> &mass_flux_y, int top_xmin, int top_xmax, int top_ymin, int top_ymax, clover::Buffer2D<double> &top_density0,
    clover::Buffer2D<double> &top_energy0, clover::Buffer2D<double> &top_pressure, clover::Buffer2D<double> &top_viscosity,
    clover::Buffer2D<double> &top_soundspeed, clover::Buffer2D<double> &top_density1, clover::Buffer2D<double> &top_energy1,
    clover::Buffer2D<double> &top_xvel0, clover::Buffer2D<double> &top_yvel0, clover::Buffer2D<double> &top_xvel1,
    clover::Buffer2D<double> &top_yvel1, clover::Buffer2D<double> &top_vol_flux_x, clover::Buffer2D<double> &top_vol_flux_y,
    clover::Buffer2D<double> &top_mass_flux_x, clover::Buffer2D<double> &top_mass_flux_y, const int fields[NUM_FIELDS], int depth) {
  // Density 0
  if (fields[field_density0] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+depth

      clover::par_ranged1(globals.context.queue, Range1d{x_min - depth + 1, x_max + depth + 2},
                          ([=](int j) { density0(j, y_max + 2 + k) = top_density0(j, top_ymin - 1 + 2 + k); }));
    }
  }

  // Density 1
  if (fields[field_density1] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+depth

      clover::par_ranged1(globals.context.queue, Range1d{x_min - depth + 1, x_max + depth + 2},
                          ([=](int j) { density1(j, y_max + 2 + k) = top_density1(j, top_ymin - 1 + 2 + k); }));
    }
  }

  // Energy 0
  if (fields[field_energy0] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+depth

      clover::par_ranged1(globals.context.queue, Range1d{x_min - depth + 1, x_max + depth + 2},
                          ([=](int j) { energy0(j, y_max + 2 + k) = top_energy0(j, top_ymin - 1 + 2 + k); }));
    }
  }

  // Energy 1
  if (fields[field_energy1] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+depth

      clover::par_ranged1(globals.context.queue, Range1d{x_min - depth + 1, x_max + depth + 2},
                          ([=](int j) { energy1(j, y_max + 2 + k) = top_energy1(j, top_ymin - 1 + 2 + k); }));
    }
  }

  // Pressure
  if (fields[field_pressure] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+depth

      clover::par_ranged1(globals.context.queue, Range1d{x_min - depth + 1, x_max + depth + 2},
                          ([=](int j) { pressure(j, y_max + 2 + k) = top_pressure(j, top_ymin - 1 + 2 + k); }));
    }
  }

  // Viscocity
  if (fields[field_viscosity] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+depth

      clover::par_ranged1(globals.context.queue, Range1d{x_min - depth + 1, x_max + depth + 2},
                          ([=](int j) { viscosity(j, y_max + 2 + k) = top_viscosity(j, top_ymin - 1 + 2 + k); }));
    }
  }

  // Soundspeed
  if (fields[field_soundspeed] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+depth

      clover::par_ranged1(globals.context.queue, Range1d{x_min - depth + 1, x_max + depth + 2},
                          ([=](int j) { soundspeed(j, y_max + 2 + k) = top_soundspeed(j, top_ymin - 1 + 2 + k); }));
    }
  }

  // XVEL 0
  if (fields[field_xvel0] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+1+depth

      clover::par_ranged1(globals.context.queue, Range1d{x_min - depth + 1, x_max + 1 + depth + 2},
                          ([=](int j) { xvel0(j, y_max + 1 + 2 + k) = top_xvel0(j, top_ymin + 1 - 1 + 2 + k); }));
    }
  }

  // XVEL 1
  if (fields[field_xvel1] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+1+depth

      clover::par_ranged1(globals.context.queue, Range1d{x_min - depth + 1, x_max + 1 + depth + 2},
                          ([=](int j) { xvel1(j, y_max + 1 + 2 + k) = top_xvel1(j, top_ymin + 1 - 1 + 2 + k); }));
    }
  }

  // YVEL 0
  if (fields[field_yvel0] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+1+depth

      clover::par_ranged1(globals.context.queue, Range1d{x_min - depth + 1, x_max + 1 + depth + 2},
                          ([=](int j) { yvel0(j, y_max + 1 + 2 + k) = top_yvel0(j, top_ymin + 1 - 1 + 2 + k); }));
    }
  }

  // YVEL 1
  if (fields[field_yvel1] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+1+depth

      clover::par_ranged1(globals.context.queue, Range1d{x_min - depth + 1, x_max + 1 + depth + 2},
                          ([=](int j) { yvel1(j, y_max + 1 + 2 + k) = top_yvel1(j, top_ymin + 1 - 1 + 2 + k); }));
    }
  }

  // VOL_FLUX_X
  if (fields[field_vol_flux_x] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+1+depth

      clover::par_ranged1(globals.context.queue, Range1d{x_min - depth + 1, x_max + 1 + depth + 2},
                          ([=](int j) { vol_flux_x(j, y_max + 2 + k) = top_vol_flux_x(j, top_ymin - 1 + 2 + k); }));
    }
  }

  // MASS_FLUX_X
  if (fields[field_mass_flux_x] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+1+depth

      clover::par_ranged1(globals.context.queue, Range1d{x_min - depth + 1, x_max + 1 + depth + 2},
                          ([=](int j) { mass_flux_x(j, y_max + 2 + k) = top_mass_flux_x(j, top_ymin - 1 + 2 + k); }));
    }
  }

  // VOL_FLUX_Y
  if (fields[field_vol_flux_y] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+depth

      clover::par_ranged1(globals.context.queue, Range1d{x_min - depth + 1, x_max + depth + 2},
                          ([=](int j) { vol_flux_y(j, y_max + 1 + 2 + k) = top_vol_flux_y(j, top_ymin + 1 - 1 + 2 + k); }));
    }
  }

  // MASS_FLUX_Y
  if (fields[field_mass_flux_y] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+depth

      clover::par_ranged1(globals.context.queue, Range1d{x_min - depth + 1, x_max + depth + 2},
                          ([=](int j) { mass_flux_y(j, y_max + 1 + 2 + k) = top_mass_flux_y(j, top_ymin + 1 - 1 + 2 + k); }));
    }
  }
}

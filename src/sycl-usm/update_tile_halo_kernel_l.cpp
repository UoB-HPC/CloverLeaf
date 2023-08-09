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

void update_tile_halo_l_kernel( //
    global_variables &globals, int x_min, int x_max, int y_min, int y_max, clover::Buffer2D<double> &density0,
    clover::Buffer2D<double> &energy0, clover::Buffer2D<double> &pressure, clover::Buffer2D<double> &viscosity,
    clover::Buffer2D<double> &soundspeed, clover::Buffer2D<double> &density1, clover::Buffer2D<double> &energy1,
    clover::Buffer2D<double> &xvel0, clover::Buffer2D<double> &yvel0, clover::Buffer2D<double> &xvel1, clover::Buffer2D<double> &yvel1,
    clover::Buffer2D<double> &vol_flux_x, clover::Buffer2D<double> &vol_flux_y, clover::Buffer2D<double> &mass_flux_x,
    clover::Buffer2D<double> &mass_flux_y, int left_xmin, int left_xmax, int left_ymin, int left_ymax,
    clover::Buffer2D<double> &left_density0, clover::Buffer2D<double> &left_energy0, clover::Buffer2D<double> &left_pressure,
    clover::Buffer2D<double> &left_viscosity, clover::Buffer2D<double> &left_soundspeed, clover::Buffer2D<double> &left_density1,
    clover::Buffer2D<double> &left_energy1, clover::Buffer2D<double> &left_xvel0, clover::Buffer2D<double> &left_yvel0,
    clover::Buffer2D<double> &left_xvel1, clover::Buffer2D<double> &left_yvel1, clover::Buffer2D<double> &left_vol_flux_x,
    clover::Buffer2D<double> &left_vol_flux_y, clover::Buffer2D<double> &left_mass_flux_x, clover::Buffer2D<double> &left_mass_flux_y,
    const int fields[NUM_FIELDS], int depth) {
  // Density 0
  if (fields[field_density0] == 1) {
    // DO k=y_min-depth,y_max+depth

    clover::par_ranged1(globals.context.queue, Range1d{y_min - depth + 1, y_max + depth + 2}, ([=](int k) {
                          for (int j = 0; j < depth; ++j) {
                            density0(x_min - j, k) = left_density0(left_xmax + 1 - j, k);
                          }
                        }));
  }

  // Density 1
  if (fields[field_density1] == 1) {
    // DO k=y_min-depth,y_max+depth

    clover::par_ranged1(globals.context.queue, Range1d{y_min - depth + 1, y_max + depth + 2}, ([=](int k) {
                          for (int j = 0; j < depth; ++j) {
                            density1(x_min - j, k) = left_density1(left_xmax + 1 - j, k);
                          }
                        }));
  }

  // Energy 0
  if (fields[field_energy0] == 1) {
    // DO k=y_min-depth,y_max+depth

    clover::par_ranged1(globals.context.queue, Range1d{y_min - depth + 1, y_max + depth + 2}, ([=](int k) {
                          for (int j = 0; j < depth; ++j) {
                            energy0(x_min - j, k) = left_energy0(left_xmax + 1 - j, k);
                          }
                        }));
  }

  // Energy 1
  if (fields[field_energy1] == 1) {
    // DO k=y_min-depth,y_max+depth

    clover::par_ranged1(globals.context.queue, Range1d{y_min - depth + 1, y_max + depth + 2}, ([=](int k) {
                          for (int j = 0; j < depth; ++j) {
                            energy1(x_min - j, k) = left_energy1(left_xmax + 1 - j, k);
                          }
                        }));
  }

  // Pressure
  if (fields[field_pressure] == 1) {
    // DO k=y_min-depth,y_max+depth

    clover::par_ranged1(globals.context.queue, Range1d{y_min - depth + 1, y_max + depth + 2}, ([=](int k) {
                          for (int j = 0; j < depth; ++j) {
                            pressure(x_min - j, k) = left_pressure(left_xmax + 1 - j, k);
                          }
                        }));
  }

  // Viscosity
  if (fields[field_viscosity] == 1) {
    // DO k=y_min-depth,y_max+depth

    clover::par_ranged1(globals.context.queue, Range1d{y_min - depth + 1, y_max + depth + 2}, ([=](int k) {
                          for (int j = 0; j < depth; ++j) {
                            viscosity(x_min - j, k) = left_viscosity(left_xmax + 1 - j, k);
                          }
                        }));
  }

  // Soundspeed
  if (fields[field_soundspeed] == 1) {
    // DO k=y_min-depth,y_max+depth

    clover::par_ranged1(globals.context.queue, Range1d{y_min - depth + 1, y_max + depth + 2}, ([=](int k) {
                          for (int j = 0; j < depth; ++j) {
                            soundspeed(x_min - j, k) = left_soundspeed(left_xmax + 1 - j, k);
                          }
                        }));
  }

  // XVEL 0
  if (fields[field_xvel0] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    clover::par_ranged1(globals.context.queue, Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, ([=](int k) {
                          for (int j = 0; j < depth; ++j) {
                            xvel0(x_min - j, k) = left_xvel0(left_xmax + 1 - j, k);
                          }
                        }));
  }

  // XVEL 1
  if (fields[field_xvel1] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    clover::par_ranged1(globals.context.queue, Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, ([=](int k) {
                          for (int j = 0; j < depth; ++j) {
                            xvel1(x_min - j, k) = left_xvel1(left_xmax + 1 - j, k);
                          }
                        }));
  }

  // YVEL 0
  if (fields[field_yvel0] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    clover::par_ranged1(globals.context.queue, Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, ([=](int k) {
                          for (int j = 0; j < depth; ++j) {
                            yvel0(x_min - j, k) = left_yvel0(left_xmax + 1 - j, k);
                          }
                        }));
  }

  // YVEL 1
  if (fields[field_yvel1] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    clover::par_ranged1(globals.context.queue, Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, ([=](int k) {
                          for (int j = 0; j < depth; ++j) {
                            yvel1(x_min - j, k) = left_yvel1(left_xmax + 1 - j, k);
                          }
                        }));
  }

  // VOL_FLUX_X
  if (fields[field_vol_flux_x] == 1) {
    // DO k=y_min-depth,y_max+depth

    clover::par_ranged1(globals.context.queue, Range1d{y_min - depth + 1, y_max + depth + 2}, ([=](int k) {
                          for (int j = 0; j < depth; ++j) {
                            vol_flux_x(x_min - j, k) = left_vol_flux_x(left_xmax + 1 - j, k);
                          }
                        }));
  }

  // MASS_FLUX_X
  if (fields[field_mass_flux_x] == 1) {
    // DO k=y_min-depth,y_max+depth

    clover::par_ranged1(globals.context.queue, Range1d{y_min - depth + 1, y_max + depth + 2}, ([=](int k) {
                          for (int j = 0; j < depth; ++j) {
                            mass_flux_x(x_min - j, k) = left_mass_flux_x(left_xmax + 1 - j, k);
                          }
                        }));
  }

  // VOL_FLUX_Y
  if (fields[field_vol_flux_y] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    clover::par_ranged1(globals.context.queue, Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, ([=](int k) {
                          for (int j = 0; j < depth; ++j) {
                            vol_flux_y(x_min - j, k) = left_vol_flux_y(left_xmax + 1 - j, k);
                          }
                        }));
  }

  // MASS_FLUX_Y
  if (fields[field_mass_flux_y] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    clover::par_ranged1(globals.context.queue, Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, ([=](int k) {
                          for (int j = 0; j < depth; ++j) {
                            mass_flux_y(x_min - j, k) = left_mass_flux_y(left_xmax + 1 - j, k);
                          }
                        }));
  }
}

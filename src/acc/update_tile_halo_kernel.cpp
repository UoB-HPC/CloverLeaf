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

#include "update_tile_halo_kernel.h"

//   @brief Fortran kernel to update the external halo cells in a chunk.
//   @author Wayne Gaudin
//   @details Updates halo cells for the required fields at the required depth
//   for any halo cells that lie on an external boundary. The location and type
//   of data governs how this is carried out. External boundaries are always
//   reflective.

void update_tile_halo_l_kernel(global_variables &globals, int x_min, int x_max, int y_min, int y_max,
                               clover::Buffer2D<double> &density0_buffer, clover::Buffer2D<double> &energy0_buffer,
                               clover::Buffer2D<double> &pressure_buffer, clover::Buffer2D<double> &viscosity_buffer,
                               clover::Buffer2D<double> &soundspeed_buffer, clover::Buffer2D<double> &density1_buffer,
                               clover::Buffer2D<double> &energy1_buffer, clover::Buffer2D<double> &xvel0_buffer,
                               clover::Buffer2D<double> &yvel0_buffer, clover::Buffer2D<double> &xvel1_buffer,
                               clover::Buffer2D<double> &yvel1_buffer, clover::Buffer2D<double> &vol_flux_x_buffer,
                               clover::Buffer2D<double> &vol_flux_y_buffer, clover::Buffer2D<double> &mass_flux_x_buffer,
                               clover::Buffer2D<double> &mass_flux_y_buffer, int left_xmin, int left_xmax, int left_ymin, int left_ymax,
                               clover::Buffer2D<double> &left_density0_buffer, clover::Buffer2D<double> &left_energy0_buffer,
                               clover::Buffer2D<double> &left_pressure_buffer, clover::Buffer2D<double> &left_viscosity_buffer,
                               clover::Buffer2D<double> &left_soundspeed_buffer, clover::Buffer2D<double> &left_density1_buffer,
                               clover::Buffer2D<double> &left_energy1_buffer, clover::Buffer2D<double> &left_xvel0_buffer,
                               clover::Buffer2D<double> &left_yvel0_buffer, clover::Buffer2D<double> &left_xvel1_buffer,
                               clover::Buffer2D<double> &left_yvel1_buffer, clover::Buffer2D<double> &left_vol_flux_x_buffer,
                               clover::Buffer2D<double> &left_vol_flux_y_buffer, clover::Buffer2D<double> &left_mass_flux_x_buffer,
                               clover::Buffer2D<double> &left_mass_flux_y_buffer, const int fields[NUM_FIELDS], int depth) {
  // Density 0
  if (fields[field_density0] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *density0 = density0_buffer.data;
    size_t density0_sizex = density0_buffer.nX();
    double *left_density0 = left_density0_buffer.data;
    size_t left_density0_sizex = left_density0_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(density0[ : density0_buffer.N()], left_density0[ : left_density0_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        density0[(x_min - j) + (k)*density0_sizex] = left_density0[(left_xmax + 1 - j) + (k)*left_density0_sizex];
      }
    }
  }

  // Density 1
  if (fields[field_density1] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *density1 = density1_buffer.data;
    size_t density1_sizex = density1_buffer.nX();
    double *left_density1 = left_density1_buffer.data;
    size_t left_density1_sizex = left_density1_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(density1[ : density1_buffer.N()], left_density1[ : left_density1_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        density1[(x_min - j) + (k)*density1_sizex] = left_density1[(left_xmax + 1 - j) + (k)*left_density1_sizex];
      }
    }
  }

  // Energy 0
  if (fields[field_energy0] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *energy0 = energy0_buffer.data;
    size_t energy0_sizex = energy0_buffer.nX();
    double *left_energy0 = left_energy0_buffer.data;
    size_t left_energy0_sizex = left_energy0_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(energy0[ : energy0_buffer.N()], left_energy0[ : left_energy0_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        energy0[(x_min - j) + (k)*energy0_sizex] = left_energy0[(left_xmax + 1 - j) + (k)*left_energy0_sizex];
      }
    }
  }

  // Energy 1
  if (fields[field_energy1] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *energy1 = energy1_buffer.data;
    size_t energy1_sizex = energy1_buffer.nX();
    double *left_energy1 = left_energy1_buffer.data;
    size_t left_energy1_sizex = left_energy1_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(energy1[ : energy1_buffer.N()], left_energy1[ : left_energy1_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        energy1[(x_min - j) + (k)*energy1_sizex] = left_energy1[(left_xmax + 1 - j) + (k)*left_energy1_sizex];
      }
    }
  }

  // Pressure
  if (fields[field_pressure] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *pressure = pressure_buffer.data;
    size_t pressure_sizex = pressure_buffer.nX();
    double *left_pressure = left_pressure_buffer.data;
    size_t left_pressure_sizex = left_pressure_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(pressure[ : pressure_buffer.N()], left_pressure[ : left_pressure_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        pressure[(x_min - j) + (k)*pressure_sizex] = left_pressure[(left_xmax + 1 - j) + (k)*left_pressure_sizex];
      }
    }
  }

  // Viscosity
  if (fields[field_viscosity] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *viscosity = viscosity_buffer.data;
    size_t viscosity_sizex = viscosity_buffer.nX();
    double *left_viscosity = left_viscosity_buffer.data;
    size_t left_viscosity_sizex = left_viscosity_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(viscosity[ : viscosity_buffer.N()], left_viscosity[ : left_viscosity_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        viscosity[(x_min - j) + (k)*viscosity_sizex] = left_viscosity[(left_xmax + 1 - j) + (k)*left_viscosity_sizex];
      }
    }
  }

  // Soundspeed
  if (fields[field_soundspeed] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *soundspeed = soundspeed_buffer.data;
    size_t soundspeed_sizex = soundspeed_buffer.nX();
    double *left_soundspeed = left_soundspeed_buffer.data;
    size_t left_soundspeed_sizex = left_soundspeed_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(soundspeed[ : soundspeed_buffer.N()], left_soundspeed[ : left_soundspeed_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        soundspeed[(x_min - j) + (k)*soundspeed_sizex] = left_soundspeed[(left_xmax + 1 - j) + (k)*left_soundspeed_sizex];
      }
    }
  }

  // XVEL 0
  if (fields[field_xvel0] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *xvel0 = xvel0_buffer.data;
    size_t xvel0_sizex = xvel0_buffer.nX();
    double *left_xvel0 = left_xvel0_buffer.data;
    size_t left_xvel0_sizex = left_xvel0_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(xvel0[ : xvel0_buffer.N()], left_xvel0[ : left_xvel0_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        xvel0[(x_min - j) + (k)*xvel0_sizex] = left_xvel0[(left_xmax + 1 - j) + (k)*left_xvel0_sizex];
      }
    }
  }

  // XVEL 1
  if (fields[field_xvel1] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *xvel1 = xvel1_buffer.data;
    size_t xvel1_sizex = xvel1_buffer.nX();
    double *left_xvel1 = left_xvel1_buffer.data;
    size_t left_xvel1_sizex = left_xvel1_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(xvel1[ : xvel1_buffer.N()], left_xvel1[ : left_xvel1_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        xvel1[(x_min - j) + (k)*xvel1_sizex] = left_xvel1[(left_xmax + 1 - j) + (k)*left_xvel1_sizex];
      }
    }
  }

  // YVEL 0
  if (fields[field_yvel0] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *yvel0 = yvel0_buffer.data;
    size_t yvel0_sizex = yvel0_buffer.nX();
    double *left_yvel0 = left_yvel0_buffer.data;
    size_t left_yvel0_sizex = left_yvel0_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(yvel0[ : yvel0_buffer.N()], left_yvel0[ : left_yvel0_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        yvel0[(x_min - j) + (k)*yvel0_sizex] = left_yvel0[(left_xmax + 1 - j) + (k)*left_yvel0_sizex];
      }
    }
  }

  // YVEL 1
  if (fields[field_yvel1] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *yvel1 = yvel1_buffer.data;
    size_t yvel1_sizex = yvel1_buffer.nX();
    double *left_yvel1 = left_yvel1_buffer.data;
    size_t left_yvel1_sizex = left_yvel1_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(yvel1[ : yvel1_buffer.N()], left_yvel1[ : left_yvel1_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        yvel1[(x_min - j) + (k)*yvel1_sizex] = left_yvel1[(left_xmax + 1 - j) + (k)*left_yvel1_sizex];
      }
    }
  }

  // VOL_FLUX_X
  if (fields[field_vol_flux_x] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *vol_flux_x = vol_flux_x_buffer.data;
    size_t vol_flux_x_sizex = vol_flux_x_buffer.nX();
    double *left_vol_flux_x = left_vol_flux_x_buffer.data;
    size_t left_vol_flux_x_sizex = left_vol_flux_x_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(vol_flux_x[ : vol_flux_x_buffer.N()], left_vol_flux_x[ : left_vol_flux_x_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        vol_flux_x[(x_min - j) + (k)*vol_flux_x_sizex] = left_vol_flux_x[(left_xmax + 1 - j) + (k)*left_vol_flux_x_sizex];
      }
    }
  }

  // MASS_FLUX_X
  if (fields[field_mass_flux_x] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *mass_flux_x = mass_flux_x_buffer.data;
    size_t mass_flux_x_sizex = mass_flux_x_buffer.nX();
    double *left_mass_flux_x = left_mass_flux_x_buffer.data;
    size_t left_mass_flux_x_sizex = left_mass_flux_x_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(mass_flux_x[ : mass_flux_x_buffer.N()], left_mass_flux_x[ : left_mass_flux_x_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        mass_flux_x[(x_min - j) + (k)*mass_flux_x_sizex] = left_mass_flux_x[(left_xmax + 1 - j) + (k)*left_mass_flux_x_sizex];
      }
    }
  }

  // VOL_FLUX_Y
  if (fields[field_vol_flux_y] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *vol_flux_y = vol_flux_y_buffer.data;
    size_t vol_flux_y_sizex = vol_flux_y_buffer.nX();
    double *left_vol_flux_y = left_vol_flux_y_buffer.data;
    size_t left_vol_flux_y_sizex = left_vol_flux_y_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(vol_flux_y[ : vol_flux_y_buffer.N()], left_vol_flux_y[ : left_vol_flux_y_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        vol_flux_y[(x_min - j) + (k)*vol_flux_y_sizex] = left_vol_flux_y[(left_xmax + 1 - j) + (k)*left_vol_flux_y_sizex];
      }
    }
  }

  // MASS_FLUX_Y
  if (fields[field_mass_flux_y] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *mass_flux_y = mass_flux_y_buffer.data;
    size_t mass_flux_y_sizex = mass_flux_y_buffer.nX();
    double *left_mass_flux_y = left_mass_flux_y_buffer.data;
    size_t left_mass_flux_y_sizex = left_mass_flux_y_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(mass_flux_y[ : mass_flux_y_buffer.N()], left_mass_flux_y[ : left_mass_flux_y_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        mass_flux_y[(x_min - j) + (k)*mass_flux_y_sizex] = left_mass_flux_y[(left_xmax + 1 - j) + (k)*left_mass_flux_y_sizex];
      }
    }
  }
}

void update_tile_halo_r_kernel(global_variables &globals, int x_min, int x_max, int y_min, int y_max,
                               clover::Buffer2D<double> &density0_buffer, clover::Buffer2D<double> &energy0_buffer,
                               clover::Buffer2D<double> &pressure_buffer, clover::Buffer2D<double> &viscosity_buffer,
                               clover::Buffer2D<double> &soundspeed_buffer, clover::Buffer2D<double> &density1_buffer,
                               clover::Buffer2D<double> &energy1_buffer, clover::Buffer2D<double> &xvel0_buffer,
                               clover::Buffer2D<double> &yvel0_buffer, clover::Buffer2D<double> &xvel1_buffer,
                               clover::Buffer2D<double> &yvel1_buffer, clover::Buffer2D<double> &vol_flux_x_buffer,
                               clover::Buffer2D<double> &vol_flux_y_buffer, clover::Buffer2D<double> &mass_flux_x_buffer,
                               clover::Buffer2D<double> &mass_flux_y_buffer, int right_xmin, int right_xmax, int right_ymin, int right_ymax,
                               clover::Buffer2D<double> &right_density0_buffer, clover::Buffer2D<double> &right_energy0_buffer,
                               clover::Buffer2D<double> &right_pressure_buffer, clover::Buffer2D<double> &right_viscosity_buffer,
                               clover::Buffer2D<double> &right_soundspeed_buffer, clover::Buffer2D<double> &right_density1_buffer,
                               clover::Buffer2D<double> &right_energy1_buffer, clover::Buffer2D<double> &right_xvel0_buffer,
                               clover::Buffer2D<double> &right_yvel0_buffer, clover::Buffer2D<double> &right_xvel1_buffer,
                               clover::Buffer2D<double> &right_yvel1_buffer, clover::Buffer2D<double> &right_vol_flux_x_buffer,
                               clover::Buffer2D<double> &right_vol_flux_y_buffer, clover::Buffer2D<double> &right_mass_flux_x_buffer,
                               clover::Buffer2D<double> &right_mass_flux_y_buffer, const int fields[NUM_FIELDS], int depth) {
{
  // Density 0
  if (fields[field_density0] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *density0 = density0_buffer.data;
    size_t density0_sizex = density0_buffer.nX();
    double *right_density0 = right_density0_buffer.data;
    size_t right_density0_sizex = right_density0_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(density0[ : density0_buffer.N()], right_density0[ : right_density0_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        density0[(x_min - j) + (k)*density0_sizex] = right_density0[(right_xmax + 1 - j) + (k)*right_density0_sizex];
      }
    }
  }

  // Density 1
  if (fields[field_density1] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *density1 = density1_buffer.data;
    size_t density1_sizex = density1_buffer.nX();
    double *right_density1 = right_density1_buffer.data;
    size_t right_density1_sizex = right_density1_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(density1[ : density1_buffer.N()], right_density1[ : right_density1_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        density1[(x_min - j) + (k)*density1_sizex] = right_density1[(right_xmax + 1 - j) + (k)*right_density1_sizex];
      }
    }
  }

  // Energy 0
  if (fields[field_energy0] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *energy0 = energy0_buffer.data;
    size_t energy0_sizex = energy0_buffer.nX();
    double *right_energy0 = right_energy0_buffer.data;
    size_t right_energy0_sizex = right_energy0_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(energy0[ : energy0_buffer.N()], right_energy0[ : right_energy0_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        energy0[(x_min - j) + (k)*energy0_sizex] = right_energy0[(right_xmax + 1 - j) + (k)*right_energy0_sizex];
      }
    }
  }

  // Energy 1
  if (fields[field_energy1] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *energy1 = energy1_buffer.data;
    size_t energy1_sizex = energy1_buffer.nX();
    double *right_energy1 = right_energy1_buffer.data;
    size_t right_energy1_sizex = right_energy1_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(energy1[ : energy1_buffer.N()], right_energy1[ : right_energy1_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        energy1[(x_min - j) + (k)*energy1_sizex] = right_energy1[(right_xmax + 1 - j) + (k)*right_energy1_sizex];
      }
    }
  }

  // Pressure
  if (fields[field_pressure] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *pressure = pressure_buffer.data;
    size_t pressure_sizex = pressure_buffer.nX();
    double *right_pressure = right_pressure_buffer.data;
    size_t right_pressure_sizex = right_pressure_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(pressure[ : pressure_buffer.N()], right_pressure[ : right_pressure_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        pressure[(x_min - j) + (k)*pressure_sizex] = right_pressure[(right_xmax + 1 - j) + (k)*right_pressure_sizex];
      }
    }
  }

  // Viscosity
  if (fields[field_viscosity] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *viscosity = viscosity_buffer.data;
    size_t viscosity_sizex = viscosity_buffer.nX();
    double *right_viscosity = right_viscosity_buffer.data;
    size_t right_viscosity_sizex = right_viscosity_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(viscosity[ : viscosity_buffer.N()], right_viscosity[ : right_viscosity_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        viscosity[(x_min - j) + (k)*viscosity_sizex] = right_viscosity[(right_xmax + 1 - j) + (k)*right_viscosity_sizex];
      }
    }
  }

  // Soundspeed
  if (fields[field_soundspeed] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *soundspeed = soundspeed_buffer.data;
    size_t soundspeed_sizex = soundspeed_buffer.nX();
    double *right_soundspeed = right_soundspeed_buffer.data;
    size_t right_soundspeed_sizex = right_soundspeed_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(soundspeed[ : soundspeed_buffer.N()], right_soundspeed[ : right_soundspeed_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        soundspeed[(x_min - j) + (k)*soundspeed_sizex] = right_soundspeed[(right_xmax + 1 - j) + (k)*right_soundspeed_sizex];
      }
    }
  }

  // XVEL 0
  if (fields[field_xvel0] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *xvel0 = xvel0_buffer.data;
    size_t xvel0_sizex = xvel0_buffer.nX();
    double *right_xvel0 = right_xvel0_buffer.data;
    size_t right_xvel0_sizex = right_xvel0_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(xvel0[ : xvel0_buffer.N()], right_xvel0[ : right_xvel0_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        xvel0[(x_min - j) + (k)*xvel0_sizex] = right_xvel0[(right_xmax + 1 - j) + (k)*right_xvel0_sizex];
      }
    }
  }

  // XVEL 1
  if (fields[field_xvel1] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *xvel1 = xvel1_buffer.data;
    size_t xvel1_sizex = xvel1_buffer.nX();
    double *right_xvel1 = right_xvel1_buffer.data;
    size_t right_xvel1_sizex = right_xvel1_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(xvel1[ : xvel1_buffer.N()], right_xvel1[ : right_xvel1_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        xvel1[(x_min - j) + (k)*xvel1_sizex] = right_xvel1[(right_xmax + 1 - j) + (k)*right_xvel1_sizex];
      }
    }
  }

  // YVEL 0
  if (fields[field_yvel0] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *yvel0 = yvel0_buffer.data;
    size_t yvel0_sizex = yvel0_buffer.nX();
    double *right_yvel0 = right_yvel0_buffer.data;
    size_t right_yvel0_sizex = right_yvel0_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(yvel0[ : yvel0_buffer.N()], right_yvel0[ : right_yvel0_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        yvel0[(x_min - j) + (k)*yvel0_sizex] = right_yvel0[(right_xmax + 1 - j) + (k)*right_yvel0_sizex];
      }
    }
  }

  // YVEL 1
  if (fields[field_yvel1] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *yvel1 = yvel1_buffer.data;
    size_t yvel1_sizex = yvel1_buffer.nX();
    double *right_yvel1 = right_yvel1_buffer.data;
    size_t right_yvel1_sizex = right_yvel1_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(yvel1[ : yvel1_buffer.N()], right_yvel1[ : right_yvel1_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        yvel1[(x_min - j) + (k)*yvel1_sizex] = right_yvel1[(right_xmax + 1 - j) + (k)*right_yvel1_sizex];
      }
    }
  }

  // VOL_FLUX_X
  if (fields[field_vol_flux_x] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *vol_flux_x = vol_flux_x_buffer.data;
    size_t vol_flux_x_sizex = vol_flux_x_buffer.nX();
    double *right_vol_flux_x = right_vol_flux_x_buffer.data;
    size_t right_vol_flux_x_sizex = right_vol_flux_x_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(vol_flux_x[ : vol_flux_x_buffer.N()], right_vol_flux_x[ : right_vol_flux_x_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        vol_flux_x[(x_min - j) + (k)*vol_flux_x_sizex] = right_vol_flux_x[(right_xmax + 1 - j) + (k)*right_vol_flux_x_sizex];
      }
    }
  }

  // MASS_FLUX_X
  if (fields[field_mass_flux_x] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *mass_flux_x = mass_flux_x_buffer.data;
    size_t mass_flux_x_sizex = mass_flux_x_buffer.nX();
    double *right_mass_flux_x = right_mass_flux_x_buffer.data;
    size_t right_mass_flux_x_sizex = right_mass_flux_x_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(mass_flux_x[ : mass_flux_x_buffer.N()], right_mass_flux_x[ : right_mass_flux_x_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        mass_flux_x[(x_min - j) + (k)*mass_flux_x_sizex] = right_mass_flux_x[(right_xmax + 1 - j) + (k)*right_mass_flux_x_sizex];
      }
    }
  }

  // VOL_FLUX_Y
  if (fields[field_vol_flux_y] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *vol_flux_y = vol_flux_y_buffer.data;
    size_t vol_flux_y_sizex = vol_flux_y_buffer.nX();
    double *right_vol_flux_y = right_vol_flux_y_buffer.data;
    size_t right_vol_flux_y_sizex = right_vol_flux_y_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(vol_flux_y[ : vol_flux_y_buffer.N()], right_vol_flux_y[ : right_vol_flux_y_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        vol_flux_y[(x_min - j) + (k)*vol_flux_y_sizex] = right_vol_flux_y[(right_xmax + 1 - j) + (k)*right_vol_flux_y_sizex];
      }
    }
  }

  // MASS_FLUX_Y
  if (fields[field_mass_flux_y] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *mass_flux_y = mass_flux_y_buffer.data;
    size_t mass_flux_y_sizex = mass_flux_y_buffer.nX();
    double *right_mass_flux_y = right_mass_flux_y_buffer.data;
    size_t right_mass_flux_y_sizex = right_mass_flux_y_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(mass_flux_y[ : mass_flux_y_buffer.N()], right_mass_flux_y[ : right_mass_flux_y_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        mass_flux_y[(x_min - j) + (k)*mass_flux_y_sizex] = right_mass_flux_y[(right_xmax + 1 - j) + (k)*right_mass_flux_y_sizex];
      }
    }
  }
}}

//  Top and bottom only do xmin -> xmax
//  This is because the corner ghosts will get communicated in the left right
//  communication

void update_tile_halo_t_kernel(
    global_variables &globals, int x_min, int x_max, int y_min, int y_max, clover::Buffer2D<double> &density0_buffer,
    clover::Buffer2D<double> &energy0_buffer, clover::Buffer2D<double> &pressure_buffer, clover::Buffer2D<double> &viscosity_buffer,
    clover::Buffer2D<double> &soundspeed_buffer, clover::Buffer2D<double> &density1_buffer, clover::Buffer2D<double> &energy1_buffer,
    clover::Buffer2D<double> &xvel0_buffer, clover::Buffer2D<double> &yvel0_buffer, clover::Buffer2D<double> &xvel1_buffer,
    clover::Buffer2D<double> &yvel1_buffer, clover::Buffer2D<double> &vol_flux_x_buffer, clover::Buffer2D<double> &vol_flux_y_buffer,
    clover::Buffer2D<double> &mass_flux_x_buffer, clover::Buffer2D<double> &mass_flux_y_buffer, int top_xmin, int top_xmax, int top_ymin,
    int top_ymax, clover::Buffer2D<double> &top_density0_buffer, clover::Buffer2D<double> &top_energy0_buffer,
    clover::Buffer2D<double> &top_pressure_buffer, clover::Buffer2D<double> &top_viscosity_buffer,
    clover::Buffer2D<double> &top_soundspeed_buffer, clover::Buffer2D<double> &top_density1_buffer,
    clover::Buffer2D<double> &top_energy1_buffer, clover::Buffer2D<double> &top_xvel0_buffer, clover::Buffer2D<double> &top_yvel0_buffer,
    clover::Buffer2D<double> &top_xvel1_buffer, clover::Buffer2D<double> &top_yvel1_buffer, clover::Buffer2D<double> &top_vol_flux_x_buffer,
    clover::Buffer2D<double> &top_vol_flux_y_buffer, clover::Buffer2D<double> &top_mass_flux_x_buffer,
    clover::Buffer2D<double> &top_mass_flux_y_buffer, const int fields[NUM_FIELDS], int depth) {
  // Density 0
  if (fields[field_density0] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *density0 = density0_buffer.data;
    size_t density0_sizex = density0_buffer.nX();
    double *top_density0 = top_density0_buffer.data;
    size_t top_density0_sizex = top_density0_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(density0[ : density0_buffer.N()], top_density0[ : top_density0_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        density0[(x_min - j) + (k)*density0_sizex] = top_density0[(top_xmax + 1 - j) + (k)*top_density0_sizex];
      }
    }
  }

  // Density 1
  if (fields[field_density1] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *density1 = density1_buffer.data;
    size_t density1_sizex = density1_buffer.nX();
    double *top_density1 = top_density1_buffer.data;
    size_t top_density1_sizex = top_density1_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(density1[ : density1_buffer.N()], top_density1[ : top_density1_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        density1[(x_min - j) + (k)*density1_sizex] = top_density1[(top_xmax + 1 - j) + (k)*top_density1_sizex];
      }
    }
  }

  // Energy 0
  if (fields[field_energy0] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *energy0 = energy0_buffer.data;
    size_t energy0_sizex = energy0_buffer.nX();
    double *top_energy0 = top_energy0_buffer.data;
    size_t top_energy0_sizex = top_energy0_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(energy0[ : energy0_buffer.N()], top_energy0[ : top_energy0_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        energy0[(x_min - j) + (k)*energy0_sizex] = top_energy0[(top_xmax + 1 - j) + (k)*top_energy0_sizex];
      }
    }
  }

  // Energy 1
  if (fields[field_energy1] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *energy1 = energy1_buffer.data;
    size_t energy1_sizex = energy1_buffer.nX();
    double *top_energy1 = top_energy1_buffer.data;
    size_t top_energy1_sizex = top_energy1_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(energy1[ : energy1_buffer.N()], top_energy1[ : top_energy1_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        energy1[(x_min - j) + (k)*energy1_sizex] = top_energy1[(top_xmax + 1 - j) + (k)*top_energy1_sizex];
      }
    }
  }

  // Pressure
  if (fields[field_pressure] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *pressure = pressure_buffer.data;
    size_t pressure_sizex = pressure_buffer.nX();
    double *top_pressure = top_pressure_buffer.data;
    size_t top_pressure_sizex = top_pressure_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(pressure[ : pressure_buffer.N()], top_pressure[ : top_pressure_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        pressure[(x_min - j) + (k)*pressure_sizex] = top_pressure[(top_xmax + 1 - j) + (k)*top_pressure_sizex];
      }
    }
  }

  // Viscosity
  if (fields[field_viscosity] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *viscosity = viscosity_buffer.data;
    size_t viscosity_sizex = viscosity_buffer.nX();
    double *top_viscosity = top_viscosity_buffer.data;
    size_t top_viscosity_sizex = top_viscosity_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(viscosity[ : viscosity_buffer.N()], top_viscosity[ : top_viscosity_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        viscosity[(x_min - j) + (k)*viscosity_sizex] = top_viscosity[(top_xmax + 1 - j) + (k)*top_viscosity_sizex];
      }
    }
  }

  // Soundspeed
  if (fields[field_soundspeed] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *soundspeed = soundspeed_buffer.data;
    size_t soundspeed_sizex = soundspeed_buffer.nX();
    double *top_soundspeed = top_soundspeed_buffer.data;
    size_t top_soundspeed_sizex = top_soundspeed_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(soundspeed[ : soundspeed_buffer.N()], top_soundspeed[ : top_soundspeed_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        soundspeed[(x_min - j) + (k)*soundspeed_sizex] = top_soundspeed[(top_xmax + 1 - j) + (k)*top_soundspeed_sizex];
      }
    }
  }

  // XVEL 0
  if (fields[field_xvel0] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *xvel0 = xvel0_buffer.data;
    size_t xvel0_sizex = xvel0_buffer.nX();
    double *top_xvel0 = top_xvel0_buffer.data;
    size_t top_xvel0_sizex = top_xvel0_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(xvel0[ : xvel0_buffer.N()], top_xvel0[ : top_xvel0_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        xvel0[(x_min - j) + (k)*xvel0_sizex] = top_xvel0[(top_xmax + 1 - j) + (k)*top_xvel0_sizex];
      }
    }
  }

  // XVEL 1
  if (fields[field_xvel1] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *xvel1 = xvel1_buffer.data;
    size_t xvel1_sizex = xvel1_buffer.nX();
    double *top_xvel1 = top_xvel1_buffer.data;
    size_t top_xvel1_sizex = top_xvel1_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(xvel1[ : xvel1_buffer.N()], top_xvel1[ : top_xvel1_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        xvel1[(x_min - j) + (k)*xvel1_sizex] = top_xvel1[(top_xmax + 1 - j) + (k)*top_xvel1_sizex];
      }
    }
  }

  // YVEL 0
  if (fields[field_yvel0] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *yvel0 = yvel0_buffer.data;
    size_t yvel0_sizex = yvel0_buffer.nX();
    double *top_yvel0 = top_yvel0_buffer.data;
    size_t top_yvel0_sizex = top_yvel0_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(yvel0[ : yvel0_buffer.N()], top_yvel0[ : top_yvel0_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        yvel0[(x_min - j) + (k)*yvel0_sizex] = top_yvel0[(top_xmax + 1 - j) + (k)*top_yvel0_sizex];
      }
    }
  }

  // YVEL 1
  if (fields[field_yvel1] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *yvel1 = yvel1_buffer.data;
    size_t yvel1_sizex = yvel1_buffer.nX();
    double *top_yvel1 = top_yvel1_buffer.data;
    size_t top_yvel1_sizex = top_yvel1_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(yvel1[ : yvel1_buffer.N()], top_yvel1[ : top_yvel1_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        yvel1[(x_min - j) + (k)*yvel1_sizex] = top_yvel1[(top_xmax + 1 - j) + (k)*top_yvel1_sizex];
      }
    }
  }

  // VOL_FLUX_X
  if (fields[field_vol_flux_x] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *vol_flux_x = vol_flux_x_buffer.data;
    size_t vol_flux_x_sizex = vol_flux_x_buffer.nX();
    double *top_vol_flux_x = top_vol_flux_x_buffer.data;
    size_t top_vol_flux_x_sizex = top_vol_flux_x_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(vol_flux_x[ : vol_flux_x_buffer.N()], top_vol_flux_x[ : top_vol_flux_x_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        vol_flux_x[(x_min - j) + (k)*vol_flux_x_sizex] = top_vol_flux_x[(top_xmax + 1 - j) + (k)*top_vol_flux_x_sizex];
      }
    }
  }

  // MASS_FLUX_X
  if (fields[field_mass_flux_x] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *mass_flux_x = mass_flux_x_buffer.data;
    size_t mass_flux_x_sizex = mass_flux_x_buffer.nX();
    double *top_mass_flux_x = top_mass_flux_x_buffer.data;
    size_t top_mass_flux_x_sizex = top_mass_flux_x_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(mass_flux_x[ : mass_flux_x_buffer.N()], top_mass_flux_x[ : top_mass_flux_x_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        mass_flux_x[(x_min - j) + (k)*mass_flux_x_sizex] = top_mass_flux_x[(top_xmax + 1 - j) + (k)*top_mass_flux_x_sizex];
      }
    }
  }

  // VOL_FLUX_Y
  if (fields[field_vol_flux_y] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *vol_flux_y = vol_flux_y_buffer.data;
    size_t vol_flux_y_sizex = vol_flux_y_buffer.nX();
    double *top_vol_flux_y = top_vol_flux_y_buffer.data;
    size_t top_vol_flux_y_sizex = top_vol_flux_y_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(vol_flux_y[ : vol_flux_y_buffer.N()], top_vol_flux_y[ : top_vol_flux_y_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        vol_flux_y[(x_min - j) + (k)*vol_flux_y_sizex] = top_vol_flux_y[(top_xmax + 1 - j) + (k)*top_vol_flux_y_sizex];
      }
    }
  }

  // MASS_FLUX_Y
  if (fields[field_mass_flux_y] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *mass_flux_y = mass_flux_y_buffer.data;
    size_t mass_flux_y_sizex = mass_flux_y_buffer.nX();
    double *top_mass_flux_y = top_mass_flux_y_buffer.data;
    size_t top_mass_flux_y_sizex = top_mass_flux_y_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(mass_flux_y[ : mass_flux_y_buffer.N()], top_mass_flux_y[ : top_mass_flux_y_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        mass_flux_y[(x_min - j) + (k)*mass_flux_y_sizex] = top_mass_flux_y[(top_xmax + 1 - j) + (k)*top_mass_flux_y_sizex];
      }
    }
  }
}

void update_tile_halo_b_kernel(
    global_variables &globals, int x_min, int x_max, int y_min, int y_max, clover::Buffer2D<double> &density0_buffer,
    clover::Buffer2D<double> &energy0_buffer, clover::Buffer2D<double> &pressure_buffer, clover::Buffer2D<double> &viscosity_buffer,
    clover::Buffer2D<double> &soundspeed_buffer, clover::Buffer2D<double> &density1_buffer, clover::Buffer2D<double> &energy1_buffer,
    clover::Buffer2D<double> &xvel0_buffer, clover::Buffer2D<double> &yvel0_buffer, clover::Buffer2D<double> &xvel1_buffer,
    clover::Buffer2D<double> &yvel1_buffer, clover::Buffer2D<double> &vol_flux_x_buffer, clover::Buffer2D<double> &vol_flux_y_buffer,
    clover::Buffer2D<double> &mass_flux_x_buffer, clover::Buffer2D<double> &mass_flux_y_buffer, int bottom_xmin, int bottom_xmax,
    int bottom_ymin, int bottom_ymax, clover::Buffer2D<double> &bottom_density0_buffer, clover::Buffer2D<double> &bottom_energy0_buffer,
    clover::Buffer2D<double> &bottom_pressure_buffer, clover::Buffer2D<double> &bottom_viscosity_buffer,
    clover::Buffer2D<double> &bottom_soundspeed_buffer, clover::Buffer2D<double> &bottom_density1_buffer,
    clover::Buffer2D<double> &bottom_energy1_buffer, clover::Buffer2D<double> &bottom_xvel0_buffer,
    clover::Buffer2D<double> &bottom_yvel0_buffer, clover::Buffer2D<double> &bottom_xvel1_buffer,
    clover::Buffer2D<double> &bottom_yvel1_buffer, clover::Buffer2D<double> &bottom_vol_flux_x_buffer,
    clover::Buffer2D<double> &bottom_vol_flux_y_buffer, clover::Buffer2D<double> &bottom_mass_flux_x_buffer,
    clover::Buffer2D<double> &bottom_mass_flux_y_buffer, const int fields[NUM_FIELDS], int depth) {
  // Density 0
  if (fields[field_density0] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *density0 = density0_buffer.data;
    size_t density0_sizex = density0_buffer.nX();
    double *bottom_density0 = bottom_density0_buffer.data;
    size_t bottom_density0_sizex = bottom_density0_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(density0[ : density0_buffer.N()], bottom_density0[ : bottom_density0_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        density0[(x_min - j) + (k)*density0_sizex] = bottom_density0[(bottom_xmax + 1 - j) + (k)*bottom_density0_sizex];
      }
    }
  }

  // Density 1
  if (fields[field_density1] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *density1 = density1_buffer.data;
    size_t density1_sizex = density1_buffer.nX();
    double *bottom_density1 = bottom_density1_buffer.data;
    size_t bottom_density1_sizex = bottom_density1_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(density1[ : density1_buffer.N()], bottom_density1[ : bottom_density1_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        density1[(x_min - j) + (k)*density1_sizex] = bottom_density1[(bottom_xmax + 1 - j) + (k)*bottom_density1_sizex];
      }
    }
  }

  // Energy 0
  if (fields[field_energy0] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *energy0 = energy0_buffer.data;
    size_t energy0_sizex = energy0_buffer.nX();
    double *bottom_energy0 = bottom_energy0_buffer.data;
    size_t bottom_energy0_sizex = bottom_energy0_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(energy0[ : energy0_buffer.N()], bottom_energy0[ : bottom_energy0_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        energy0[(x_min - j) + (k)*energy0_sizex] = bottom_energy0[(bottom_xmax + 1 - j) + (k)*bottom_energy0_sizex];
      }
    }
  }

  // Energy 1
  if (fields[field_energy1] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *energy1 = energy1_buffer.data;
    size_t energy1_sizex = energy1_buffer.nX();
    double *bottom_energy1 = bottom_energy1_buffer.data;
    size_t bottom_energy1_sizex = bottom_energy1_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(energy1[ : energy1_buffer.N()], bottom_energy1[ : bottom_energy1_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        energy1[(x_min - j) + (k)*energy1_sizex] = bottom_energy1[(bottom_xmax + 1 - j) + (k)*bottom_energy1_sizex];
      }
    }
  }

  // Pressure
  if (fields[field_pressure] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *pressure = pressure_buffer.data;
    size_t pressure_sizex = pressure_buffer.nX();
    double *bottom_pressure = bottom_pressure_buffer.data;
    size_t bottom_pressure_sizex = bottom_pressure_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(pressure[ : pressure_buffer.N()], bottom_pressure[ : bottom_pressure_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        pressure[(x_min - j) + (k)*pressure_sizex] = bottom_pressure[(bottom_xmax + 1 - j) + (k)*bottom_pressure_sizex];
      }
    }
  }

  // Viscosity
  if (fields[field_viscosity] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *viscosity = viscosity_buffer.data;
    size_t viscosity_sizex = viscosity_buffer.nX();
    double *bottom_viscosity = bottom_viscosity_buffer.data;
    size_t bottom_viscosity_sizex = bottom_viscosity_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(viscosity[ : viscosity_buffer.N()], bottom_viscosity[ : bottom_viscosity_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        viscosity[(x_min - j) + (k)*viscosity_sizex] = bottom_viscosity[(bottom_xmax + 1 - j) + (k)*bottom_viscosity_sizex];
      }
    }
  }

  // Soundspeed
  if (fields[field_soundspeed] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *soundspeed = soundspeed_buffer.data;
    size_t soundspeed_sizex = soundspeed_buffer.nX();
    double *bottom_soundspeed = bottom_soundspeed_buffer.data;
    size_t bottom_soundspeed_sizex = bottom_soundspeed_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(soundspeed[ : soundspeed_buffer.N()], bottom_soundspeed[ : bottom_soundspeed_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        soundspeed[(x_min - j) + (k)*soundspeed_sizex] = bottom_soundspeed[(bottom_xmax + 1 - j) + (k)*bottom_soundspeed_sizex];
      }
    }
  }

  // XVEL 0
  if (fields[field_xvel0] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *xvel0 = xvel0_buffer.data;
    size_t xvel0_sizex = xvel0_buffer.nX();
    double *bottom_xvel0 = bottom_xvel0_buffer.data;
    size_t bottom_xvel0_sizex = bottom_xvel0_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(xvel0[ : xvel0_buffer.N()], bottom_xvel0[ : bottom_xvel0_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        xvel0[(x_min - j) + (k)*xvel0_sizex] = bottom_xvel0[(bottom_xmax + 1 - j) + (k)*bottom_xvel0_sizex];
      }
    }
  }

  // XVEL 1
  if (fields[field_xvel1] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *xvel1 = xvel1_buffer.data;
    size_t xvel1_sizex = xvel1_buffer.nX();
    double *bottom_xvel1 = bottom_xvel1_buffer.data;
    size_t bottom_xvel1_sizex = bottom_xvel1_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(xvel1[ : xvel1_buffer.N()], bottom_xvel1[ : bottom_xvel1_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        xvel1[(x_min - j) + (k)*xvel1_sizex] = bottom_xvel1[(bottom_xmax + 1 - j) + (k)*bottom_xvel1_sizex];
      }
    }
  }

  // YVEL 0
  if (fields[field_yvel0] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *yvel0 = yvel0_buffer.data;
    size_t yvel0_sizex = yvel0_buffer.nX();
    double *bottom_yvel0 = bottom_yvel0_buffer.data;
    size_t bottom_yvel0_sizex = bottom_yvel0_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(yvel0[ : yvel0_buffer.N()], bottom_yvel0[ : bottom_yvel0_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        yvel0[(x_min - j) + (k)*yvel0_sizex] = bottom_yvel0[(bottom_xmax + 1 - j) + (k)*bottom_yvel0_sizex];
      }
    }
  }

  // YVEL 1
  if (fields[field_yvel1] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *yvel1 = yvel1_buffer.data;
    size_t yvel1_sizex = yvel1_buffer.nX();
    double *bottom_yvel1 = bottom_yvel1_buffer.data;
    size_t bottom_yvel1_sizex = bottom_yvel1_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(yvel1[ : yvel1_buffer.N()], bottom_yvel1[ : bottom_yvel1_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        yvel1[(x_min - j) + (k)*yvel1_sizex] = bottom_yvel1[(bottom_xmax + 1 - j) + (k)*bottom_yvel1_sizex];
      }
    }
  }

  // VOL_FLUX_X
  if (fields[field_vol_flux_x] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *vol_flux_x = vol_flux_x_buffer.data;
    size_t vol_flux_x_sizex = vol_flux_x_buffer.nX();
    double *bottom_vol_flux_x = bottom_vol_flux_x_buffer.data;
    size_t bottom_vol_flux_x_sizex = bottom_vol_flux_x_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(vol_flux_x[ : vol_flux_x_buffer.N()], bottom_vol_flux_x[ : bottom_vol_flux_x_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        vol_flux_x[(x_min - j) + (k)*vol_flux_x_sizex] = bottom_vol_flux_x[(bottom_xmax + 1 - j) + (k)*bottom_vol_flux_x_sizex];
      }
    }
  }

  // MASS_FLUX_X
  if (fields[field_mass_flux_x] == 1) {
    // DO k=y_min-depth,y_max+depth

    double *mass_flux_x = mass_flux_x_buffer.data;
    size_t mass_flux_x_sizex = mass_flux_x_buffer.nX();
    double *bottom_mass_flux_x = bottom_mass_flux_x_buffer.data;
    size_t bottom_mass_flux_x_sizex = bottom_mass_flux_x_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(mass_flux_x[ : mass_flux_x_buffer.N()], bottom_mass_flux_x[ : bottom_mass_flux_x_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        mass_flux_x[(x_min - j) + (k)*mass_flux_x_sizex] = bottom_mass_flux_x[(bottom_xmax + 1 - j) + (k)*bottom_mass_flux_x_sizex];
      }
    }
  }

  // VOL_FLUX_Y
  if (fields[field_vol_flux_y] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *vol_flux_y = vol_flux_y_buffer.data;
    size_t vol_flux_y_sizex = vol_flux_y_buffer.nX();
    double *bottom_vol_flux_y = bottom_vol_flux_y_buffer.data;
    size_t bottom_vol_flux_y_sizex = bottom_vol_flux_y_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(vol_flux_y[ : vol_flux_y_buffer.N()], bottom_vol_flux_y[ : bottom_vol_flux_y_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        vol_flux_y[(x_min - j) + (k)*vol_flux_y_sizex] = bottom_vol_flux_y[(bottom_xmax + 1 - j) + (k)*bottom_vol_flux_y_sizex];
      }
    }
  }

  // MASS_FLUX_Y
  if (fields[field_mass_flux_y] == 1) {
    // DO k=y_min-depth,y_max+1+depth

    double *mass_flux_y = mass_flux_y_buffer.data;
    size_t mass_flux_y_sizex = mass_flux_y_buffer.nX();
    double *bottom_mass_flux_y = bottom_mass_flux_y_buffer.data;
    size_t bottom_mass_flux_y_sizex = bottom_mass_flux_y_buffer.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(mass_flux_y[ : mass_flux_y_buffer.N()], bottom_mass_flux_y[ : bottom_mass_flux_y_buffer.N()])
    for (int k = (y_min - depth + 1); k < (y_max + 1 + depth + 2); k++) {
      for (int j = 0; j < depth; ++j) {
        mass_flux_y[(x_min - j) + (k)*mass_flux_y_sizex] = bottom_mass_flux_y[(bottom_xmax + 1 - j) + (k)*bottom_mass_flux_y_sizex];
      }
    }
  }
}

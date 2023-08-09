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

  // Density 0
  if (fields[field_density0] == 1) {
    // DO k=y_min-depth,y_max+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto right_density0 = right_density0_buffer.access<R>(h);
      auto density0 = density0_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_r_density0>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          density0[x_max + 2 + j][k[0]] = right_density0[right_xmin - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // Density 1
  if (fields[field_density1] == 1) {
    // DO k=y_min-depth,y_max+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto right_density1 = right_density1_buffer.access<R>(h);
      auto density1 = density1_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_r_density1>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          density1[x_max + 2 + j][k[0]] = right_density1[right_xmin - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // Energy 0
  if (fields[field_energy0] == 1) {
    // DO k=y_min-depth,y_max+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto right_energy0 = right_energy0_buffer.access<R>(h);
      auto energy0 = energy0_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_r_energy0>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          energy0[x_max + 2 + j][k[0]] = right_energy0[right_xmin - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // Energy 1
  if (fields[field_energy1] == 1) {
    // DO k=y_min-depth,y_max+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto right_energy1 = right_energy1_buffer.access<R>(h);
      auto energy1 = energy1_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_r_energy1>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          energy1[x_max + 2 + j][k[0]] = right_energy1[right_xmin - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // Pressure
  if (fields[field_pressure] == 1) {
    // DO k=y_min-depth,y_max+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto pressure = pressure_buffer.access<W>(h);
      auto right_pressure = right_pressure_buffer.access<R>(h);
      clover::par_ranged<class upd_halo_r_pressure>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          pressure[x_max + 2 + j][k[0]] = right_pressure[right_xmin - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // Viscosity
  if (fields[field_viscosity] == 1) {
    // DO k=y_min-depth,y_max+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto right_viscosity = right_viscosity_buffer.access<R>(h);
      auto viscosity = viscosity_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_r_viscosity>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          viscosity[x_max + 2 + j][k[0]] = right_viscosity[right_xmin - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // Soundspeed
  if (fields[field_soundspeed] == 1) {
    // DO k=y_min-depth,y_max+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto right_soundspeed = right_soundspeed_buffer.access<R>(h);
      auto soundspeed = soundspeed_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_r_soundspeed>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          soundspeed[x_max + 2 + j][k[0]] = right_soundspeed[right_xmin - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // XVEL 0
  if (fields[field_xvel0] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto right_xvel0 = right_xvel0_buffer.access<R>(h);
      auto xvel0 = xvel0_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_r_xvel0>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          xvel0[x_max + 1 + 2 + j][k[0]] = right_xvel0[right_xmin + 1 - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // XVEL 1
  if (fields[field_xvel1] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto right_xvel1 = right_xvel1_buffer.access<R>(h);
      auto xvel1 = xvel1_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_r_xvel1>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          xvel1[x_max + 1 + 2 + j][k[0]] = right_xvel1[right_xmin + 1 - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // YVEL 0
  if (fields[field_yvel0] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto right_yvel0 = right_yvel0_buffer.access<R>(h);
      auto yvel0 = yvel0_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_r_yvel0>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          yvel0[x_max + 1 + 2 + j][k[0]] = right_yvel0[right_xmin + 1 - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // YVEL 1
  if (fields[field_yvel1] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto right_yvel1 = right_yvel1_buffer.access<R>(h);
      auto yvel1 = yvel1_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_r_yvel1>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          yvel1[x_max + 1 + 2 + j][k[0]] = right_yvel1[right_xmin + 1 - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // VOL_FLUX_X
  if (fields[field_vol_flux_x] == 1) {
    // DO k=y_min-depth,y_max+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto right_vol_flux_x = right_vol_flux_x_buffer.access<R>(h);
      auto vol_flux_x = vol_flux_x_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_r_vol_flux_x>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          vol_flux_x[x_max + 1 + 2 + j][k[0]] = right_vol_flux_x[right_xmin + 1 - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // MASS_FLUX_X
  if (fields[field_mass_flux_x] == 1) {
    // DO k=y_min-depth,y_max+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto right_mass_flux_x = right_mass_flux_x_buffer.access<R>(h);
      auto mass_flux_x = mass_flux_x_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_r_mass_flux_x>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          mass_flux_x[x_max + 1 + 2 + j][k[0]] = right_mass_flux_x[right_xmin + 1 - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // VOL_FLUX_Y
  if (fields[field_vol_flux_y] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto right_vol_flux_y = right_vol_flux_y_buffer.access<R>(h);
      auto vol_flux_y = vol_flux_y_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_r_vol_flux_y>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          vol_flux_y[x_max + 2 + j][k[0]] = right_vol_flux_y[right_xmin - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // MASS_FLUX_Y
  if (fields[field_mass_flux_y] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto right_mass_flux_y = right_mass_flux_y_buffer.access<R>(h);
      auto mass_flux_y = mass_flux_y_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_r_mass_flux_y>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          mass_flux_y[x_max + 2 + j][k[0]] = right_mass_flux_y[right_xmin - 1 + 2 + j][k[0]];
        }
      });
    });
  }
}

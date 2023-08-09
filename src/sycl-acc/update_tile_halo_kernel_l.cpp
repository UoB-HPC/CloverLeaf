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
    clover::execute(globals.context.queue, [&](handler &h) {
      auto left_density0 = left_density0_buffer.access<R>(h);
      auto density0 = density0_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_l_density0>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          density0[x_min - j][k[0]] = left_density0[left_xmax + 1 - j][k[0]];
        }
      });
    });
  }

  // Density 1
  if (fields[field_density1] == 1) {
    // DO k=y_min-depth,y_max+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto left_density1 = left_density1_buffer.access<R>(h);
      auto density1 = density1_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_l_density1>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          density1[x_min - j][k[0]] = left_density1[left_xmax + 1 - j][k[0]];
        }
      });
    });
  }

  // Energy 0
  if (fields[field_energy0] == 1) {
    // DO k=y_min-depth,y_max+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto left_energy0 = left_energy0_buffer.access<R>(h);
      auto energy0 = energy0_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_l_energy0>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          energy0[x_min - j][k[0]] = left_energy0[left_xmax + 1 - j][k[0]];
        }
      });
    });
  }

  // Energy 1
  if (fields[field_energy1] == 1) {
    // DO k=y_min-depth,y_max+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto left_energy1 = left_energy1_buffer.access<R>(h);
      auto energy1 = energy1_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_l_energy1>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          energy1[x_min - j][k[0]] = left_energy1[left_xmax + 1 - j][k[0]];
        }
      });
    });
  }

  // Pressure
  if (fields[field_pressure] == 1) {
    // DO k=y_min-depth,y_max+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto left_pressure = left_pressure_buffer.access<R>(h);
      auto pressure = pressure_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_l_pressure>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          pressure[x_min - j][k[0]] = left_pressure[left_xmax + 1 - j][k[0]];
        }
      });
    });
  }

  // Viscosity
  if (fields[field_viscosity] == 1) {
    // DO k=y_min-depth,y_max+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto left_viscosity = left_viscosity_buffer.access<R>(h);
      auto viscosity = viscosity_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_l_viscosity>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          viscosity[x_min - j][k[0]] = left_viscosity[left_xmax + 1 - j][k[0]];
        }
      });
    });
  }

  // Soundspeed
  if (fields[field_soundspeed] == 1) {
    // DO k=y_min-depth,y_max+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto left_soundspeed = left_soundspeed_buffer.access<R>(h);
      auto soundspeed = soundspeed_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_l_soundspeed>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          soundspeed[x_min - j][k[0]] = left_soundspeed[left_xmax + 1 - j][k[0]];
        }
      });
    });
  }

  // XVEL 0
  if (fields[field_xvel0] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto left_xvel0 = left_xvel0_buffer.access<R>(h);
      auto xvel0 = xvel0_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_l_xvel0>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          xvel0[x_min - j][k[0]] = left_xvel0[left_xmax + 1 - j][k[0]];
        }
      });
    });
  }

  // XVEL 1
  if (fields[field_xvel1] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto left_xvel1 = left_xvel1_buffer.access<R>(h);
      auto xvel1 = xvel1_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_l_xvel1>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          xvel1[x_min - j][k[0]] = left_xvel1[left_xmax + 1 - j][k[0]];
        }
      });
    });
  }

  // YVEL 0
  if (fields[field_yvel0] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto left_yvel0 = left_yvel0_buffer.access<R>(h);
      auto yvel0 = yvel0_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_l_yvel0>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          yvel0[x_min - j][k[0]] = left_yvel0[left_xmax + 1 - j][k[0]];
        }
      });
    });
  }

  // YVEL 1
  if (fields[field_yvel1] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto left_yvel1 = left_yvel1_buffer.access<R>(h);
      auto yvel1 = yvel1_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_l_yvel1>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          yvel1[x_min - j][k[0]] = left_yvel1[left_xmax + 1 - j][k[0]];
        }
      });
    });
  }

  // VOL_FLUX_X
  if (fields[field_vol_flux_x] == 1) {
    // DO k=y_min-depth,y_max+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto left_vol_flux_x = left_vol_flux_x_buffer.access<R>(h);
      auto vol_flux_x = vol_flux_x_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_l_vol_flux_x>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          vol_flux_x[x_min - j][k[0]] = left_vol_flux_x[left_xmax + 1 - j][k[0]];
        }
      });
    });
  }

  // MASS_FLUX_X
  if (fields[field_mass_flux_x] == 1) {
    // DO k=y_min-depth,y_max+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto left_mass_flux_x = left_mass_flux_x_buffer.access<R>(h);
      auto mass_flux_x = mass_flux_x_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_l_mass_flux_x>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          mass_flux_x[x_min - j][k[0]] = left_mass_flux_x[left_xmax + 1 - j][k[0]];
        }
      });
    });
  }

  // VOL_FLUX_Y
  if (fields[field_vol_flux_y] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto left_vol_flux_y = left_vol_flux_y_buffer.access<R>(h);
      auto vol_flux_y = vol_flux_y_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_l_vol_flux_y>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          vol_flux_y[x_min - j][k[0]] = left_vol_flux_y[left_xmax + 1 - j][k[0]];
        }
      });
    });
  }

  // MASS_FLUX_Y
  if (fields[field_mass_flux_y] == 1) {
    // DO k=y_min-depth,y_max+1+depth
    clover::execute(globals.context.queue, [&](handler &h) {
      auto left_mass_flux_y = left_mass_flux_y_buffer.access<R>(h);
      auto mass_flux_y = mass_flux_y_buffer.access<W>(h);
      clover::par_ranged<class upd_halo_l_mass_flux_y>(h, {y_min - depth + 1, y_max + 1 + depth + 2}, [=](id<1> k) {
        for (int j = 0; j < depth; ++j) {
          mass_flux_y[x_min - j][k[0]] = left_mass_flux_y[left_xmax + 1 - j][k[0]];
        }
      });
    });
  }
}

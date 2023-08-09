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
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+depth
      clover::execute(globals.context.queue, [&](handler &h) {
        auto top_density0 = top_density0_buffer.access<R>(h);
        auto density0 = density0_buffer.access<W>(h);
        clover::par_ranged<class upd_halo_t_density0>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          density0[j[0]][y_max + 2 + k] = top_density0[j[0]][top_ymin - 1 + 2 + k];
        });
      });
    }
  }

  // Density 1
  if (fields[field_density1] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+depth
      clover::execute(globals.context.queue, [&](handler &h) {
        auto top_density1 = top_density1_buffer.access<R>(h);
        auto density1 = density1_buffer.access<W>(h);
        clover::par_ranged<class upd_halo_t_density1>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          density1[j[0]][y_max + 2 + k] = top_density1[j[0]][top_ymin - 1 + 2 + k];
        });
      });
    }
  }

  // Energy 0
  if (fields[field_energy0] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+depth
      clover::execute(globals.context.queue, [&](handler &h) {
        auto top_energy0 = top_energy0_buffer.access<R>(h);
        auto energy0 = energy0_buffer.access<W>(h);
        clover::par_ranged<class upd_halo_t_energy0>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          energy0[j[0]][y_max + 2 + k] = top_energy0[j[0]][top_ymin - 1 + 2 + k];
        });
      });
    }
  }

  // Energy 1
  if (fields[field_energy1] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+depth
      clover::execute(globals.context.queue, [&](handler &h) {
        auto top_energy1 = top_energy1_buffer.access<R>(h);
        auto energy1 = energy1_buffer.access<W>(h);
        clover::par_ranged<class upd_halo_t_energy1>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          energy1[j[0]][y_max + 2 + k] = top_energy1[j[0]][top_ymin - 1 + 2 + k];
        });
      });
    }
  }

  // Pressure
  if (fields[field_pressure] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+depth
      clover::execute(globals.context.queue, [&](handler &h) {
        auto top_pressure = top_pressure_buffer.access<R>(h);
        auto pressure = pressure_buffer.access<W>(h);
        clover::par_ranged<class upd_halo_t_pressure>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          pressure[j[0]][y_max + 2 + k] = top_pressure[j[0]][top_ymin - 1 + 2 + k];
        });
      });
    }
  }

  // Viscocity
  if (fields[field_viscosity] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+depth
      clover::execute(globals.context.queue, [&](handler &h) {
        auto top_viscosity = top_viscosity_buffer.access<R>(h);
        auto viscosity = viscosity_buffer.access<W>(h);
        clover::par_ranged<class upd_halo_t_viscosity>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          viscosity[j[0]][y_max + 2 + k] = top_viscosity[j[0]][top_ymin - 1 + 2 + k];
        });
      });
    }
  }

  // Soundspeed
  if (fields[field_soundspeed] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+depth
      clover::execute(globals.context.queue, [&](handler &h) {
        auto top_soundspeed = top_soundspeed_buffer.access<R>(h);
        auto soundspeed = soundspeed_buffer.access<W>(h);
        clover::par_ranged<class upd_halo_t_soundspeed>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          soundspeed[j[0]][y_max + 2 + k] = top_soundspeed[j[0]][top_ymin - 1 + 2 + k];
        });
      });
    }
  }

  // XVEL 0
  if (fields[field_xvel0] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+1+depth
      clover::execute(globals.context.queue, [&](handler &h) {
        auto top_xvel0 = top_xvel0_buffer.access<R>(h);
        auto xvel0 = xvel0_buffer.access<W>(h);
        clover::par_ranged<class upd_halo_t_xvel0>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](id<1> j) {
          xvel0[j[0]][y_max + 1 + 2 + k] = top_xvel0[j[0]][top_ymin + 1 - 1 + 2 + k];
        });
      });
    }
  }

  // XVEL 1
  if (fields[field_xvel1] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+1+depth
      clover::execute(globals.context.queue, [&](handler &h) {
        auto top_xvel1 = top_xvel1_buffer.access<R>(h);
        auto xvel1 = xvel1_buffer.access<W>(h);
        clover::par_ranged<class upd_halo_t_xvel1>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](id<1> j) {
          xvel1[j[0]][y_max + 1 + 2 + k] = top_xvel1[j[0]][top_ymin + 1 - 1 + 2 + k];
        });
      });
    }
  }

  // YVEL 0
  if (fields[field_yvel0] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+1+depth
      clover::execute(globals.context.queue, [&](handler &h) {
        auto top_yvel0 = top_yvel0_buffer.access<R>(h);
        auto yvel0 = yvel0_buffer.access<W>(h);
        clover::par_ranged<class upd_halo_t_yvel0>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](id<1> j) {
          yvel0[j[0]][y_max + 1 + 2 + k] = top_yvel0[j[0]][top_ymin + 1 - 1 + 2 + k];
        });
      });
    }
  }

  // YVEL 1
  if (fields[field_yvel1] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+1+depth
      clover::execute(globals.context.queue, [&](handler &h) {
        auto top_yvel1 = top_yvel1_buffer.access<R>(h);
        auto yvel1 = yvel1_buffer.access<W>(h);
        clover::par_ranged<class upd_halo_t_yvel1>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](id<1> j) {
          yvel1[j[0]][y_max + 1 + 2 + k] = top_yvel1[j[0]][top_ymin + 1 - 1 + 2 + k];
        });
      });
    }
  }

  // VOL_FLUX_X
  if (fields[field_vol_flux_x] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+1+depth
      clover::execute(globals.context.queue, [&](handler &h) {
        auto top_vol_flux_x = top_vol_flux_x_buffer.access<R>(h);
        auto vol_flux_x = vol_flux_x_buffer.access<W>(h);
        clover::par_ranged<class upd_halo_t_vol_flux_x>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](id<1> j) {
          vol_flux_x[j[0]][y_max + 2 + k] = top_vol_flux_x[j[0]][top_ymin - 1 + 2 + k];
        });
      });
    }
  }

  // MASS_FLUX_X
  if (fields[field_mass_flux_x] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+1+depth
      clover::execute(globals.context.queue, [&](handler &h) {
        auto top_mass_flux_x = top_mass_flux_x_buffer.access<R>(h);
        auto mass_flux_x = mass_flux_x_buffer.access<W>(h);
        clover::par_ranged<class upd_halo_t_mass_flux_x>(h, {x_min - depth + 1, x_max + 1 + depth + 2}, [=](id<1> j) {
          mass_flux_x[j[0]][y_max + 2 + k] = top_mass_flux_x[j[0]][top_ymin - 1 + 2 + k];
        });
      });
    }
  }

  // VOL_FLUX_Y
  if (fields[field_vol_flux_y] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+depth
      clover::execute(globals.context.queue, [&](handler &h) {
        auto top_vol_flux_y = top_vol_flux_y_buffer.access<R>(h);
        auto vol_flux_y = vol_flux_y_buffer.access<W>(h);
        clover::par_ranged<class upd_halo_t_vol_flux_y>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          vol_flux_y[j[0]][y_max + 1 + 2 + k] = top_vol_flux_y[j[0]][top_ymin + 1 - 1 + 2 + k];
        });
      });
    }
  }

  // MASS_FLUX_Y
  if (fields[field_mass_flux_y] == 1) {
    for (int k = 0; k < depth; ++k) {
      // DO j=x_min-depth, x_max+depth
      clover::execute(globals.context.queue, [&](handler &h) {
        auto top_mass_flux_y = top_mass_flux_y_buffer.access<R>(h);
        auto mass_flux_y = mass_flux_y_buffer.access<W>(h);
        clover::par_ranged<class upd_halo_t_mass_flux_y>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          mass_flux_y[j[0]][y_max + 1 + 2 + k] = top_mass_flux_y[j[0]][top_ymin + 1 - 1 + 2 + k];
        });
      });
    }
  }
}

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

#include "comms.h"
#include "comms_kernel.h"
#include "context.h"
#include "timer.h"
#include "update_halo.h"
#include "update_tile_halo.h"

void update_halo_kernel_1(queue &queue, int x_min, int x_max, int y_min, int y_max, const std::array<int, 4> &chunk_neighbours,
                          const std::array<int, 4> &tile_neighbours, field_type &field, int fields[NUM_FIELDS], int depth) {

  //  Update values in external halo cells based on depth and fields requested
  //  Even though half of these loops look the wrong way around, it should be noted
  //  that depth is either 1 or 2 so that it is more efficient to always thread
  //  loop along the mesh edge.
  if (fields[field_density0] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+depth
      clover::execute(queue, [&](handler &h) {
        auto density0 = field.density0.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          for (int k = 0; k < depth; ++k) {
            density0[j[0]][1 - k] = density0[j[0]][2 + k];
          }
        });
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth
      clover::execute(queue, [&](handler &h) {
        auto density0 = field.density0.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          for (int k = 0; k < depth; ++k) {
            density0[j[0]][y_max + 2 + k] = density0[j[0]][y_max + 1 - k];
          }
        });
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+depth
      clover::execute(queue, [&](handler &h) {
        auto density0 = field.density0.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
          for (int j = 0; j < depth; ++j) {
            density0[1 - j][k[0]] = density0[2 + j][k[0]];
          }
        });
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+depth
      clover::execute(queue, [&](handler &h) {
        auto density0 = field.density0.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
          for (int j = 0; j < depth; ++j) {
            density0[x_max + 2 + j][k[0]] = density0[x_max + 1 - j][k[0]];
          }
        });
      });
    }
  }

  if (fields[field_density1] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO k=y_min-depth,y_max+depth
      clover::execute(queue, [&](handler &h) {
        auto density1 = field.density1.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          for (int k = 0; k < depth; ++k) {
            density1[j[0]][1 - k] = density1[j[0]][2 + k];
          }
        });
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth
      clover::execute(queue, [&](handler &h) {
        auto density1 = field.density1.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          for (int k = 0; k < depth; ++k) {
            density1[j[0]][y_max + 2 + k] = density1[j[0]][y_max + 1 - k];
          }
        });
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+depth
      clover::execute(queue, [&](handler &h) {
        auto density1 = field.density1.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
          for (int j = 0; j < depth; ++j) {
            density1[1 - j][k[0]] = density1[2 + j][k[0]];
          }
        });
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+depth
      clover::execute(queue, [&](handler &h) {
        auto density1 = field.density1.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
          for (int j = 0; j < depth; ++j) {
            density1[x_max + 2 + j][k[0]] = density1[x_max + 1 - j][k[0]];
          }
        });
      });
    }
  }

  if (fields[field_energy0] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      //  DO j=x_min-depth,x_max+depth
      clover::execute(queue, [&](handler &h) {
        auto energy0 = field.energy0.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          for (int k = 0; k < depth; ++k) {
            energy0[j[0]][1 - k] = energy0[j[0]][2 + k];
          }
        });
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth
      clover::execute(queue, [&](handler &h) {
        auto energy0 = field.energy0.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          for (int k = 0; k < depth; ++k) {
            energy0[j[0]][y_max + 2 + k] = energy0[j[0]][y_max + 1 - k];
          }
        });
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+depth
      clover::execute(queue, [&](handler &h) {
        auto energy0 = field.energy0.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
          for (int j = 0; j < depth; ++j) {
            energy0[1 - j][k[0]] = energy0[2 + j][k[0]];
          }
        });
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+depth
      clover::execute(queue, [&](handler &h) {
        auto energy0 = field.energy0.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
          for (int j = 0; j < depth; ++j) {
            energy0[x_max + 2 + j][k[0]] = energy0[x_max + 1 - j][k[0]];
          }
        });
      });
    }
  }

  if (fields[field_energy1] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+depth
      clover::execute(queue, [&](handler &h) {
        auto energy1 = field.energy1.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          for (int k = 0; k < depth; ++k) {
            energy1[j[0]][1 - k] = energy1[j[0]][2 + k];
          }
        });
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth
      clover::execute(queue, [&](handler &h) {
        auto energy1 = field.energy1.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          for (int k = 0; k < depth; ++k) {
            energy1[j[0]][y_max + 2 + k] = energy1[j[0]][y_max + 1 - k];
          }
        });
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+depth
      clover::execute(queue, [&](handler &h) {
        auto energy1 = field.energy1.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
          for (int j = 0; j < depth; ++j) {
            energy1[1 - j][k[0]] = energy1[2 + j][k[0]];
          }
        });
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+depth
      clover::execute(queue, [&](handler &h) {
        auto energy1 = field.energy1.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
          for (int j = 0; j < depth; ++j) {
            energy1[x_max + 2 + j][k[0]] = energy1[x_max + 1 - j][k[0]];
          }
        });
      });
    }
  }

  if (fields[field_pressure] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+depth
      clover::execute(queue, [&](handler &h) {
        auto pressure = field.pressure.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          for (int k = 0; k < depth; ++k) {
            pressure[j[0]][1 - k] = pressure[j[0]][2 + k];
          }
        });
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth
      clover::execute(queue, [&](handler &h) {
        auto pressure = field.pressure.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          for (int k = 0; k < depth; ++k) {
            pressure[j[0]][y_max + 2 + k] = pressure[j[0]][y_max + 1 - k];
          }
        });
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+depth
      clover::execute(queue, [&](handler &h) {
        auto pressure = field.pressure.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
          for (int j = 0; j < depth; ++j) {
            pressure[1 - j][k[0]] = pressure[2 + j][k[0]];
          }
        });
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+depth
      clover::execute(queue, [&](handler &h) {
        auto pressure = field.pressure.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
          for (int j = 0; j < depth; ++j) {
            pressure[x_max + 2 + j][k[0]] = pressure[x_max + 1 - j][k[0]];
          }
        });
      });
    }
  }

  if (fields[field_viscosity] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+depth
      clover::execute(queue, [&](handler &h) {
        auto viscosity = field.viscosity.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          for (int k = 0; k < depth; ++k) {
            viscosity[j[0]][1 - k] = viscosity[j[0]][2 + k];
          }
        });
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth
      clover::execute(queue, [&](handler &h) {
        auto viscosity = field.viscosity.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          for (int k = 0; k < depth; ++k) {
            viscosity[j[0]][y_max + 2 + k] = viscosity[j[0]][y_max + 1 - k];
          }
        });
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+depth
      clover::execute(queue, [&](handler &h) {
        auto viscosity = field.viscosity.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
          for (int j = 0; j < depth; ++j) {
            viscosity[1 - j][k[0]] = viscosity[2 + j][k[0]];
          }
        });
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+depth
      clover::execute(queue, [&](handler &h) {
        auto viscosity = field.viscosity.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
          for (int j = 0; j < depth; ++j) {
            viscosity[x_max + 2 + j][k[0]] = viscosity[x_max + 1 - j][k[0]];
          }
        });
      });
    }
  }

  if (fields[field_soundspeed] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+depth
      clover::execute(queue, [&](handler &h) {
        auto soundspeed = field.soundspeed.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          for (int k = 0; k < depth; ++k) {
            soundspeed[j[0]][1 - k] = soundspeed[j[0]][2 + k];
          }
        });
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth
      clover::execute(queue, [&](handler &h) {
        auto soundspeed = field.soundspeed.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {x_min - depth + 1, x_max + depth + 2}, [=](id<1> j) {
          for (int k = 0; k < depth; ++k) {
            soundspeed[j[0]][y_max + 2 + k] = soundspeed[j[0]][y_max + 1 - k];
          }
        });
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      //  DO k=y_min-depth,y_max+depth
      clover::execute(queue, [&](handler &h) {
        auto soundspeed = field.soundspeed.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
          for (int j = 0; j < depth; ++j) {
            soundspeed[1 - j][k[0]] = soundspeed[2 + j][k[0]];
          }
        });
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      //  DO k=y_min-depth,y_max+depth
      clover::execute(queue, [&](handler &h) {
        auto soundspeed = field.soundspeed.access<RW>(h);
        clover::par_ranged<class APPEND_LN(update_halo)>(h, {y_min - depth + 1, y_max + depth + 2}, [=](id<1> k) {
          for (int j = 0; j < depth; ++j) {
            soundspeed[x_max + 2 + j][k[0]] = soundspeed[x_max + 1 - j][k[0]];
          }
        });
      });
    }
  }
}

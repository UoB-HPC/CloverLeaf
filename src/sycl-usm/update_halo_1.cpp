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

void update_halo_kernel_1(sycl::queue &queue, int x_min, int x_max, int y_min, int y_max, const std::array<int, 4> &chunk_neighbours,
                          const std::array<int, 4> &tile_neighbours, field_type &field, const int fields[NUM_FIELDS], int depth) {

  //  Update values in external halo cells based on depth and fields requested
  //  Even though half of these loops look the wrong way around, it should be noted
  //  that depth is either 1 or 2 so that it is more efficient to always thread
  //  loop along the mesh edge.
  if (fields[field_density0] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.density0(j, 1 - k) = field.density0(j, 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.density0(j, y_max + 2 + k) = field.density0(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.density0(1 - j, k) = field.density0(2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.density0(x_max + 2 + j, k) = field.density0(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_density1] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.density1(j, 1 - k) = field.density1(j, 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.density1(j, y_max + 2 + k) = field.density1(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.density1(1 - j, k) = field.density1(2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.density1(x_max + 2 + j, k) = field.density1(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_energy0] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      //  DO j=x_min-depth,x_max+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.energy0(j, 1 - k) = field.energy0(j, 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.energy0(j, y_max + 2 + k) = field.energy0(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.energy0(1 - j, k) = field.energy0(2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.energy0(x_max + 2 + j, k) = field.energy0(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_energy1] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.energy1(j, 1 - k) = field.energy1(j, 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.energy1(j, y_max + 2 + k) = field.energy1(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.energy1(1 - j, k) = field.energy1(2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.energy1(x_max + 2 + j, k) = field.energy1(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_pressure] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + depth + 2}, [=](int j) { // FIXME
        for (int k = 0; k < depth; ++k) {
          field.pressure(j, 1 - k) = field.pressure(j, 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + depth + 2}, [=](int j) { // FIXME
        for (int k = 0; k < depth; ++k) {
          field.pressure(j, y_max + 2 + k) = field.pressure(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + depth + 2}, [=](int k) { // FIXME
        for (int j = 0; j < depth; ++j) {
          field.pressure(1 - j, k) = field.pressure(2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + depth + 2}, [=](int k) { // FIXME
        for (int j = 0; j < depth; ++j) {
          field.pressure(x_max + 2 + j, k) = field.pressure(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_viscosity] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + depth + 2}, [=](int j) { // FIXME par bad
        for (int k = 0; k < depth; ++k) {
          field.viscosity(j, 1 - k) = field.viscosity(j, 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + depth + 2}, [=](int j) { // FIXME par bad
        for (int k = 0; k < depth; ++k) {
          field.viscosity(j, y_max + 2 + k) = field.viscosity(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + depth + 2}, [=](int k) { // FIXME par bad
        for (int j = 0; j < depth; ++j) {
          field.viscosity(1 - j, k) = field.viscosity(2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + depth + 2}, [=](int k) { // FIXME par bad
        for (int j = 0; j < depth; ++j) {
          field.viscosity(x_max + 2 + j, k) = field.viscosity(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_soundspeed] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.soundspeed(j, 1 - k) = field.soundspeed(j, 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.soundspeed(j, y_max + 2 + k) = field.soundspeed(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      //  DO k=y_min-depth,y_max+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.soundspeed(1 - j, k) = field.soundspeed(2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      //  DO k=y_min-depth,y_max+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.soundspeed(x_max + 2 + j, k) = field.soundspeed(x_max + 1 - j, k);
        }
      });
    }
  }
}

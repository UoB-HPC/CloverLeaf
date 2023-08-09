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

void update_halo_kernel_2(sycl::queue &queue, int x_min, int x_max, int y_min, int y_max, const std::array<int, 4> &chunk_neighbours,
                          const std::array<int, 4> &tile_neighbours, field_type &field, const int fields[NUM_FIELDS], int depth) {

  if (fields[field_xvel0] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.xvel0(j, 1 - k) = field.xvel0(j, 1 + 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.xvel0(j, y_max + 1 + 2 + k) = field.xvel0(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.xvel0(1 - j, k) = -field.xvel0(1 + 2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.xvel0(x_max + 2 + 1 + j, k) = -field.xvel0(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_xvel1] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.xvel1(j, 1 - k) = field.xvel1(j, 1 + 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.xvel1(j, y_max + 1 + 2 + k) = field.xvel1(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.xvel1(1 - j, k) = -field.xvel1(1 + 2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.xvel1(x_max + 2 + 1 + j, k) = -field.xvel1(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_yvel0] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.yvel0(j, 1 - k) = -field.yvel0(j, 1 + 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.yvel0(j, y_max + 1 + 2 + k) = -field.yvel0(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.yvel0(1 - j, k) = field.yvel0(1 + 2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.yvel0(x_max + 2 + 1 + j, k) = field.yvel0(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_yvel1] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.yvel1(j, 1 - k) = -field.yvel1(j, 1 + 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.yvel1(j, y_max + 1 + 2 + k) = -field.yvel1(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.yvel1(1 - j, k) = field.yvel1(1 + 2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.yvel1(x_max + 2 + 1 + j, k) = field.yvel1(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_vol_flux_x] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.vol_flux_x(j, 1 - k) = field.vol_flux_x(j, 1 + 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.vol_flux_x(j, y_max + 2 + k) = field.vol_flux_x(j, y_max - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.vol_flux_x(1 - j, k) = -field.vol_flux_x(1 + 2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.vol_flux_x(x_max + j + 1 + 2, k) = -field.vol_flux_x(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_mass_flux_x] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.mass_flux_x(j, 1 - k) = field.mass_flux_x(j, 1 + 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.mass_flux_x(j, y_max + 2 + k) = field.mass_flux_x(j, y_max - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.mass_flux_x(1 - j, k) = -field.mass_flux_x(1 + 2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.mass_flux_x(x_max + j + 1 + 2, k) = -field.mass_flux_x(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_vol_flux_y] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.vol_flux_y(j, 1 - k) = -field.vol_flux_y(j, 1 + 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.vol_flux_y(j, y_max + k + 1 + 2) = -field.vol_flux_y(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.vol_flux_y(1 - j, k) = field.vol_flux_y(1 + 2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.vol_flux_y(x_max + 2 + j, k) = field.vol_flux_y(x_max - j, k);
        }
      });
    }
  }

  if (fields[field_mass_flux_y] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.mass_flux_y(j, 1 - k) = -field.mass_flux_y(j, 1 + 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

      clover::par_ranged1(queue, Range1d{x_min - depth + 1, x_max + depth + 2}, [=](int j) {
        for (int k = 0; k < depth; ++k) {
          field.mass_flux_y(j, y_max + k + 1 + 2) = -field.mass_flux_y(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.mass_flux_y(1 - j, k) = field.mass_flux_y(1 + 2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

      clover::par_ranged1(queue, Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=](int k) {
        for (int j = 0; j < depth; ++j) {
          field.mass_flux_y(x_max + 2 + j, k) = field.mass_flux_y(x_max - j, k);
        }
      });
    }
  }
}

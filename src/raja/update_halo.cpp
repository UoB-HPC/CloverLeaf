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

#include "update_halo.h"
#include "comms.h"
#include "comms_kernel.h"
#include "context.h"
#include "timer.h"
#include "update_tile_halo.h"

#include <RAJA/RAJA.hpp>

//   @brief Fortran kernel to update the external halo cells in a chunk.
//   @author Wayne Gaudin
//   @details Updates halo cells for the required fields at the required depth
//   for any halo cells that lie on an external boundary. The location and type
//   of data governs how this is carried out. External boundaries are always
//   reflective.
void update_halo_kernel(int x_min, int x_max, int y_min, int y_max, const std::array<int, 4> &chunk_neighbours,
                        const std::array<int, 4> &tile_neighbours, field_type &field, const int fields[NUM_FIELDS], int depth) {

  //  Update values in external halo cells based on depth and fields requested
  //  Even though half of these loops look the wrong way around, it should be noted
  //  that depth is either 1 or 2 so that it is more efficient to always thread
  //  loop along the mesh edge.
  if (fields[field_density0] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

      RAJA::forall<raja_default_policy>(
        RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + depth + 2), [=] RAJA_HOST_DEVICE (int j ) {
        for (int k = 0; k < depth; ++k) {
          field.density0(j, 1 - k) = field.density0(j, 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + depth + 2}, [=] DEVICE_KERNEL(int j) {
      RAJA::forall<raja_default_policy>(
        RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + depth + 2), [=] RAJA_HOST_DEVICE (int j ) {
        for (int k = 0; k < depth; ++k) {
          field.density0(j, y_max + 2 + k) = field.density0(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

      //clover::par_ranged1(Range1d{y_min - depth + 1, y_max + depth + 2}, [=] DEVICE_KERNEL(int k) {
      RAJA::forall<raja_default_policy>(
        RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.density0(1 - j, k) = field.density0(2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

      RAJA::forall<raja_default_policy>(
        RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.density0(x_max + 2 + j, k) = field.density0(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_density1] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + depth + 2}, [=] DEVICE_KERNEL(int j) {
      RAJA::forall<raja_default_policy>(
        RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + depth + 2), [=] RAJA_HOST_DEVICE (int j ) {
        for (int k = 0; k < depth; ++k) {
          field.density1(j, 1 - k) = field.density1(j, 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + depth + 2}, [=] DEVICE_KERNEL(int j) {
      RAJA::forall<raja_default_policy>(
        RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + depth + 2), [=] RAJA_HOST_DEVICE (int j ) {
        for (int k = 0; k < depth; ++k) {
          field.density1(j, y_max + 2 + k) = field.density1(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + depth + 2}, [=] DEVICE_KERNEL(int k) {
        RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, y_max + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.density1(1 - j, k) = field.density1(2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + depth + 2}, [=] DEVICE_KERNEL(int k) {
       RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.density1(x_max + 2 + j, k) = field.density1(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_energy0] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      //  DO j=x_min-depth,x_max+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + depth + 2}, [=] DEVICE_KERNEL(int j) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.energy0(j, 1 - k) = field.energy0(j, 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + depth + 2}, [=] DEVICE_KERNEL(int j) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.energy0(j, y_max + 2 + k) = field.energy0(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + depth + 2}, [=] DEVICE_KERNEL(int k) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.energy0(1 - j, k) = field.energy0(2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + depth + 2}, [=] DEVICE_KERNEL(int k) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.energy0(x_max + 2 + j, k) = field.energy0(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_energy1] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + depth + 2}, [=] DEVICE_KERNEL(int j) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.energy1(j, 1 - k) = field.energy1(j, 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + depth + 2}, [=] DEVICE_KERNEL(int j) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.energy1(j, y_max + 2 + k) = field.energy1(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + depth + 2}, [=] DEVICE_KERNEL(int k) {
     RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.energy1(1 - j, k) = field.energy1(2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + depth + 2}, [=] DEVICE_KERNEL(int k) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.energy1(x_max + 2 + j, k) = field.energy1(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_pressure] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + depth + 2}, [=] DEVICE_KERNEL(int j) { // FIXME
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.pressure(j, 1 - k) = field.pressure(j, 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

      //clover::par_ranged1(Range1d{x_min - depth + 1, x_max + depth + 2}, [=] DEVICE_KERNEL(int j) { // FIXME
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.pressure(j, y_max + 2 + k) = field.pressure(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + depth + 2}, [=] DEVICE_KERNEL(int k) { // FIXME
       RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.pressure(1 - j, k) = field.pressure(2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+depth
      //clover::par_ranged1(Range1d{y_min - depth + 1, y_max + depth + 2}, [=] DEVICE_KERNEL(int k) { // FIXME
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.pressure(x_max + 2 + j, k) = field.pressure(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_viscosity] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + depth + 2}, [=] DEVICE_KERNEL(int j) { // FIXME par bad
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.viscosity(j, 1 - k) = field.viscosity(j, 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + depth + 2}, [=] DEVICE_KERNEL(int j) { // FIXME par bad
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.viscosity(j, y_max + 2 + k) = field.viscosity(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

      //clover::par_ranged1(Range1d{y_min - depth + 1, y_max + depth + 2}, [=] DEVICE_KERNEL(int k) { // FIXME par bad
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.viscosity(1 - j, k) = field.viscosity(2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

      //clover::par_ranged1(Range1d{y_min - depth + 1, y_max + depth + 2}, [=] DEVICE_KERNEL(int k) { // FIXME par bad
       RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.viscosity(x_max + 2 + j, k) = field.viscosity(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_soundspeed] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+depth
//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + depth + 2}, [=] DEVICE_KERNEL(int j) {
       RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.soundspeed(j, 1 - k) = field.soundspeed(j, 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

      //clover::par_ranged1(Range1d{x_min - depth + 1, x_max + depth + 2}, [=] DEVICE_KERNEL(int j) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.soundspeed(j, y_max + 2 + k) = field.soundspeed(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      //  DO k=y_min-depth,y_max+depth

      //clover::par_ranged1(Range1d{y_min - depth + 1, y_max + depth + 2}, [=] DEVICE_KERNEL(int k) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.soundspeed(1 - j, k) = field.soundspeed(2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      //  DO k=y_min-depth,y_max+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + depth + 2}, [=] DEVICE_KERNEL(int k) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.soundspeed(x_max + 2 + j, k) = field.soundspeed(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_xvel0] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int j) {
     RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.xvel0(j, 1 - k) = field.xvel0(j, 1 + 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int j) {
     RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.xvel0(j, y_max + 1 + 2 + k) = field.xvel0(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int k) {
     RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.xvel0(1 - j, k) = -field.xvel0(1 + 2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int k) {
        RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.xvel0(x_max + 2 + 1 + j, k) = -field.xvel0(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_xvel1] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int j) {
        RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.xvel1(j, 1 - k) = field.xvel1(j, 1 + 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int j) {
        RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.xvel1(j, y_max + 1 + 2 + k) = field.xvel1(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int k) {
        RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.xvel1(1 - j, k) = -field.xvel1(1 + 2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

      //clover::par_ranged1(Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int k) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.xvel1(x_max + 2 + 1 + j, k) = -field.xvel1(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_yvel0] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int j) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.yvel0(j, 1 - k) = -field.yvel0(j, 1 + 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth
//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int j) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.yvel0(j, y_max + 1 + 2 + k) = -field.yvel0(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int k) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.yvel0(1 - j, k) = field.yvel0(1 + 2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int k) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.yvel0(x_max + 2 + 1 + j, k) = field.yvel0(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_yvel1] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int j) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.yvel1(j, 1 - k) = -field.yvel1(j, 1 + 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int j) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.yvel1(j, y_max + 1 + 2 + k) = -field.yvel1(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int k) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.yvel1(1 - j, k) = field.yvel1(1 + 2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int k) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.yvel1(x_max + 2 + 1 + j, k) = field.yvel1(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_vol_flux_x] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int j) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.vol_flux_x(j, 1 - k) = field.vol_flux_x(j, 1 + 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int j) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.vol_flux_x(j, y_max + 2 + k) = field.vol_flux_x(j, y_max - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + depth + 2}, [=] DEVICE_KERNEL(int k) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.vol_flux_x(1 - j, k) = -field.vol_flux_x(1 + 2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + depth + 2}, [=] DEVICE_KERNEL(int k) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.vol_flux_x(x_max + j + 1 + 2, k) = -field.vol_flux_x(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_mass_flux_x] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int j) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.mass_flux_x(j, 1 - k) = field.mass_flux_x(j, 1 + 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+1+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int j) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.mass_flux_x(j, y_max + 2 + k) = field.mass_flux_x(j, y_max - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + depth + 2}, [=] DEVICE_KERNEL(int k) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.mass_flux_x(1 - j, k) = -field.mass_flux_x(1 + 2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + depth + 2}, [=] DEVICE_KERNEL(int k) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.mass_flux_x(x_max + j + 1 + 2, k) = -field.mass_flux_x(x_max + 1 - j, k);
        }
      });
    }
  }

  if (fields[field_vol_flux_y] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + depth + 2}, [=] DEVICE_KERNEL(int j) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.vol_flux_y(j, 1 - k) = -field.vol_flux_y(j, 1 + 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + depth + 2}, [=] DEVICE_KERNEL(int j) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.vol_flux_y(j, y_max + k + 1 + 2) = -field.vol_flux_y(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int k) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.vol_flux_y(1 - j, k) = field.vol_flux_y(1 + 2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int k) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.vol_flux_y(x_max + 2 + j, k) = field.vol_flux_y(x_max - j, k);
        }
      });
    }
  }

  if (fields[field_mass_flux_y] == 1) {
    if ((chunk_neighbours[chunk_bottom] == external_face) && (tile_neighbours[tile_bottom] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + depth + 2}, [=] DEVICE_KERNEL(int j) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.mass_flux_y(j, 1 - k) = -field.mass_flux_y(j, 1 + 2 + k);
        }
      });
    }
    if ((chunk_neighbours[chunk_top] == external_face) && (tile_neighbours[tile_top] == external_tile)) {
      // DO j=x_min-depth,x_max+depth

//      clover::par_ranged1(Range1d{x_min - depth + 1, x_max + depth + 2}, [=] DEVICE_KERNEL(int j) {
      RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(x_min - depth + 1, x_max + depth + 2), [=] RAJA_HOST_DEVICE (int j) {
        for (int k = 0; k < depth; ++k) {
          field.mass_flux_y(j, y_max + k + 1 + 2) = -field.mass_flux_y(j, y_max + 1 - k);
        }
      });
    }
    if ((chunk_neighbours[chunk_left] == external_face) && (tile_neighbours[tile_left] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int k) {
     RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.mass_flux_y(1 - j, k) = field.mass_flux_y(1 + 2 + j, k);
        }
      });
    }
    if ((chunk_neighbours[chunk_right] == external_face) && (tile_neighbours[tile_right] == external_tile)) {
      // DO k=y_min-depth,y_max+1+depth

//      clover::par_ranged1(Range1d{y_min - depth + 1, y_max + 1 + depth + 2}, [=] DEVICE_KERNEL(int k) {
     RAJA::forall<raja_default_policy>(
         RAJA::TypedRangeSegment<int>(y_min - depth + 1, y_max + 1 + depth + 2), [=] RAJA_HOST_DEVICE (int k) {
        for (int j = 0; j < depth; ++j) {
          field.mass_flux_y(x_max + 2 + j, k) = field.mass_flux_y(x_max - j, k);
        }
      });
    }
  }
}

//  @brief Driver for the halo updates
//  @author Wayne Gaudin
//  @details Invokes the kernels for the internal and external halo cells for
//  the fields specified.
void update_halo(global_variables &globals, int fields[NUM_FIELDS], int depth) {

  double kernel_time = 0;
  if (globals.profiler_on) kernel_time = timer();
  update_tile_halo(globals, fields, depth);
  if (globals.profiler_on) {
    clover::checkError(rajaDeviceSynchronize());
    globals.profiler.tile_halo_exchange += timer() - kernel_time;
    kernel_time = timer();
  }

  clover_exchange(globals, fields, depth);

  if (globals.profiler_on) {
    clover::checkError(rajaDeviceSynchronize());
    globals.profiler.mpi_halo_exchange += timer() - kernel_time;
    kernel_time = timer();
  }

  if ((globals.chunk.chunk_neighbours[chunk_left] == external_face) || (globals.chunk.chunk_neighbours[chunk_right] == external_face) ||
      (globals.chunk.chunk_neighbours[chunk_bottom] == external_face) || (globals.chunk.chunk_neighbours[chunk_top] == external_face)) {

    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
      tile_type &t = globals.chunk.tiles[tile];
      update_halo_kernel(t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, globals.chunk.chunk_neighbours, t.info.tile_neighbours,
                         t.field, fields, depth);
    }
  }

  if (globals.profiler_on) {
    clover::checkError(rajaDeviceSynchronize());
    globals.profiler.self_halo_exchange += timer() - kernel_time;
  }
}

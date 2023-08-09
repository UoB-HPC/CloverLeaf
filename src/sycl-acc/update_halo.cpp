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

void update_halo_kernel_1(queue &queue, int x_min, int x_max, int y_min, int y_max, const std::array<int, 4> &chunk_neighbours,
                          const std::array<int, 4> &tile_neighbours, field_type &field, int fields[NUM_FIELDS], int depth);

void update_halo_kernel_2(queue &queue, int x_min, int x_max, int y_min, int y_max, const std::array<int, 4> &chunk_neighbours,
                          const std::array<int, 4> &tile_neighbours, field_type &field, int fields[NUM_FIELDS], int depth);

//  @brief Driver for the halo updates
//  @author Wayne Gaudin
//  @details Invokes the kernels for the internal and external halo cells for
//  the fields specified.
void update_halo(global_variables &globals, int fields[NUM_FIELDS], int depth) {
  double kernel_time = 0;
  if (globals.profiler_on) kernel_time = timer();
  update_tile_halo(globals, fields, depth);
  if (globals.profiler_on) {
    globals.profiler.tile_halo_exchange += timer() - kernel_time;
    kernel_time = timer();
  }

  clover_exchange(globals, fields, depth);

  if (globals.profiler_on) {
    globals.profiler.mpi_halo_exchange += timer() - kernel_time;
    kernel_time = timer();
  }

  if ((globals.chunk.chunk_neighbours[chunk_left] == external_face) || (globals.chunk.chunk_neighbours[chunk_right] == external_face) ||
      (globals.chunk.chunk_neighbours[chunk_bottom] == external_face) || (globals.chunk.chunk_neighbours[chunk_top] == external_face)) {

    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
      tile_type &t = globals.chunk.tiles[tile];
      update_halo_kernel_1(globals.context.queue, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax,
                           globals.chunk.chunk_neighbours, t.info.tile_neighbours, t.field, fields, depth);
      update_halo_kernel_2(globals.context.queue, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax,
                           globals.chunk.chunk_neighbours, t.info.tile_neighbours, t.field, fields, depth);
    }
  }

  if (globals.profiler_on) globals.profiler.self_halo_exchange += timer() - kernel_time;
}

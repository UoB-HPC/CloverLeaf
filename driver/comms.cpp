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

//  @brief Communication Utilities
//  @author Wayne Gaudin
//  @details Contains all utilities required to run CloverLeaf in a distributed
//  environment, including initialisation, mesh decompostion, reductions and
//  halo exchange using explicit buffers.
//
//  Note the halo exchange is currently coded as simply as possible and no
//  optimisations have been implemented, such as post receives before sends or packing
//  buffers with multiple data fields. This is intentional so the effect of these
//  optimisations can be measured on large systems, as and when they are added.
//
//  Even without these modifications CloverLeaf weak scales well on moderately sized
//  systems of the order of 10K cores.

#include "comms.h"
#include "context.h"
#include "pack_kernel.h"

#include <cstdlib>

extern std::ostream g_out;

// Set up parallel structure
parallel_::parallel_() {

  parallel = true;
  MPI_Comm_rank(MPI_COMM_WORLD, &task);
  MPI_Comm_size(MPI_COMM_WORLD, &max_task);

  boss = task == 0;
}

void clover_abort() { MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); }

void clover_barrier() { MPI_Barrier(MPI_COMM_WORLD); }

void clover_barrier(global_variables &globals) {
  clover_barrier();
}

//  This decomposes the mesh into a number of chunks.
//  The number of chunks may be a multiple of the number of mpi tasks
//  Doesn't always return the best split if there are few factors
//  All factors need to be stored and the best picked. But its ok for now
std::array<int, 4> clover_decompose(const global_config &globals, parallel_ &parallel, int x_cells, int y_cells, int &left, int &right,
                                    int &bottom, int &top) {

  std::array<int, 4> chunk_neighbours{};

  int number_of_chunks = globals.number_of_chunks;

  // 2D Decomposition of the mesh

  double mesh_ratio = (double)x_cells / (double)y_cells;

  int chunk_x = number_of_chunks;
  int chunk_y = 1;

  int split_found = 0; // Used to detect 1D decomposition

  double factor_x, factor_y;

  for (int c = 1; c <= number_of_chunks; ++c) {
    if (number_of_chunks % c == 0) {
      factor_x = number_of_chunks / (double)c;
      factor_y = c;
      // Compare the factor ratio with the mesh ratio
      if (factor_x / factor_y <= mesh_ratio) {
        chunk_y = c;
        chunk_x = number_of_chunks / c;
        split_found = 1;
        break;
      }
    }
  }

  if (split_found == 0 || chunk_y == number_of_chunks) { // Prime number or 1D decomp detected
    if (mesh_ratio >= 1.0) {
      chunk_x = number_of_chunks;
      chunk_y = 1;
    } else {
      chunk_x = 1;
      chunk_y = number_of_chunks;
    }
  }

  int delta_x = x_cells / chunk_x;
  int delta_y = y_cells / chunk_y;
  int mod_x = x_cells % chunk_x;
  int mod_y = y_cells % chunk_y;

  // Set up chunk mesh ranges and chunk connectivity

  int add_x_prev = 0;
  int add_y_prev = 0;
  int cnk = 1;
  for (int cy = 1; cy <= chunk_y; ++cy) {
    for (int cx = 1; cx <= chunk_x; ++cx) {
      int add_x = 0;
      int add_y = 0;
      if (cx <= mod_x) add_x = 1;
      if (cy <= mod_y) add_y = 1;

      if (cnk == parallel.task + 1) {
        left = (cx - 1) * delta_x + 1 + add_x_prev;
        right = left + delta_x - 1 + add_x;
        bottom = (cy - 1) * delta_y + 1 + add_y_prev;
        top = bottom + delta_y - 1 + add_y;

        chunk_neighbours[chunk_left] = chunk_x * (cy - 1) + cx - 1;
        chunk_neighbours[chunk_right] = chunk_x * (cy - 1) + cx + 1;
        chunk_neighbours[chunk_bottom] = chunk_x * (cy - 2) + cx;
        chunk_neighbours[chunk_top] = chunk_x * (cy) + cx;

        if (cx == 1) chunk_neighbours[chunk_left] = external_face;
        if (cx == chunk_x) chunk_neighbours[chunk_right] = external_face;
        if (cy == 1) chunk_neighbours[chunk_bottom] = external_face;
        if (cy == chunk_y) chunk_neighbours[chunk_top] = external_face;
      }

      if (cx <= mod_x) add_x_prev = add_x_prev + 1;

      cnk = cnk + 1;
    }
    add_x_prev = 0;
    if (cy <= mod_y) add_y_prev = add_y_prev + 1;
  }

  if (parallel.boss) {
    g_out << std::endl
          << "Mesh ratio of " << mesh_ratio << std::endl
          << "Decomposing the mesh into " << chunk_x << " by " << chunk_y << " chunks" << std::endl
          << "Decomposing the chunk with " << globals.tiles_per_chunk << " tiles" << std::endl
          << std::endl;
  }
  return chunk_neighbours;
}

std::vector<tile_info> clover_tile_decompose(global_variables &globals, int chunk_x_cells, int chunk_y_cells) {

  std::vector<tile_info> tiles(globals.config.tiles_per_chunk);

  int chunk_mesh_ratio = (double)chunk_x_cells / (double)chunk_y_cells;

  int tile_x = globals.config.tiles_per_chunk;
  int tile_y = 1;

  int split_found = 0; // Used to detect 1D decomposition
  for (int t = 1; t <= globals.config.tiles_per_chunk; ++t) {
    if (globals.config.tiles_per_chunk % t == 0) {
      int factor_x = globals.config.tiles_per_chunk / (double)t;
      int factor_y = t;
      // Compare the factor ratio with the mesh ratio
      if (factor_x / factor_y <= chunk_mesh_ratio) {
        tile_y = t;
        tile_x = globals.config.tiles_per_chunk / t;
        split_found = 1;
        break;
      }
    }
  }

  if (split_found == 0 || tile_y == globals.config.tiles_per_chunk) { // Prime number or 1D decomp detected
    if (chunk_mesh_ratio >= 1.0) {
      tile_x = globals.config.tiles_per_chunk;
      tile_y = 1;
    } else {
      tile_x = 1;
      tile_y = globals.config.tiles_per_chunk;
    }
  }

  int chunk_delta_x = chunk_x_cells / tile_x;
  int chunk_delta_y = chunk_y_cells / tile_y;
  int chunk_mod_x = chunk_x_cells % tile_x;
  int chunk_mod_y = chunk_y_cells % tile_y;

  int add_x_prev = 0;
  int add_y_prev = 0;
  int tile = 0; // Used to index globals.chunk.tiles array
  for (int ty = 1; ty <= tile_y; ++ty) {
    for (int tx = 1; tx <= tile_x; ++tx) {
      int add_x = 0;
      int add_y = 0;
      if (tx <= chunk_mod_x) add_x = 1;
      if (ty <= chunk_mod_y) add_y = 1;

      int left = globals.chunk.left + (tx - 1) * chunk_delta_x + add_x_prev;
      int right = left + chunk_delta_x - 1 + add_x;
      int bottom = globals.chunk.bottom + (ty - 1) * chunk_delta_y + add_y_prev;
      int top = bottom + chunk_delta_y - 1 + add_y;

      tiles[tile].tile_neighbours[tile_left] = tile_x * (ty - 1) + tx - 1;
      tiles[tile].tile_neighbours[tile_right] = tile_x * (ty - 1) + tx + 1;
      tiles[tile].tile_neighbours[tile_bottom] = tile_x * (ty - 2) + tx;
      tiles[tile].tile_neighbours[tile_top] = tile_x * (ty) + tx;

      // initial set the external tile mask to 0 for each tile
      for (int i = 0; i < 4; ++i) {
        tiles[tile].external_tile_mask[i] = 0;
      }

      if (tx == 1) {
        tiles[tile].tile_neighbours[tile_left] = external_tile;
        tiles[tile].external_tile_mask[tile_left] = 1;
      }
      if (tx == tile_x) {
        tiles[tile].tile_neighbours[tile_right] = external_tile;
        tiles[tile].external_tile_mask[tile_right] = 1;
      }
      if (ty == 1) {
        tiles[tile].tile_neighbours[tile_bottom] = external_tile;
        tiles[tile].external_tile_mask[tile_bottom] = 1;
      }
      if (ty == tile_y) {
        tiles[tile].tile_neighbours[tile_top] = external_tile;
        tiles[tile].external_tile_mask[tile_top] = 1;
      }

      if (tx <= chunk_mod_x) add_x_prev = add_x_prev + 1;

      tiles[tile].t_xmin = 1;
      tiles[tile].t_xmax = right - left + 1;
      tiles[tile].t_ymin = 1;
      tiles[tile].t_ymax = top - bottom + 1;

      tiles[tile].t_left = left;
      tiles[tile].t_right = right;
      tiles[tile].t_top = top;
      tiles[tile].t_bottom = bottom;

      tile = tile + 1;
    }
    add_x_prev = 0;
    if (ty <= chunk_mod_y) add_y_prev = add_y_prev + 1;
  }
  return tiles;
}

void clover_sum(double &value) {
  double total = value;
  MPI_Reduce(&value, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  value = total;
}

void clover_min(double &value) {
  double minimum = value;
  MPI_Allreduce(&value, &minimum, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  value = minimum;
}

void clover_allgather(double value, std::vector<double> &values) {
  values[0] = value; // Just to ensure it will work in serial
  MPI_Allgather(&value, 1, MPI_DOUBLE, values.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
}

void clover_check_error(int &error) {
  int maximum = error;
  MPI_Allreduce(&error, &maximum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  error = maximum;
}

void clover_pack_left(global_variables &globals, clover::Buffer1D<double> &left_snd_buffer, int tile, const int fields[NUM_FIELDS],
                      int depth, int left_right_offset[NUM_FIELDS]) {

  tile_type &t = globals.chunk.tiles[tile];
  int t_offset = (t.info.t_bottom - globals.chunk.bottom) * depth;

  if (fields[field_density0] == 1) {
    clover_pack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.density0, left_snd_buffer,
                             cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                             left_right_offset[field_density0] + t_offset);
  }
  if (fields[field_density1] == 1) {
    clover_pack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.density1, left_snd_buffer,
                             cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                             left_right_offset[field_density1] + t_offset);
  }
  if (fields[field_energy0] == 1) {
    clover_pack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.energy0, left_snd_buffer,
                             cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                             left_right_offset[field_energy0] + t_offset);
  }
  if (fields[field_energy1] == 1) {
    clover_pack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.energy1, left_snd_buffer,
                             cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                             left_right_offset[field_energy1] + t_offset);
  }
  if (fields[field_pressure] == 1) {
    clover_pack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.pressure, left_snd_buffer,
                             cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                             left_right_offset[field_pressure] + t_offset);
  }
  if (fields[field_viscosity] == 1) {
    clover_pack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.viscosity, left_snd_buffer,
                             cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                             left_right_offset[field_viscosity] + t_offset);
  }
  if (fields[field_soundspeed] == 1) {
    clover_pack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.soundspeed, left_snd_buffer,
                             cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                             left_right_offset[field_soundspeed] + t_offset);
  }
  if (fields[field_xvel0] == 1) {
    clover_pack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.xvel0, left_snd_buffer, cell_data,
                             vertex_data, x_face_data, y_face_data, depth, vertex_data, left_right_offset[field_xvel0] + t_offset);
  }
  if (fields[field_xvel1] == 1) {
    clover_pack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.xvel1, left_snd_buffer, cell_data,
                             vertex_data, x_face_data, y_face_data, depth, vertex_data, left_right_offset[field_xvel1] + t_offset);
  }
  if (fields[field_yvel0] == 1) {
    clover_pack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.yvel0, left_snd_buffer, cell_data,
                             vertex_data, x_face_data, y_face_data, depth, vertex_data, left_right_offset[field_yvel0] + t_offset);
  }
  if (fields[field_yvel1] == 1) {
    clover_pack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.yvel1, left_snd_buffer, cell_data,
                             vertex_data, x_face_data, y_face_data, depth, vertex_data, left_right_offset[field_yvel1] + t_offset);
  }
  if (fields[field_vol_flux_x] == 1) {
    clover_pack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.vol_flux_x, left_snd_buffer,
                             cell_data, vertex_data, x_face_data, y_face_data, depth, x_face_data,
                             left_right_offset[field_vol_flux_x] + t_offset);
  }
  if (fields[field_vol_flux_y] == 1) {
    clover_pack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.vol_flux_y, left_snd_buffer,
                             cell_data, vertex_data, x_face_data, y_face_data, depth, y_face_data,
                             left_right_offset[field_vol_flux_y] + t_offset);
  }
  if (fields[field_mass_flux_x] == 1) {
    clover_pack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.mass_flux_x, left_snd_buffer,
                             cell_data, vertex_data, x_face_data, y_face_data, depth, x_face_data,
                             left_right_offset[field_mass_flux_x] + t_offset);
  }
  if (fields[field_mass_flux_y] == 1) {
    clover_pack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.mass_flux_y, left_snd_buffer,
                             cell_data, vertex_data, x_face_data, y_face_data, depth, y_face_data,
                             left_right_offset[field_mass_flux_y] + t_offset);
  }
}

void clover_unpack_left(global_variables &globals, clover::Buffer1D<double> &left_rcv_buffer, const int fields[NUM_FIELDS], int tile,
                        int depth, int left_right_offset[NUM_FIELDS]) {

  tile_type &t = globals.chunk.tiles[tile];
  int t_offset = (t.info.t_bottom - globals.chunk.bottom) * depth;

  if (fields[field_density0] == 1) {
    clover_unpack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.density0, left_rcv_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                               left_right_offset[field_density0] + t_offset);
  }
  if (fields[field_density1] == 1) {
    clover_unpack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.density1, left_rcv_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                               left_right_offset[field_density1] + t_offset);
  }
  if (fields[field_energy0] == 1) {
    clover_unpack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.energy0, left_rcv_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                               left_right_offset[field_energy0] + t_offset);
  }
  if (fields[field_energy1] == 1) {
    clover_unpack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.energy1, left_rcv_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                               left_right_offset[field_energy1] + t_offset);
  }
  if (fields[field_pressure] == 1) {
    clover_unpack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.pressure, left_rcv_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                               left_right_offset[field_pressure] + t_offset);
  }
  if (fields[field_viscosity] == 1) {
    clover_unpack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.viscosity, left_rcv_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                               left_right_offset[field_viscosity] + t_offset);
  }
  if (fields[field_soundspeed] == 1) {
    clover_unpack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.soundspeed, left_rcv_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                               left_right_offset[field_soundspeed] + t_offset);
  }
  if (fields[field_xvel0] == 1) {
    clover_unpack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.xvel0, left_rcv_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, vertex_data,
                               left_right_offset[field_xvel0] + t_offset);
  }
  if (fields[field_xvel1] == 1) {
    clover_unpack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.xvel1, left_rcv_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, vertex_data,
                               left_right_offset[field_xvel1] + t_offset);
  }
  if (fields[field_yvel0] == 1) {
    clover_unpack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.yvel0, left_rcv_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, vertex_data,
                               left_right_offset[field_yvel0] + t_offset);
  }
  if (fields[field_yvel1] == 1) {
    clover_unpack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.yvel1, left_rcv_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, vertex_data,
                               left_right_offset[field_yvel1] + t_offset);
  }
  if (fields[field_vol_flux_x] == 1) {
    clover_unpack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.vol_flux_x, left_rcv_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, x_face_data,
                               left_right_offset[field_vol_flux_x] + t_offset);
  }
  if (fields[field_vol_flux_y] == 1) {
    clover_unpack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.vol_flux_y, left_rcv_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, y_face_data,
                               left_right_offset[field_vol_flux_y] + t_offset);
  }
  if (fields[field_mass_flux_x] == 1) {
    clover_unpack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.mass_flux_x, left_rcv_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, x_face_data,
                               left_right_offset[field_mass_flux_x] + t_offset);
  }
  if (fields[field_mass_flux_y] == 1) {
    clover_unpack_message_left(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.mass_flux_y, left_rcv_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, y_face_data,
                               left_right_offset[field_mass_flux_y] + t_offset);
  }
}

void clover_pack_right(global_variables &globals, clover::Buffer1D<double> &right_snd_buffer, int tile, const int fields[NUM_FIELDS],
                       int depth, int left_right_offset[NUM_FIELDS]) {

  tile_type &t = globals.chunk.tiles[tile];
  int t_offset = (t.info.t_bottom - globals.chunk.bottom) * depth;

  if (fields[field_density0] == 1) {
    clover_pack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.density0, right_snd_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                              left_right_offset[field_density0] + t_offset);
  }
  if (fields[field_density1] == 1) {
    clover_pack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.density1, right_snd_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                              left_right_offset[field_density1] + t_offset);
  }
  if (fields[field_energy0] == 1) {
    clover_pack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.energy0, right_snd_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                              left_right_offset[field_energy0] + t_offset);
  }
  if (fields[field_energy1] == 1) {
    clover_pack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.energy1, right_snd_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                              left_right_offset[field_energy1] + t_offset);
  }
  if (fields[field_pressure] == 1) {
    clover_pack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.pressure, right_snd_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                              left_right_offset[field_pressure] + t_offset);
  }
  if (fields[field_viscosity] == 1) {
    clover_pack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.viscosity, right_snd_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                              left_right_offset[field_viscosity] + t_offset);
  }
  if (fields[field_soundspeed] == 1) {
    clover_pack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.soundspeed, right_snd_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                              left_right_offset[field_soundspeed] + t_offset);
  }
  if (fields[field_xvel0] == 1) {
    clover_pack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.xvel0, right_snd_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, vertex_data,
                              left_right_offset[field_xvel0] + t_offset);
  }
  if (fields[field_xvel1] == 1) {
    clover_pack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.xvel1, right_snd_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, vertex_data,
                              left_right_offset[field_xvel1] + t_offset);
  }
  if (fields[field_yvel0] == 1) {
    clover_pack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.yvel0, right_snd_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, vertex_data,
                              left_right_offset[field_yvel0] + t_offset);
  }
  if (fields[field_yvel1] == 1) {
    clover_pack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.yvel1, right_snd_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, vertex_data,
                              left_right_offset[field_yvel1] + t_offset);
  }
  if (fields[field_vol_flux_x] == 1) {
    clover_pack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.vol_flux_x, right_snd_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, x_face_data,
                              left_right_offset[field_vol_flux_x] + t_offset);
  }
  if (fields[field_vol_flux_y] == 1) {
    clover_pack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.vol_flux_y, right_snd_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, y_face_data,
                              left_right_offset[field_vol_flux_y] + t_offset);
  }
  if (fields[field_mass_flux_x] == 1) {
    clover_pack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.mass_flux_x, right_snd_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, x_face_data,
                              left_right_offset[field_mass_flux_x] + t_offset);
  }
  if (fields[field_mass_flux_y] == 1) {
    clover_pack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.mass_flux_y, right_snd_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, y_face_data,
                              left_right_offset[field_mass_flux_y] + t_offset);
  }
}

void clover_unpack_right(global_variables &globals, clover::Buffer1D<double> &right_rcv_buffer, const int fields[NUM_FIELDS], int tile,
                         int depth, int left_right_offset[NUM_FIELDS]) {

  tile_type &t = globals.chunk.tiles[tile];
  int t_offset = (t.info.t_bottom - globals.chunk.bottom) * depth;

  if (fields[field_density0] == 1) {
    clover_unpack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.density0, right_rcv_buffer,
                                cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                                left_right_offset[field_density0] + t_offset);
  }
  if (fields[field_density1] == 1) {
    clover_unpack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.density1, right_rcv_buffer,
                                cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                                left_right_offset[field_density1] + t_offset);
  }
  if (fields[field_energy0] == 1) {
    clover_unpack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.energy0, right_rcv_buffer,
                                cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                                left_right_offset[field_energy0] + t_offset);
  }
  if (fields[field_energy1] == 1) {
    clover_unpack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.energy1, right_rcv_buffer,
                                cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                                left_right_offset[field_energy1] + t_offset);
  }
  if (fields[field_pressure] == 1) {
    clover_unpack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.pressure, right_rcv_buffer,
                                cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                                left_right_offset[field_pressure] + t_offset);
  }
  if (fields[field_viscosity] == 1) {
    clover_unpack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.viscosity, right_rcv_buffer,
                                cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                                left_right_offset[field_viscosity] + t_offset);
  }
  if (fields[field_soundspeed] == 1) {
    clover_unpack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.soundspeed, right_rcv_buffer,
                                cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                                left_right_offset[field_soundspeed] + t_offset);
  }
  if (fields[field_xvel0] == 1) {
    clover_unpack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.xvel0, right_rcv_buffer,
                                cell_data, vertex_data, x_face_data, y_face_data, depth, vertex_data,
                                left_right_offset[field_xvel0] + t_offset);
  }
  if (fields[field_xvel1] == 1) {
    clover_unpack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.xvel1, right_rcv_buffer,
                                cell_data, vertex_data, x_face_data, y_face_data, depth, vertex_data,
                                left_right_offset[field_xvel1] + t_offset);
  }
  if (fields[field_yvel0] == 1) {
    clover_unpack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.yvel0, right_rcv_buffer,
                                cell_data, vertex_data, x_face_data, y_face_data, depth, vertex_data,
                                left_right_offset[field_yvel0] + t_offset);
  }
  if (fields[field_yvel1] == 1) {
    clover_unpack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.yvel1, right_rcv_buffer,
                                cell_data, vertex_data, x_face_data, y_face_data, depth, vertex_data,
                                left_right_offset[field_yvel1] + t_offset);
  }
  if (fields[field_vol_flux_x] == 1) {
    clover_unpack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.vol_flux_x, right_rcv_buffer,
                                cell_data, vertex_data, x_face_data, y_face_data, depth, x_face_data,
                                left_right_offset[field_vol_flux_x] + t_offset);
  }
  if (fields[field_vol_flux_y] == 1) {
    clover_unpack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.vol_flux_y, right_rcv_buffer,
                                cell_data, vertex_data, x_face_data, y_face_data, depth, y_face_data,
                                left_right_offset[field_vol_flux_y] + t_offset);
  }
  if (fields[field_mass_flux_x] == 1) {
    clover_unpack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.mass_flux_x, right_rcv_buffer,
                                cell_data, vertex_data, x_face_data, y_face_data, depth, x_face_data,
                                left_right_offset[field_mass_flux_x] + t_offset);
  }
  if (fields[field_mass_flux_y] == 1) {
    clover_unpack_message_right(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.mass_flux_y, right_rcv_buffer,
                                cell_data, vertex_data, x_face_data, y_face_data, depth, y_face_data,
                                left_right_offset[field_mass_flux_y] + t_offset);
  }
}

void clover_pack_top(global_variables &globals, clover::Buffer1D<double> &top_snd_buffer, int tile, const int fields[NUM_FIELDS], int depth,
                     int bottom_top_offset[NUM_FIELDS]) {

  tile_type &t = globals.chunk.tiles[tile];
  int t_offset = (t.info.t_left - globals.chunk.left) * depth;

  if (fields[field_density0] == 1) {
    clover_pack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.density0, top_snd_buffer,
                            cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                            bottom_top_offset[field_density0] + t_offset);
  }
  if (fields[field_density1] == 1) {
    clover_pack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.density1, top_snd_buffer,
                            cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                            bottom_top_offset[field_density1] + t_offset);
  }
  if (fields[field_energy0] == 1) {
    clover_pack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.energy0, top_snd_buffer, cell_data,
                            vertex_data, x_face_data, y_face_data, depth, cell_data, bottom_top_offset[field_energy0] + t_offset);
  }
  if (fields[field_energy1] == 1) {
    clover_pack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.energy1, top_snd_buffer, cell_data,
                            vertex_data, x_face_data, y_face_data, depth, cell_data, bottom_top_offset[field_energy1] + t_offset);
  }
  if (fields[field_pressure] == 1) {
    clover_pack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.pressure, top_snd_buffer,
                            cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                            bottom_top_offset[field_pressure] + t_offset);
  }
  if (fields[field_viscosity] == 1) {
    clover_pack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.viscosity, top_snd_buffer,
                            cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                            bottom_top_offset[field_viscosity] + t_offset);
  }
  if (fields[field_soundspeed] == 1) {
    clover_pack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.soundspeed, top_snd_buffer,
                            cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                            bottom_top_offset[field_soundspeed] + t_offset);
  }
  if (fields[field_xvel0] == 1) {
    clover_pack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.xvel0, top_snd_buffer, cell_data,
                            vertex_data, x_face_data, y_face_data, depth, vertex_data, bottom_top_offset[field_xvel0] + t_offset);
  }
  if (fields[field_xvel1] == 1) {
    clover_pack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.xvel1, top_snd_buffer, cell_data,
                            vertex_data, x_face_data, y_face_data, depth, vertex_data, bottom_top_offset[field_xvel1] + t_offset);
  }
  if (fields[field_yvel0] == 1) {
    clover_pack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.yvel0, top_snd_buffer, cell_data,
                            vertex_data, x_face_data, y_face_data, depth, vertex_data, bottom_top_offset[field_yvel0] + t_offset);
  }
  if (fields[field_yvel1] == 1) {
    clover_pack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.yvel1, top_snd_buffer, cell_data,
                            vertex_data, x_face_data, y_face_data, depth, vertex_data, bottom_top_offset[field_yvel1] + t_offset);
  }
  if (fields[field_vol_flux_x] == 1) {
    clover_pack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.vol_flux_x, top_snd_buffer,
                            cell_data, vertex_data, x_face_data, y_face_data, depth, x_face_data,
                            bottom_top_offset[field_vol_flux_x] + t_offset);
  }
  if (fields[field_vol_flux_y] == 1) {
    clover_pack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.vol_flux_y, top_snd_buffer,
                            cell_data, vertex_data, x_face_data, y_face_data, depth, y_face_data,
                            bottom_top_offset[field_vol_flux_y] + t_offset);
  }
  if (fields[field_mass_flux_x] == 1) {
    clover_pack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.mass_flux_x, top_snd_buffer,
                            cell_data, vertex_data, x_face_data, y_face_data, depth, x_face_data,
                            bottom_top_offset[field_mass_flux_x] + t_offset);
  }
  if (fields[field_mass_flux_y] == 1) {
    clover_pack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.mass_flux_y, top_snd_buffer,
                            cell_data, vertex_data, x_face_data, y_face_data, depth, y_face_data,
                            bottom_top_offset[field_mass_flux_y] + t_offset);
  }
}

void clover_unpack_top(global_variables &globals, clover::Buffer1D<double> &top_rcv_buffer, const int fields[NUM_FIELDS], int tile,
                       int depth, int bottom_top_offset[NUM_FIELDS]) {

  tile_type &t = globals.chunk.tiles[tile];
  int t_offset = (t.info.t_left - globals.chunk.left) * depth;

  if (fields[field_density0] == 1) {
    clover_unpack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.density0, top_rcv_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                              bottom_top_offset[field_density0] + t_offset);
  }
  if (fields[field_density1] == 1) {
    clover_unpack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.density1, top_rcv_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                              bottom_top_offset[field_density1] + t_offset);
  }
  if (fields[field_energy0] == 1) {
    clover_unpack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.energy0, top_rcv_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                              bottom_top_offset[field_energy0] + t_offset);
  }
  if (fields[field_energy1] == 1) {
    clover_unpack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.energy1, top_rcv_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                              bottom_top_offset[field_energy1] + t_offset);
  }
  if (fields[field_pressure] == 1) {
    clover_unpack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.pressure, top_rcv_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                              bottom_top_offset[field_pressure] + t_offset);
  }
  if (fields[field_viscosity] == 1) {
    clover_unpack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.viscosity, top_rcv_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                              bottom_top_offset[field_viscosity] + t_offset);
  }
  if (fields[field_soundspeed] == 1) {
    clover_unpack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.soundspeed, top_rcv_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                              bottom_top_offset[field_soundspeed] + t_offset);
  }
  if (fields[field_xvel0] == 1) {
    clover_unpack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.xvel0, top_rcv_buffer, cell_data,
                              vertex_data, x_face_data, y_face_data, depth, vertex_data, bottom_top_offset[field_xvel0] + t_offset);
  }
  if (fields[field_xvel1] == 1) {
    clover_unpack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.xvel1, top_rcv_buffer, cell_data,
                              vertex_data, x_face_data, y_face_data, depth, vertex_data, bottom_top_offset[field_xvel1] + t_offset);
  }
  if (fields[field_yvel0] == 1) {
    clover_unpack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.yvel0, top_rcv_buffer, cell_data,
                              vertex_data, x_face_data, y_face_data, depth, vertex_data, bottom_top_offset[field_yvel0] + t_offset);
  }
  if (fields[field_yvel1] == 1) {
    clover_unpack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.yvel1, top_rcv_buffer, cell_data,
                              vertex_data, x_face_data, y_face_data, depth, vertex_data, bottom_top_offset[field_yvel1] + t_offset);
  }
  if (fields[field_vol_flux_x] == 1) {
    clover_unpack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.vol_flux_x, top_rcv_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, x_face_data,
                              bottom_top_offset[field_vol_flux_x] + t_offset);
  }
  if (fields[field_vol_flux_y] == 1) {
    clover_unpack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.vol_flux_y, top_rcv_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, y_face_data,
                              bottom_top_offset[field_vol_flux_y] + t_offset);
  }
  if (fields[field_mass_flux_x] == 1) {
    clover_unpack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.mass_flux_x, top_rcv_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, x_face_data,
                              bottom_top_offset[field_mass_flux_x] + t_offset);
  }
  if (fields[field_mass_flux_y] == 1) {
    clover_unpack_message_top(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.mass_flux_y, top_rcv_buffer,
                              cell_data, vertex_data, x_face_data, y_face_data, depth, y_face_data,
                              bottom_top_offset[field_mass_flux_y] + t_offset);
  }
}

void clover_pack_bottom(global_variables &globals, clover::Buffer1D<double> &bottom_snd_buffer, int tile, const int fields[NUM_FIELDS],
                        int depth, int bottom_top_offset[NUM_FIELDS]) {

  tile_type &t = globals.chunk.tiles[tile];
  int t_offset = (t.info.t_left - globals.chunk.left) * depth;

  if (fields[field_density0] == 1) {
    clover_pack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.density0, bottom_snd_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                               bottom_top_offset[field_density0] + t_offset);
  }
  if (fields[field_density1] == 1) {
    clover_pack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.density1, bottom_snd_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                               bottom_top_offset[field_density1] + t_offset);
  }
  if (fields[field_energy0] == 1) {
    clover_pack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.energy0, bottom_snd_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                               bottom_top_offset[field_energy0] + t_offset);
  }
  if (fields[field_energy1] == 1) {
    clover_pack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.energy1, bottom_snd_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                               bottom_top_offset[field_energy1] + t_offset);
  }
  if (fields[field_pressure] == 1) {
    clover_pack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.pressure, bottom_snd_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                               bottom_top_offset[field_pressure] + t_offset);
  }
  if (fields[field_viscosity] == 1) {
    clover_pack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.viscosity, bottom_snd_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                               bottom_top_offset[field_viscosity] + t_offset);
  }
  if (fields[field_soundspeed] == 1) {
    clover_pack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.soundspeed, bottom_snd_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                               bottom_top_offset[field_soundspeed] + t_offset);
  }
  if (fields[field_xvel0] == 1) {
    clover_pack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.xvel0, bottom_snd_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, vertex_data,
                               bottom_top_offset[field_xvel0] + t_offset);
  }
  if (fields[field_xvel1] == 1) {
    clover_pack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.xvel1, bottom_snd_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, vertex_data,
                               bottom_top_offset[field_xvel1] + t_offset);
  }
  if (fields[field_yvel0] == 1) {
    clover_pack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.yvel0, bottom_snd_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, vertex_data,
                               bottom_top_offset[field_yvel0] + t_offset);
  }
  if (fields[field_yvel1] == 1) {
    clover_pack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.yvel1, bottom_snd_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, vertex_data,
                               bottom_top_offset[field_yvel1] + t_offset);
  }
  if (fields[field_vol_flux_x] == 1) {
    clover_pack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.vol_flux_x, bottom_snd_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, x_face_data,
                               bottom_top_offset[field_vol_flux_x] + t_offset);
  }
  if (fields[field_vol_flux_y] == 1) {
    clover_pack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.vol_flux_y, bottom_snd_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, y_face_data,
                               bottom_top_offset[field_vol_flux_y] + t_offset);
  }
  if (fields[field_mass_flux_x] == 1) {
    clover_pack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.mass_flux_x, bottom_snd_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, x_face_data,
                               bottom_top_offset[field_mass_flux_x] + t_offset);
  }
  if (fields[field_mass_flux_y] == 1) {
    clover_pack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.mass_flux_y, bottom_snd_buffer,
                               cell_data, vertex_data, x_face_data, y_face_data, depth, y_face_data,
                               bottom_top_offset[field_mass_flux_y] + t_offset);
  }
}

void clover_unpack_bottom(global_variables &globals, clover::Buffer1D<double> &bottom_rcv_buffer, const int fields[NUM_FIELDS], int tile,
                          int depth, int bottom_top_offset[NUM_FIELDS]) {

  tile_type &t = globals.chunk.tiles[tile];
  int t_offset = (t.info.t_left - globals.chunk.left) * depth;

  if (fields[field_density0] == 1) {
    clover_unpack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.density0, bottom_rcv_buffer,
                                 cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                                 bottom_top_offset[field_density0] + t_offset);
  }
  if (fields[field_density1] == 1) {
    clover_unpack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.density1, bottom_rcv_buffer,
                                 cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                                 bottom_top_offset[field_density1] + t_offset);
  }
  if (fields[field_energy0] == 1) {
    clover_unpack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.energy0, bottom_rcv_buffer,
                                 cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                                 bottom_top_offset[field_energy0] + t_offset);
  }
  if (fields[field_energy1] == 1) {
    clover_unpack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.energy1, bottom_rcv_buffer,
                                 cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                                 bottom_top_offset[field_energy1] + t_offset);
  }
  if (fields[field_pressure] == 1) {
    clover_unpack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.pressure, bottom_rcv_buffer,
                                 cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                                 bottom_top_offset[field_pressure] + t_offset);
  }
  if (fields[field_viscosity] == 1) {
    clover_unpack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.viscosity, bottom_rcv_buffer,
                                 cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                                 bottom_top_offset[field_viscosity] + t_offset);
  }
  if (fields[field_soundspeed] == 1) {
    clover_unpack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.soundspeed, bottom_rcv_buffer,
                                 cell_data, vertex_data, x_face_data, y_face_data, depth, cell_data,
                                 bottom_top_offset[field_soundspeed] + t_offset);
  }
  if (fields[field_xvel0] == 1) {
    clover_unpack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.xvel0, bottom_rcv_buffer,
                                 cell_data, vertex_data, x_face_data, y_face_data, depth, vertex_data,
                                 bottom_top_offset[field_xvel0] + t_offset);
  }
  if (fields[field_xvel1] == 1) {
    clover_unpack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.xvel1, bottom_rcv_buffer,
                                 cell_data, vertex_data, x_face_data, y_face_data, depth, vertex_data,
                                 bottom_top_offset[field_xvel1] + t_offset);
  }
  if (fields[field_yvel0] == 1) {
    clover_unpack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.yvel0, bottom_rcv_buffer,
                                 cell_data, vertex_data, x_face_data, y_face_data, depth, vertex_data,
                                 bottom_top_offset[field_yvel0] + t_offset);
  }
  if (fields[field_yvel1] == 1) {
    clover_unpack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.yvel1, bottom_rcv_buffer,
                                 cell_data, vertex_data, x_face_data, y_face_data, depth, vertex_data,
                                 bottom_top_offset[field_yvel1] + t_offset);
  }
  if (fields[field_vol_flux_x] == 1) {
    clover_unpack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.vol_flux_x, bottom_rcv_buffer,
                                 cell_data, vertex_data, x_face_data, y_face_data, depth, x_face_data,
                                 bottom_top_offset[field_vol_flux_x] + t_offset);
  }
  if (fields[field_vol_flux_y] == 1) {
    clover_unpack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.vol_flux_y, bottom_rcv_buffer,
                                 cell_data, vertex_data, x_face_data, y_face_data, depth, y_face_data,
                                 bottom_top_offset[field_vol_flux_y] + t_offset);
  }
  if (fields[field_mass_flux_x] == 1) {
    clover_unpack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.mass_flux_x,
                                 bottom_rcv_buffer, cell_data, vertex_data, x_face_data, y_face_data, depth, x_face_data,
                                 bottom_top_offset[field_mass_flux_x] + t_offset);
  }
  if (fields[field_mass_flux_y] == 1) {
    clover_unpack_message_bottom(globals, t.info.t_xmin, t.info.t_xmax, t.info.t_ymin, t.info.t_ymax, t.field.mass_flux_y,
                                 bottom_rcv_buffer, cell_data, vertex_data, x_face_data, y_face_data, depth, y_face_data,
                                 bottom_top_offset[field_mass_flux_y] + t_offset);
  }
}

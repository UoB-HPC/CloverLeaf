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

#include "comms_kernel.h"
#include "comms.h"
#include "pack_kernel.h"

void clover_allocate_buffers(global_variables &globals, parallel_ &parallel) {

  // Unallocated buffers for external boundaries caused issues on some systems so they are now
  //  all allocated
  if (parallel.task == globals.chunk.task) {

    //		new(&globals.chunk.left_snd)   Kokkos::View<double *>("left_snd", 10 * 2 * (globals.chunk.y_max +	5));
    //		new(&globals.chunk.left_rcv)   Kokkos::View<double *>("left_rcv", 10 * 2 * (globals.chunk.y_max +	5));
    //		new(&globals.chunk.right_snd)  Kokkos::View<double *>("right_snd", 10 * 2 * (globals.chunk.y_max +	5));
    //		new(&globals.chunk.right_rcv)  Kokkos::View<double *>("right_rcv", 10 * 2 * (globals.chunk.y_max +	5));
    //		new(&globals.chunk.bottom_snd) Kokkos::View<double *>("bottom_snd", 10 * 2 * (globals.chunk.x_max +	5));
    //		new(&globals.chunk.bottom_rcv) Kokkos::View<double *>("bottom_rcv", 10 * 2 * (globals.chunk.x_max +	5));
    //		new(&globals.chunk.top_snd)    Kokkos::View<double *>("top_snd", 10 * 2 * (globals.chunk.x_max +	5));
    //		new(&globals.chunk.top_rcv)    Kokkos::View<double *>("top_rcv", 10 * 2 * (globals.chunk.x_max +	5));
    //
    //		// Create host mirrors of device buffers. This makes this, and deep_copy, a no-op if the View is in host memory already.
    //		globals.chunk.hm_left_snd = Kokkos::create_mirror_view(
    //				globals.chunk.left_snd);
    //		globals.chunk.hm_left_rcv = Kokkos::create_mirror_view(
    //				globals.chunk.left_rcv);
    //		globals.chunk.hm_right_snd = Kokkos::create_mirror_view(
    //				globals.chunk.right_snd);
    //		globals.chunk.hm_right_rcv = Kokkos::create_mirror_view(
    //				globals.chunk.right_rcv);
    //		globals.chunk.hm_bottom_snd = Kokkos::create_mirror_view(
    //				globals.chunk.bottom_snd);
    //		globals.chunk.hm_bottom_rcv = Kokkos::create_mirror_view(
    //				globals.chunk.bottom_rcv);
    //		globals.chunk.hm_top_snd = Kokkos::create_mirror_view(globals.chunk.top_snd);
    //		globals.chunk.hm_top_rcv = Kokkos::create_mirror_view(globals.chunk.top_rcv);
  }
}

void clover_exchange(global_variables &globals, const int fields[NUM_FIELDS], const int depth) {

  // Assuming 1 patch per task, this will be changed

  int left_right_offset[NUM_FIELDS];
  int bottom_top_offset[NUM_FIELDS];

  MPI_Request request[4] = {0};
  int message_count = 0;

  int end_pack_index_left_right = 0;
  int end_pack_index_bottom_top = 0;
  for (int field = 0; field < NUM_FIELDS; ++field) {
    if (fields[field] == 1) {
      left_right_offset[field] = end_pack_index_left_right;
      bottom_top_offset[field] = end_pack_index_bottom_top;
      end_pack_index_left_right += depth * (globals.chunk.y_max + 5);
      end_pack_index_bottom_top += depth * (globals.chunk.x_max + 5);
    }
  }

  clover::Buffer1D<double> left_rcv_device(globals.context, end_pack_index_left_right);
  clover::Buffer1D<double> left_snd_device(globals.context, end_pack_index_left_right);
  clover::Buffer1D<double> right_rcv_device(globals.context, end_pack_index_left_right);
  clover::Buffer1D<double> right_snd_device(globals.context, end_pack_index_left_right);

  clover::Buffer1D<double> top_rcv_device(globals.context, end_pack_index_bottom_top);
  clover::Buffer1D<double> top_snd_device(globals.context, end_pack_index_bottom_top);
  clover::Buffer1D<double> bottom_rcv_device(globals.context, end_pack_index_bottom_top);
  clover::Buffer1D<double> bottom_snd_device(globals.context, end_pack_index_bottom_top);

  bool stage = globals.config.staging_buffer;

  static auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator alloc = rm.getAllocator(clover::CLResourceManager::allocator_ids[clover::CLResourceManager::HOST]);
  double *left_rcv_staging = stage ? static_cast<double *>(alloc.allocate(left_rcv_device.size * sizeof(double))) : nullptr;
  double *left_snd_staging = stage ? static_cast<double *>(alloc.allocate(left_snd_device.size * sizeof(double))) : nullptr;
  double *right_rcv_staging = stage ? static_cast<double *>(alloc.allocate(right_rcv_device.size * sizeof(double))) : nullptr;
  double *right_snd_staging = stage ? static_cast<double *>(alloc.allocate(right_snd_device.size * sizeof(double))) : nullptr;

  double *top_rcv_staging = stage ? static_cast<double *>(alloc.allocate(top_rcv_device.size * sizeof(double))) : nullptr;
  double *top_snd_staging = stage ? static_cast<double *>(alloc.allocate(top_snd_device.size * sizeof(double))) : nullptr;
  double *bottom_rcv_staging = stage ? static_cast<double *>(alloc.allocate(bottom_rcv_device.size * sizeof(double))) : nullptr;
  double *bottom_snd_staging = stage ? static_cast<double *>(alloc.allocate(bottom_snd_device.size * sizeof(double))) : nullptr;

  auto deviceToStaging = [](double *staging, clover::Buffer1D<double> &device) {
    raja_copy(staging, device.data, device.size * sizeof(double));
  };

  auto stagingToDevice = [](double *staging, clover::Buffer1D<double> &device) {
    raja_copy(device.data, staging, device.size * sizeof(double));
  };

  if (globals.chunk.chunk_neighbours[chunk_left] != external_face) {
    // do left exchanges
    // Find left hand tiles
    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
      if (globals.chunk.tiles[tile].info.external_tile_mask[tile_left] == 1) {
        clover_pack_left(globals, left_snd_device, tile, fields, depth, left_right_offset);
      }
    }
    if (stage) deviceToStaging(left_snd_staging, left_snd_device);
    else {
      left_snd_staging = left_snd_device.data;
      left_rcv_staging = left_rcv_device.data;
    }
    // send and recv messages to the left
    clover_send_recv_message_left(globals, left_snd_staging, left_rcv_staging, end_pack_index_left_right, 1, 2, request[message_count],
                                  request[message_count + 1]);
    message_count += 2;
  }

  if (globals.chunk.chunk_neighbours[chunk_right] != external_face) {
    // do right exchanges
    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
      if (globals.chunk.tiles[tile].info.external_tile_mask[tile_right] == 1) {
        clover_pack_right(globals, right_snd_device, tile, fields, depth, left_right_offset);
      }
    }

    if (stage) deviceToStaging(right_snd_staging, right_snd_device);
    else {
      right_snd_staging = right_snd_device.data;
      right_rcv_staging = right_rcv_device.data;
    }
    // send message to the right
    clover_send_recv_message_right(globals, right_snd_staging, right_rcv_staging, end_pack_index_left_right, 2, 1, request[message_count],
                                   request[message_count + 1]);
    message_count += 2;
  }

  // make a call to wait / sync
  MPI_Waitall(message_count, request, MPI_STATUS_IGNORE);
  if (stage) {
    stagingToDevice(left_rcv_staging, left_rcv_device);
    stagingToDevice(right_rcv_staging, right_rcv_device);
  }

  // Copy back to the device

  // unpack in left direction
  if (globals.chunk.chunk_neighbours[chunk_left] != external_face) {
    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
      if (globals.chunk.tiles[tile].info.external_tile_mask[tile_left] == 1) {
        clover_unpack_left(globals, left_rcv_device, fields, tile, depth, left_right_offset);
      }
    }
  }

  // unpack in right direction
  if (globals.chunk.chunk_neighbours[chunk_right] != external_face) {
    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
      if (globals.chunk.tiles[tile].info.external_tile_mask[tile_right] == 1) {
        clover_unpack_right(globals, right_rcv_device, fields, tile, depth, left_right_offset);
      }
    }
  }

  message_count = 0;
  for (MPI_Request &i : request)
    i = {};

  if (globals.chunk.chunk_neighbours[chunk_bottom] != external_face) {
    // do bottom exchanges
    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
      if (globals.chunk.tiles[tile].info.external_tile_mask[tile_bottom] == 1) {
        clover_pack_bottom(globals, bottom_snd_device, tile, fields, depth, bottom_top_offset);
      }
    }

    if (stage) deviceToStaging(bottom_snd_staging, bottom_snd_device);
    else {
      bottom_snd_staging = bottom_snd_device.data;
      bottom_rcv_staging = bottom_rcv_device.data;
    }
    // send message downwards
    clover_send_recv_message_bottom(globals, bottom_snd_staging, bottom_rcv_staging, end_pack_index_bottom_top, 3, 4,
                                    request[message_count], request[message_count + 1]);
    message_count += 2;
  }

  if (globals.chunk.chunk_neighbours[chunk_top] != external_face) {
    // do top exchanges
    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
      if (globals.chunk.tiles[tile].info.external_tile_mask[tile_top] == 1) {
        clover_pack_top(globals, top_snd_device, tile, fields, depth, bottom_top_offset);
      }
    }

    if (stage) deviceToStaging(top_snd_staging, top_snd_device);
    else {
      top_snd_staging = top_snd_device.data;
      top_rcv_staging = top_rcv_device.data;
    }
    // send message upwards
    clover_send_recv_message_top(globals, top_snd_staging, top_rcv_staging, end_pack_index_bottom_top, 4, 3, request[message_count],
                                 request[message_count + 1]);
    message_count += 2;
  }

  // need to make a call to wait / sync
  MPI_Waitall(message_count, request, MPI_STATUS_IGNORE);
  if (stage) {
    stagingToDevice(bottom_rcv_staging, bottom_rcv_device);
    stagingToDevice(top_rcv_staging, top_rcv_device);
  }

  // Copy back to the device

  // unpack in top direction
  if (globals.chunk.chunk_neighbours[chunk_top] != external_face) {
    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
      if (globals.chunk.tiles[tile].info.external_tile_mask[tile_top] == 1) {
        clover_unpack_top(globals, top_rcv_device, fields, tile, depth, bottom_top_offset);
      }
    }
  }

  // unpack in bottom direction
  if (globals.chunk.chunk_neighbours[chunk_bottom] != external_face) {
    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
      if (globals.chunk.tiles[tile].info.external_tile_mask[tile_bottom] == 1) {
        clover_unpack_bottom(globals, bottom_rcv_device, fields, tile, depth, bottom_top_offset);
      }
    }
  }

  left_rcv_device.release();
  left_snd_device.release();
  right_rcv_device.release();
  right_snd_device.release();
  top_rcv_device.release();
  top_snd_device.release();
  bottom_rcv_device.release();
  bottom_snd_device.release();

  if (stage && left_rcv_staging) clover::dealloc(left_rcv_staging);
  if (stage && left_snd_staging) clover::dealloc(left_snd_staging);
  if (stage && right_rcv_staging) clover::dealloc(right_rcv_staging);
  if (stage && right_snd_staging) clover::dealloc(right_snd_staging);
  if (stage && top_rcv_staging) clover::dealloc(top_rcv_staging);
  if (stage && top_snd_staging) clover::dealloc(top_snd_staging);
  if (stage && bottom_rcv_staging) clover::dealloc(bottom_rcv_staging);
  if (stage && bottom_snd_staging) clover::dealloc(bottom_snd_staging);
}

void clover_send_recv_message_left(global_variables &globals, double *left_snd_buffer, double *left_rcv_buffer, int total_size,
                                   int tag_send, int tag_recv, MPI_Request &req_send, MPI_Request &req_recv) {
  clover::checkError(rajaDeviceSynchronize());
  int left_task = globals.chunk.chunk_neighbours[chunk_left] - 1;
  MPI_Isend(left_snd_buffer, total_size, MPI_DOUBLE, left_task, tag_send, MPI_COMM_WORLD, &req_send);
  MPI_Irecv(left_rcv_buffer, total_size, MPI_DOUBLE, left_task, tag_recv, MPI_COMM_WORLD, &req_recv);
}
void clover_send_recv_message_right(global_variables &globals, double *right_snd_buffer, double *right_rcv_buffer, int total_size,
                                    int tag_send, int tag_recv, MPI_Request &req_send, MPI_Request &req_recv) {
  clover::checkError(rajaDeviceSynchronize());
  int right_task = globals.chunk.chunk_neighbours[chunk_right] - 1;
  MPI_Isend(right_snd_buffer, total_size, MPI_DOUBLE, right_task, tag_send, MPI_COMM_WORLD, &req_send);
  MPI_Irecv(right_rcv_buffer, total_size, MPI_DOUBLE, right_task, tag_recv, MPI_COMM_WORLD, &req_recv);
}
void clover_send_recv_message_top(global_variables &globals, double *top_snd_buffer, double *top_rcv_buffer, int total_size, int tag_send,
                                  int tag_recv, MPI_Request &req_send, MPI_Request &req_recv) {
  clover::checkError(rajaDeviceSynchronize());
  int top_task = globals.chunk.chunk_neighbours[chunk_top] - 1;
  MPI_Isend(top_snd_buffer, total_size, MPI_DOUBLE, top_task, tag_send, MPI_COMM_WORLD, &req_send);
  MPI_Irecv(top_rcv_buffer, total_size, MPI_DOUBLE, top_task, tag_recv, MPI_COMM_WORLD, &req_recv);
}
void clover_send_recv_message_bottom(global_variables &globals, double *bottom_snd_buffer, double *bottom_rcv_buffer, int total_size,
                                     int tag_send, int tag_recv, MPI_Request &req_send, MPI_Request &req_recv) {
  clover::checkError(rajaDeviceSynchronize());
  int bottom_task = globals.chunk.chunk_neighbours[chunk_bottom] - 1;
  MPI_Isend(bottom_snd_buffer, total_size, MPI_DOUBLE, bottom_task, tag_send, MPI_COMM_WORLD, &req_send);
  MPI_Irecv(bottom_rcv_buffer, total_size, MPI_DOUBLE, bottom_task, tag_recv, MPI_COMM_WORLD, &req_recv);
}

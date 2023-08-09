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
#include "context.h"
#include "pack_kernel.h"

void clover_allocate_buffers(global_variables &globals, parallel_ &parallel) {

  // Unallocated buffers for external boundaries caused issues on some systems so they are now
  //  all allocated
  if (parallel.task == globals.chunk.task) {

    //		new(&globals.chunk.left_snd_buffer)   Kokkos::View<double *>("left_snd_buffer", 10 * 2 *
    //(globals.chunk.y_max +	5)); 		new(&globals.chunk.left_rcv_buffer)   Kokkos::View<double
    //*>("left_rcv_buffer", 10 * 2 * (globals.chunk.y_max +	5)); 		new(&globals.chunk.right_snd_buffer)
    // Kokkos::View<double *>("right_snd_buffer", 10
    //* 2 * (globals.chunk.y_max +	5)); 		new(&globals.chunk.right_rcv_buffer)  Kokkos::View<double
    //*>("right_rcv_buffer", 10 * 2 * (globals.chunk.y_max +	5)); 		new(&globals.chunk.bottom_snd_buffer)
    // Kokkos::View<double *>("bottom_snd_buffer", 10 * 2 * (globals.chunk.x_max +	5));
    //		new(&globals.chunk.bottom_rcv_buffer) Kokkos::View<double *>("bottom_rcv_buffer", 10 * 2 *
    //(globals.chunk.x_max +	5)); 		new(&globals.chunk.top_snd_buffer)    Kokkos::View<double
    //*>("top_snd_buffer", 10
    //* 2 * (globals.chunk.x_max +	5)); 		new(&globals.chunk.top_rcv_buffer)    Kokkos::View<double
    //*>("top_rcv_buffer", 10 * 2 * (globals.chunk.x_max +	5));
    //
    //		// Create host mirrors of device buffers. This makes this, and deep_copy, a no-op if the View is in host
    // memory already. 		globals.chunk.hm_left_snd_buffer = Kokkos::create_mirror_view(
    // globals.chunk.left_snd_buffer); 		globals.chunk.hm_left_rcv_buffer = Kokkos::create_mirror_view(
    //				globals.chunk.left_rcv_buffer);
    //		globals.chunk.hm_right_snd_buffer = Kokkos::create_mirror_view(
    //				globals.chunk.right_snd_buffer);
    //		globals.chunk.hm_right_rcv_buffer = Kokkos::create_mirror_view(
    //				globals.chunk.right_rcv_buffer);
    //		globals.chunk.hm_bottom_snd_buffer = Kokkos::create_mirror_view(
    //				globals.chunk.bottom_snd_buffer);
    //		globals.chunk.hm_bottom_rcv_buffer = Kokkos::create_mirror_view(
    //				globals.chunk.bottom_rcv_buffer);
    //		globals.chunk.hm_top_snd_buffer = Kokkos::create_mirror_view(globals.chunk.top_snd_buffer);
    //		globals.chunk.hm_top_rcv_buffer = Kokkos::create_mirror_view(globals.chunk.top_rcv_buffer);
  }
}
#if !(defined(__HIPSYCL__) || defined(__OPENSYCL__))
template <typename A> decltype(auto) get_native_ptr_or_throw(sycl::interop_handle &ih, A accessor) {
  using sycl::backend;
  using T = std::remove_cv_t<typename decltype(accessor)::value_type>;
  switch (ih.get_backend()) {
    case backend::ext_oneapi_level_zero: return reinterpret_cast<T *>(ih.get_native_mem<backend::ext_oneapi_level_zero>(accessor));
  #ifdef SYCL_EXT_ONEAPI_BACKEND_CUDA
    case backend::ext_oneapi_cuda: return reinterpret_cast<T *>(ih.get_native_mem<backend::ext_oneapi_cuda>(accessor));
  #endif
  #ifdef SYCL_EXT_ONEAPI_BACKEND_HIP
    case backend::ext_oneapi_hip: return reinterpret_cast<T *>(ih.get_native_mem<backend::ext_oneapi_hip>(accessor));
  #endif
    default:
      std::stringstream ss;
      ss << "backend " << ih.get_backend() << " does not support a pointer-based sycl::interop_handle::get_native_mem";
      throw std::logic_error(ss.str());
  }
}
#endif

void clover_send_recv_message(global_variables &globals, chunk_neighbour_type tpe, clover::Buffer1D<double> &snd_buffer,
                              clover::Buffer1D<double> &rcv_buffer, int total_size, int tag_send, int tag_recv, MPI_Request &req_send,
                              MPI_Request &req_recv) {
  int task = globals.chunk.chunk_neighbours[tpe] - 1;
#ifdef USE_HOSTTASK
  if (globals.config.staging_buffer) {
    globals.context.queue.submit([&](sycl::handler &h) {
      auto snd_buffer_acc = snd_buffer.buffer.get_host_access(h, sycl::read_only);
      auto rcv_buffer_acc = rcv_buffer.buffer.get_host_access(h, sycl::write_only);
      h.host_task([=, &req_send, &req_recv]() { // XXX pass handle arg here as copy, not ref!
        MPI_Isend(snd_buffer_acc.get_pointer(), total_size, MPI_DOUBLE, task, tag_send, MPI_COMM_WORLD, &req_send);
        MPI_Irecv(rcv_buffer_acc.get_pointer(), total_size, MPI_DOUBLE, task, tag_recv, MPI_COMM_WORLD, &req_recv);
      });
    });
  } else {
    globals.context.queue.submit([&](sycl::handler &h) {
      auto snd_buffer_acc = snd_buffer.buffer.get_access<sycl::access_mode::read>(h);
      auto rcv_buffer_acc = rcv_buffer.buffer.get_access<sycl::access_mode::write>(h);

      h.host_task([=, &req_send, &req_recv](sycl::interop_handle ih) { // XXX pass handle arg here as copy, not ref!
        MPI_Isend(get_native_ptr_or_throw(ih, snd_buffer_acc), total_size, MPI_DOUBLE, task, tag_send, MPI_COMM_WORLD, &req_send);
        MPI_Irecv(get_native_ptr_or_throw(ih, rcv_buffer_acc), total_size, MPI_DOUBLE, task, tag_recv, MPI_COMM_WORLD, &req_recv);
      });
    });
  }
#else
  if (globals.config.staging_buffer) {
    globals.context.queue.wait_and_throw();
    MPI_Isend(snd_buffer.access_ptr<R>(total_size), total_size, MPI_DOUBLE, task, tag_send, MPI_COMM_WORLD, &req_send);
    MPI_Irecv(rcv_buffer.access_ptr<W>(total_size), total_size, MPI_DOUBLE, task, tag_recv, MPI_COMM_WORLD, &req_recv);
  } else {
  #if defined(__HIPSYCL__) || defined(__OPENSYCL__)
    auto d = globals.context.queue.get_device();
    // Construct the buffers so that get_pointer is not nullptr, only happens once per rank for the lifetime of the program
    if (!rcv_buffer.buffer.get_pointer(d))
      globals.context.queue.submit([&](sycl::handler &h) { h.update(sycl::accessor{rcv_buffer.buffer, h}); }).wait_and_throw();
    if (!snd_buffer.buffer.get_pointer(d))
      globals.context.queue.submit([&](sycl::handler &h) { h.update(sycl::accessor{snd_buffer.buffer, h}); }).wait_and_throw();
    // We can't use host_task here, but since we can pull out the pointers directly, if we synchronise before MPI_Waitall
    // the desired concurrency should still be there
    MPI_Isend(snd_buffer.buffer.get_pointer(d), total_size, MPI_DOUBLE, task, tag_send, MPI_COMM_WORLD, &req_send);
    MPI_Irecv(rcv_buffer.buffer.get_pointer(d), total_size, MPI_DOUBLE, task, tag_recv, MPI_COMM_WORLD, &req_recv);
  #else
    throw std::logic_error("host_task is disabled and staging is also disabled, this won't work");
  #endif
  }
#endif
}

void clover_wait_messages(global_variables &globals, int message_count, std::array<MPI_Request, 4> &request) {
  // need to make a call to wait / sync
#ifdef USE_HOSTTASK
  globals.context.queue
      .submit([&](sycl::handler &h) {
        h.host_task([=, &request]() { // XXX pass handle arg here as copy, not ref!
          MPI_Waitall(message_count, request.data(), MPI_STATUS_IGNORE);
        });
      })
      .wait_and_throw(); // don't proceed until MPI_Waitall call is done
#else
  if (globals.config.staging_buffer) MPI_Waitall(message_count, request.data(), MPI_STATUS_IGNORE);
  else {
  #if defined(__HIPSYCL__) || defined(__OPENSYCL__)
    globals.context.queue.wait_and_throw();
    MPI_Waitall(message_count, request.data(), MPI_STATUS_IGNORE);
  #else
    throw std::logic_error("host_task is disabled and staging is also disabled, this won't work");
  #endif
  }
#endif
}

void clover_exchange(global_variables &globals, const int fields[NUM_FIELDS], const int depth) {

  // Assuming 1 patch per task, this will be changed

  int left_right_offset[NUM_FIELDS];
  int bottom_top_offset[NUM_FIELDS];

  std::array<MPI_Request, 4> request = {};
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

  //  std::cerr << "[C] LR = " << end_pack_index_left_right << " BT = " << end_pack_index_bottom_top << std::endl;
  //  std::cerr << "[S] LR  = " << (globals.chunk.left_rcv_buffer.buffer.size() +
  //  globals.chunk.right_rcv_buffer.buffer.size()) << " BT = " <<  (globals.chunk.top_rcv_buffer.buffer.size() +
  //  globals.chunk.bottom_rcv_buffer.buffer.size()) << std::endl;

  static clover::Buffer1D<double> left_rcv_buffer(globals.context, end_pack_index_left_right);
  static clover::Buffer1D<double> left_snd_buffer(globals.context, end_pack_index_left_right);
  static clover::Buffer1D<double> right_rcv_buffer(globals.context, end_pack_index_left_right);
  static clover::Buffer1D<double> right_snd_buffer(globals.context, end_pack_index_left_right);

  static clover::Buffer1D<double> top_rcv_buffer(globals.context, end_pack_index_bottom_top);
  static clover::Buffer1D<double> top_snd_buffer(globals.context, end_pack_index_bottom_top);
  static clover::Buffer1D<double> bottom_rcv_buffer(globals.context, end_pack_index_bottom_top);
  static clover::Buffer1D<double> bottom_snd_buffer(globals.context, end_pack_index_bottom_top);

  if (globals.chunk.chunk_neighbours[chunk_left] != external_face) {
    // do left exchanges
    // Find left hand tiles
    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
      if (globals.chunk.tiles[tile].info.external_tile_mask[tile_left] == 1) {
        clover_pack_left(globals, left_snd_buffer, tile, fields, depth, left_right_offset);
      }
    }

    // send and recv messages to the left
    clover_send_recv_message_left(globals, left_snd_buffer, left_rcv_buffer, end_pack_index_left_right, 1, 2, request[message_count],
                                  request[message_count + 1]);
    message_count += 2;
  }

  if (globals.chunk.chunk_neighbours[chunk_right] != external_face) {
    // do right exchanges
    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
      if (globals.chunk.tiles[tile].info.external_tile_mask[tile_right] == 1) {
        clover_pack_right(globals, right_snd_buffer, tile, fields, depth, left_right_offset);
      }
    }

    // send message to the right
    clover_send_recv_message_right(globals, right_snd_buffer, right_rcv_buffer, end_pack_index_left_right, 2, 1, request[message_count],
                                   request[message_count + 1]);
    message_count += 2;
  }

  clover_wait_messages(globals, message_count, request);

  // Copy back to the device
  //	Kokkos::deep_copy( left_rcv_buffer, hm_left_rcv_buffer);
  //	Kokkos::deep_copy( right_rcv_buffer, hm_right_rcv_buffer);

  // unpack in left direction
  if (globals.chunk.chunk_neighbours[chunk_left] != external_face) {
    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
      if (globals.chunk.tiles[tile].info.external_tile_mask[tile_left] == 1) {
        clover_unpack_left(globals, left_rcv_buffer, fields, tile, depth, left_right_offset);
      }
    }
  }

  // unpack in right direction
  if (globals.chunk.chunk_neighbours[chunk_right] != external_face) {
    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
      if (globals.chunk.tiles[tile].info.external_tile_mask[tile_right] == 1) {
        clover_unpack_right(globals, right_rcv_buffer, fields, tile, depth, left_right_offset);
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
        clover_pack_bottom(globals, bottom_snd_buffer, tile, fields, depth, bottom_top_offset);
      }
    }

    // send message downwards
    clover_send_recv_message_bottom(globals, bottom_snd_buffer, bottom_rcv_buffer, end_pack_index_bottom_top, 3, 4, request[message_count],
                                    request[message_count + 1]);
    message_count += 2;
  }

  if (globals.chunk.chunk_neighbours[chunk_top] != external_face) {
    // do top exchanges
    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
      if (globals.chunk.tiles[tile].info.external_tile_mask[tile_top] == 1) {
        clover_pack_top(globals, top_snd_buffer, tile, fields, depth, bottom_top_offset);
      }
    }

    // send message upwards
    clover_send_recv_message_top(globals, top_snd_buffer, top_rcv_buffer, end_pack_index_bottom_top, 4, 3, request[message_count],
                                 request[message_count + 1]);
    message_count += 2;
  }

  clover_wait_messages(globals, message_count, request);

  // Copy back to the device
  //	Kokkos::deep_copy(globals.chunk.bottom_rcv_buffer, hm_bottom_rcv_buffer);
  //	Kokkos::deep_copy(globals.chunk.top_rcv_buffer, hm_top_rcv_buffer);

  // unpack in top direction
  if (globals.chunk.chunk_neighbours[chunk_top] != external_face) {
    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
      if (globals.chunk.tiles[tile].info.external_tile_mask[tile_top] == 1) {
        clover_unpack_top(globals, top_rcv_buffer, fields, tile, depth, bottom_top_offset);
      }
    }
  }

  // unpack in bottom direction
  if (globals.chunk.chunk_neighbours[chunk_bottom] != external_face) {
    for (int tile = 0; tile < globals.config.tiles_per_chunk; ++tile) {
      if (globals.chunk.tiles[tile].info.external_tile_mask[tile_bottom] == 1) {
        clover_unpack_bottom(globals, bottom_rcv_buffer, fields, tile, depth, bottom_top_offset);
      }
    }
  }
}

void clover_send_recv_message_left(global_variables &globals, clover::Buffer1D<double> &left_snd_buffer,
                                   clover::Buffer1D<double> &left_rcv_buffer, int total_size, int tag_send, int tag_recv,
                                   MPI_Request &req_send, MPI_Request &req_recv) {
  // First copy send buffer from device to host
  clover_send_recv_message(globals, chunk_left, left_snd_buffer, left_rcv_buffer, total_size, tag_send, tag_recv, req_send, req_recv);
}
void clover_send_recv_message_right(global_variables &globals, clover::Buffer1D<double> &right_snd_buffer,
                                    clover::Buffer1D<double> &right_rcv_buffer, int total_size, int tag_send, int tag_recv,
                                    MPI_Request &req_send, MPI_Request &req_recv) {
  // First copy send buffer from device to host
  clover_send_recv_message(globals, chunk_right, right_snd_buffer, right_rcv_buffer, total_size, tag_send, tag_recv, req_send, req_recv);
}
void clover_send_recv_message_top(global_variables &globals, clover::Buffer1D<double> &top_snd_buffer,
                                  clover::Buffer1D<double> &top_rcv_buffer, int total_size, int tag_send, int tag_recv,
                                  MPI_Request &req_send, MPI_Request &req_recv) {
  // First copy send buffer from device to host
  clover_send_recv_message(globals, chunk_top, top_snd_buffer, top_rcv_buffer, total_size, tag_send, tag_recv, req_send, req_recv);
}
void clover_send_recv_message_bottom(global_variables &globals, clover::Buffer1D<double> &bottom_snd_buffer,
                                     clover::Buffer1D<double> &bottom_rcv_buffer, int total_size, int tag_send, int tag_recv,
                                     MPI_Request &req_send, MPI_Request &req_recv) {
  // First copy send buffer from device to host
  clover_send_recv_message(globals, chunk_bottom, bottom_snd_buffer, bottom_rcv_buffer, total_size, tag_send, tag_recv, req_send, req_recv);
}

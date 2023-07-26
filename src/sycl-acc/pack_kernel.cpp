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

//  @brief Fortran mpi buffer packing kernel
//  @author Wayne Gaudin
//  @details Packs/unpacks mpi send and receive buffers

#include "pack_kernel.h"
#include "context.h"

void clover_pack_message_left(global_variables &global, int x_min, int x_max, int y_min, int y_max, clover::Buffer2D<double> &field_buffer,
                              clover::Buffer1D<double> &left_snd_buffer_buffer, int cell_data, int vertex_data, int x_face_data,
                              int y_face_data, int depth, int field_type, int buffer_offset) {

  // Pack

  int x_inc = 0, y_inc = 0;

  // These array modifications still need to be added on, plus the donor data location changes as in update_halo
  if (field_type == cell_data) {
    x_inc = 0;
    y_inc = 0;
  }
  if (field_type == vertex_data) {
    x_inc = 1;
    y_inc = 1;
  }
  if (field_type == x_face_data) {
    x_inc = 1;
    y_inc = 0;
  }
  if (field_type == y_face_data) {
    x_inc = 0;
    y_inc = 1;
  }

  // DO k=y_min-depth,y_max+y_inc+depth

  clover::execute(global.context.queue, [&](handler &h) {
    auto field = field_buffer.access<R>(h);
    auto left_snd_buffer = left_snd_buffer_buffer.access<W>(h);
    clover::par_ranged<class APPEND_LN(clover_pack_message_left)>(h, {y_min - depth + 1, y_max + y_inc + depth + 2}, [=](id<1> k) {
      for (int j = 0; j < depth; ++j) {
        size_t index = buffer_offset + j + k.get(0) * depth;
        left_snd_buffer[index] = field[x_min + x_inc - 1 + j + 2][k.get(0)];
      }
    });
  });
}

void clover_unpack_message_left(global_variables &global, int x_min, int x_max, int y_min, int y_max,
                                clover::Buffer2D<double> &field_buffer, clover::Buffer1D<double> &left_rcv_buffer_buffer, int cell_data,
                                int vertex_data, int x_face_data, int y_face_data, int depth, int field_type, int buffer_offset) {

  // Upnack

  int y_inc = 0;

  // These array modifications still need to be added on, plus the donor data location changes as in update_halo
  if (field_type == cell_data) {
    y_inc = 0;
  }
  if (field_type == vertex_data) {
    y_inc = 1;
  }
  if (field_type == x_face_data) {
    y_inc = 0;
  }
  if (field_type == y_face_data) {
    y_inc = 1;
  }

  // DO k=y_min-depth,y_max+y_inc+depth

  clover::execute(global.context.queue, [&](handler &h) {
    auto field = field_buffer.access<W>(h);
    auto left_rcv_buffer = left_rcv_buffer_buffer.access<R>(h);
    clover::par_ranged<class APPEND_LN(clover_unpack_message_left)>(h, {y_min - depth + 1, y_max + y_inc + depth + 2}, [=](id<1> k) {
      for (int j = 0; j < depth; ++j) {
        size_t index = buffer_offset + j + k.get(0) * depth;
        field[x_min - j][k.get(0)] = left_rcv_buffer[index];
      }
    });
  });
}

void clover_pack_message_right(global_variables &global, int x_min, int x_max, int y_min, int y_max, clover::Buffer2D<double> &field_buffer,
                               clover::Buffer1D<double> &right_snd_buffer_buffer, int cell_data, int vertex_data, int x_face_data,
                               int y_face_data, int depth, int field_type, int buffer_offset) {

  // Pack

  int y_inc = 0;

  // These array modifications still need to be added on, plus the donor data location changes as in update_halo
  if (field_type == cell_data) {
    y_inc = 0;
  }
  if (field_type == vertex_data) {
    y_inc = 1;
  }
  if (field_type == x_face_data) {
    y_inc = 0;
  }
  if (field_type == y_face_data) {
    y_inc = 1;
  }
  // DO k=y_min-depth,y_max+y_inc+depth
  clover::execute(global.context.queue, [&](handler &h) {
    auto field = field_buffer.access<R>(h);
    auto right_snd_buffer = right_snd_buffer_buffer.access<W>(h);
    clover::par_ranged<class APPEND_LN(clover_pack_message_right)>(h, {y_min - depth + 1, y_max + y_inc + depth + 2}, [=](id<1> k) {
      for (int j = 0; j < depth; ++j) {
        size_t index = buffer_offset + j + k.get(0) * depth;
        right_snd_buffer[index] = field[x_max + 1 - j][k.get(0)];
      }
    });
  });
}

void clover_unpack_message_right(global_variables &global, int x_min, int x_max, int y_min, int y_max,
                                 clover::Buffer2D<double> &field_buffer, clover::Buffer1D<double> &right_rcv_buffer_buffer, int cell_data,
                                 int vertex_data, int x_face_data, int y_face_data, int depth, int field_type, int buffer_offset) {

  // Upnack

  int x_inc = 0, y_inc = 0;

  // These array modifications still need to be added on, plus the donor data location changes as in update_halo
  if (field_type == cell_data) {
    x_inc = 0;
    y_inc = 0;
  }
  if (field_type == vertex_data) {
    x_inc = 1;
    y_inc = 1;
  }
  if (field_type == x_face_data) {
    x_inc = 1;
    y_inc = 0;
  }
  if (field_type == y_face_data) {
    x_inc = 0;
    y_inc = 1;
  }

  // DO k=y_min-depth,y_max+y_inc+depth

  clover::execute(global.context.queue, [&](handler &h) {
    auto field = field_buffer.access<W>(h);
    auto right_rcv_buffer = right_rcv_buffer_buffer.access<R>(h);
    clover::par_ranged<class APPEND_LN(clover_unpack_message_right)>(h, {y_min - depth + 1, y_max + y_inc + depth + 2}, [=](id<1> k) {
      for (int j = 0; j < depth; ++j) {
        size_t index = buffer_offset + j + k.get(0) * depth;
        field[x_max + x_inc + j + 2][k.get(0)] = right_rcv_buffer[index];
      }
    });
  });
}

void clover_pack_message_top(global_variables &global, int x_min, int x_max, int y_min, int y_max, clover::Buffer2D<double> &field_buffer,
                             clover::Buffer1D<double> &top_snd_buffer_buffer, int cell_data, int vertex_data, int x_face_data,
                             int y_face_data, int depth, int field_type, int buffer_offset) {

  // Pack

  int x_inc = 0;

  // These array modifications still need to be added on, plus the donor data location changes as in update_halo
  if (field_type == cell_data) {
    x_inc = 0;
  }
  if (field_type == vertex_data) {
    x_inc = 1;
  }
  if (field_type == x_face_data) {
    x_inc = 1;
  }
  if (field_type == y_face_data) {
    x_inc = 0;
  }

  for (int k = 0; k < depth; ++k) {
    // DO j=x_min-depth,x_max+x_inc+depth
    clover::execute(global.context.queue, [&](handler &h) {
      auto field = field_buffer.access<R>(h);
      auto top_snd_buffer = top_snd_buffer_buffer.access<W>(h);
      clover::par_ranged<class APPEND_LN(clover_pack_message_top)>(h, {x_min - depth + 1, x_max + x_inc + depth + 2}, [=](id<1> j) {
        size_t index = buffer_offset + k + j.get(0) * depth;
        top_snd_buffer[index] = field[j.get(0)][y_max + 1 - k];
      });
    });
  }
}

void clover_unpack_message_top(global_variables &global, int x_min, int x_max, int y_min, int y_max, clover::Buffer2D<double> &field_buffer,
                               clover::Buffer1D<double> &top_rcv_buffer_buffer, int cell_data, int vertex_data, int x_face_data,
                               int y_face_data, int depth, int field_type, int buffer_offset) {

  // Unpack

  int x_inc = 0, y_inc = 0;

  // These array modifications still need to be added on, plus the donor data location changes as in update_halo
  if (field_type == cell_data) {
    x_inc = 0;
    y_inc = 0;
  }
  if (field_type == vertex_data) {
    x_inc = 1;
    y_inc = 1;
  }
  if (field_type == x_face_data) {
    x_inc = 1;
    y_inc = 0;
  }
  if (field_type == y_face_data) {
    x_inc = 0;
    y_inc = 1;
  }

  for (int k = 0; k < depth; ++k) {
    // DO j=x_min-depth,x_max+x_inc+depth

    clover::execute(global.context.queue, [&](handler &h) {
      auto field = field_buffer.access<W>(h);
      auto top_rcv_buffer = top_rcv_buffer_buffer.access<R>(h);
      clover::par_ranged<class APPEND_LN(clover_unpack_message_top)>(h, {x_min - depth + 1, x_max + x_inc + depth + 2}, [=](id<1> j) {
        size_t index = buffer_offset + k + j.get(0) * depth;
        field[j.get(0)][y_max + y_inc + k + 2] = top_rcv_buffer[index];
      });
    });
  }
}

void clover_pack_message_bottom(global_variables &global, int x_min, int x_max, int y_min, int y_max,
                                clover::Buffer2D<double> &field_buffer, clover::Buffer1D<double> &bottom_snd_buffer_buffer, int cell_data,
                                int vertex_data, int x_face_data, int y_face_data, int depth, int field_type, int buffer_offset) {

  // Pack

  int x_inc = 0, y_inc = 0;

  // These array modifications still need to be added on, plus the donor data location changes as in update_halo
  if (field_type == cell_data) {
    x_inc = 0;
    y_inc = 0;
  }
  if (field_type == vertex_data) {
    x_inc = 1;
    y_inc = 1;
  }
  if (field_type == x_face_data) {
    x_inc = 1;
    y_inc = 0;
  }
  if (field_type == y_face_data) {
    x_inc = 0;
    y_inc = 1;
  }

  for (int k = 0; k < depth; ++k) {
    // DO j=x_min-depth,x_max+x_inc+depth

    clover::execute(global.context.queue, [&](handler &h) {
      auto field = field_buffer.access<R>(h);
      auto bottom_snd_buffer = bottom_snd_buffer_buffer.access<W>(h);
      clover::par_ranged<class APPEND_LN(clover_pack_message_bottom)>(h, {x_min - depth + 1, x_max + x_inc + depth + 2}, [=](id<1> j) {
        size_t index = buffer_offset + k + j.get(0) * depth;
        bottom_snd_buffer[index] = field[j.get(0)][y_min + y_inc - 1 + k + 2];
      });
    });
  }
}

void clover_unpack_message_bottom(global_variables &global, int x_min, int x_max, int y_min, int y_max,
                                  clover::Buffer2D<double> &field_buffer, clover::Buffer1D<double> &bottom_rcv_buffer_buffer, int cell_data,
                                  int vertex_data, int x_face_data, int y_face_data, int depth, int field_type, int buffer_offset) {

  // Unpack

  int x_inc = 0;

  // These array modifications still need to be added on, plus the donor data location changes as in update_halo
  if (field_type == cell_data) {
    x_inc = 0;
  }
  if (field_type == vertex_data) {
    x_inc = 1;
  }
  if (field_type == x_face_data) {
    x_inc = 1;
  }
  if (field_type == y_face_data) {
    x_inc = 0;
  }

  for (int k = 0; k < depth; ++k) {
    // DO j=x_min-depth,x_max+x_inc+depth
    clover::execute(global.context.queue, [&](handler &h) {
      auto field = field_buffer.access<W>(h);
      auto bottom_rcv_buffer = bottom_rcv_buffer_buffer.access<R>(h);
      clover::par_ranged<class APPEND_LN(clover_unpack_message_top)>(h, {x_min - depth + 1, x_max + x_inc + depth + 2}, [=](id<1> j) {
        size_t index = buffer_offset + k + j.get(0) * depth;
        field[j.get(0)][y_min - k] = bottom_rcv_buffer[index];
      });
    });
  }
}

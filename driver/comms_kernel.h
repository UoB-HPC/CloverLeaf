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

#pragma once

#include "comms.h"
#include "context.h"
#include "definitions.h"

void clover_allocate_buffers(global_variables &globals, parallel_ &parallel);

void clover_exchange(global_variables &globals, const int fields[NUM_FIELDS], int depth);

void clover_send_recv_message_left(global_variables &globals, clover::StagingBuffer1D<double> left_snd_buffer,
                                   clover::StagingBuffer1D<double> left_rcv_buffer, int total_size, int tag_send, int tag_recv,
                                   MPI_Request &req_send, MPI_Request &req_recv);
void clover_send_recv_message_right(global_variables &globals, clover::StagingBuffer1D<double> right_snd_buffer,
                                    clover::StagingBuffer1D<double> right_rcv_buffer, int total_size, int tag_send, int tag_recv,
                                    MPI_Request &req_send, MPI_Request &req_recv);
void clover_send_recv_message_top(global_variables &globals, clover::StagingBuffer1D<double> top_snd_buffer,
                                  clover::StagingBuffer1D<double> top_rcv_buffer, int total_size, int tag_send, int tag_recv,
                                  MPI_Request &req_send, MPI_Request &req_recv);
void clover_send_recv_message_bottom(global_variables &globals, clover::StagingBuffer1D<double> bottom_snd_buffer,
                                     clover::StagingBuffer1D<double> top_rcv_buffer, int total_size, int tag_send, int tag_recv,
                                     MPI_Request &req_send, MPI_Request &req_recv);
